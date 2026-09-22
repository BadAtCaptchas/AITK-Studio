# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Ant Group. All rights reserved.
# Copyright contributors to the vLLM-Omni project.
# Adapted from the pinned sources documented in UPSTREAM.md.
"""Frozen Ming text/image conditioning with bounded GPU residency.

CPU master components are loaded once. A call stages vision, one Bailing block
at a time, then the connector. Parent ``.to('cuda')`` calls change the execution
device without accidentally transferring the complete 16B MoE to the GPU.
"""

from contextlib import contextmanager
import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model, Qwen2VLImageProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

from .bailing import BailingBlock, BailingRMSNorm, video_rotary_embeddings


def ming_positions(length: int, reference_start: int | None = None, reference_grid=None) -> torch.Tensor:
    """Uncentered reference grids; every other token, including queries, is text.

    ``reference_start`` is the index of the first *patch* token, excluding the
    image-start delimiter. Grid H/W are already divided by spatial merge size.
    """
    if reference_start is None:
        return torch.arange(length).unsqueeze(0).expand(3, -1).clone()
    if reference_grid is None or len(reference_grid) != 3:
        raise ValueError("Ming reference positions require a T/H/W grid")
    frames, height, width = (int(value) for value in reference_grid)
    count = frames * height * width
    if min(frames, height, width) < 1 or reference_start < 0 or reference_start + count > length:
        raise ValueError("Ming reference patch grid lies outside the prompt")
    before = torch.arange(reference_start).unsqueeze(0).expand(3, -1)
    grid = torch.stack(torch.meshgrid(torch.arange(frames), torch.arange(height), torch.arange(width), indexing="ij")).reshape(3, -1)
    grid = grid + reference_start
    end = reference_start + count
    after = torch.arange(length - end).unsqueeze(0).expand(3, -1) + int(grid.max()) + 1
    return torch.cat((before, grid, after), dim=1)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _checkpoint_index(folder: Path) -> dict[str, str]:
    index_path = folder / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json(index_path).get("weight_map")
        if not isinstance(index, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in index.items()):
            raise ValueError(f"Invalid safetensors index: {index_path}")
        for filename in set(index.values()):
            if Path(filename).name != filename or not (folder / filename).is_file():
                raise ValueError(f"Checkpoint shard must exist within {folder}: {filename}")
        return index
    single = folder / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"Ming conditioner weights missing from {folder}")
    with safe_open(single, framework="pt", device="cpu") as checkpoint:
        return {key: single.name for key in checkpoint.keys()}


def _load_prefixed(module: nn.Module, folder: Path, index: dict[str, str], prefix: str, dtype: torch.dtype) -> None:
    expected = set(module.state_dict())
    wanted = {prefix + key: key for key in expected}
    missing = wanted.keys() - index.keys()
    if missing:
        raise ValueError(f"Ming checkpoint is missing required weights: {sorted(missing)[:8]}")
    unexpected = {key for key in index if key.startswith(prefix)} - wanted.keys()
    if unexpected:
        raise ValueError(f"Unexpected Ming component weights under {prefix}: {sorted(unexpected)[:8]}")
    state = {}
    for filename in sorted({index[key] for key in wanted}):
        with safe_open(folder / filename, framework="pt", device="cpu") as checkpoint:
            for source, target in wanted.items():
                if index[source] == filename:
                    value = checkpoint.get_tensor(source)
                    state[target] = value.to(dtype=dtype) if value.is_floating_point() else value
    module.load_state_dict(state, strict=True, assign=True)
    module.eval().requires_grad_(False)


class MingConditioner(nn.Module):
    """Frozen native conditioner. ``encode`` returns unbatched final DiT features."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_placement", torch.zeros((), dtype=torch.bfloat16), persistent=False)
        self._execution_device = torch.device("cpu")
        self._compute_dtype = torch.bfloat16
        self._lock = RLock()
        self.offload = True

    @property
    def device(self) -> torch.device:
        return self._execution_device

    @property
    def dtype(self) -> torch.dtype:
        return self._compute_dtype

    def _apply(self, fn, recurse=True):
        # BaseModel/trainer/sample coordinators call .to() on text encoders.
        # Apply that operation to a scalar only; all large child modules keep
        # their CPU residency until the explicit stage context below.
        probe = fn(self._placement)
        self._execution_device = probe.device
        self._compute_dtype = probe.dtype
        if probe.device.type == "meta":
            # MemoryManager.free deliberately destroys the encoder after
            # caching. Honor that operation recursively so CPU masters are
            # released rather than kept alive behind the placement shell.
            return super()._apply(fn, recurse=recurse)
        self._buffers["_placement"] = probe.to("cpu")
        return self

    @contextmanager
    def _stage(self, *modules):
        try:
            for module in modules:
                module.to(device=self.device, dtype=self.dtype)
            yield
        finally:
            # Always return to CPU, including after cancellation/error. The
            # offload=False option still stages components sequentially; it
            # never permits a hidden whole-MoE GPU allocation.
            for module in modules:
                module.to("cpu")

    @classmethod
    def load(cls, modelroot: str | Path, device="cpu", dtype=torch.bfloat16, offload=True):
        root = Path(modelroot)
        if not root.is_dir():
            root = Path(snapshot_download(str(modelroot), allow_patterns=["mllm/*", "mlp/*", "connector/*"]))
        holder = cls()
        holder._execution_device = torch.device(device)
        holder._compute_dtype = dtype
        holder.offload = bool(offload)
        mllm = _read_json(root / "mllm" / "config.json")
        mlp = _read_json(root / "mlp" / "config.json")
        if (mlp.get("selected_hidden_states_layers") != [5, 12, 20]
                or mlp.get("img_gen_scales") != [16]
                or not all(mlp.get(key) is True for key in
                    ("use_identity_mlp", "use_learnable_token_condition", "use_vlm_directvlm_condition"))):
            raise ValueError("Unsupported Ming conditioner configuration; expected both released conditioning streams")
        holder.connector_norm = bool(mlp.get("connector_norm", False))
        holder.text_encoder_norm = bool(mlp.get("text_encoder_norm", False))
        holder.config = SimpleNamespace(**mllm["llm_config"])
        config = holder.config
        if config.hidden_size != 2048 or config.num_hidden_layers != 20:
            raise ValueError("Ming conditioner requires its 2048-wide, 20-layer Bailing backbone")
        if config.hidden_act != "silu" or config.router_type != "MultiRouter":
            raise ValueError("Unsupported Ming Bailing activation or router")
        if config.rope_scaling.get("type") != "video_rope":
            raise ValueError("Ming requires the Bailing video_rope convention")
        holder.selected_layers = tuple(mlp["selected_hidden_states_layers"])
        holder.tokenizer = AutoTokenizer.from_pretrained(root / "mllm", local_files_only=True)
        holder.image_processor = Qwen2VLImageProcessor.from_pretrained(root / "mllm", local_files_only=True)
        index = _checkpoint_index(root / "mllm")
        vision_config = Qwen2_5_VLVisionConfig(**mllm["vision_config"])
        vision_config._attn_implementation = "sdpa"
        with init_empty_weights():
            holder.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            holder.blocks = nn.ModuleList([BailingBlock(config, i) for i in range(config.num_hidden_layers)])
            holder.final_norm = BailingRMSNorm(config.hidden_size, config.rms_norm_eps)
            holder.vision = Qwen2_5_VisionTransformerPretrainedModel(vision_config)
            holder.linear_proj = nn.Sequential(nn.Linear(vision_config.out_hidden_size, config.hidden_size),
                nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size))
        _load_prefixed(holder.word_embeddings, root / "mllm", index, "model.model.word_embeddings.", dtype)
        for i, block in enumerate(holder.blocks):
            _load_prefixed(block, root / "mllm", index, f"model.model.layers.{i}.", dtype)
        _load_prefixed(holder.final_norm, root / "mllm", index, "model.model.norm.", dtype)
        _load_prefixed(holder.vision, root / "mllm", index, "vision.", dtype)
        _load_prefixed(holder.linear_proj, root / "mllm", index, "linear_proj.", dtype)
        connector_config = Qwen2Config.from_pretrained(root / "connector", local_files_only=True)
        connector_config._attn_implementation = "sdpa"
        holder.connector = Qwen2Model.from_pretrained(root / "connector", config=connector_config,
            dtype=dtype, local_files_only=True)
        for block in holder.connector.layers:
            block.self_attn.is_causal = False
        holder.proj_in = nn.Linear(config.hidden_size, connector_config.hidden_size)
        holder.proj_out = nn.Linear(connector_config.hidden_size, int(mlp["diffusion_c_input_dim"]))
        holder.direct_projector = nn.Sequential(nn.RMSNorm(config.hidden_size * 3, eps=1e-5),
            nn.Linear(config.hidden_size * 3, int(mlp["diffusion_inner_dim"])))
        state = load_file(root / "mlp" / "model.safetensors", device="cpu")
        holder.query_tokens = nn.Parameter(state["query_tokens_dict.16x16"].to(dtype), requires_grad=False)
        if holder.query_tokens.shape != (256, config.hidden_size):
            raise ValueError("Ming must provide 256 learned image-query tokens")
        for module, prefix in ((holder.proj_in, "proj_in."), (holder.proj_out, "proj_out."),
                               (holder.direct_projector, "proj_directvlm.")):
            module.load_state_dict({key: state[prefix + key].to(dtype) for key in module.state_dict()}, strict=True, assign=True)
        holder.eval().requires_grad_(False)
        return holder

    @torch.no_grad()
    def encode(self, prompt: str, reference: Image.Image | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(prompt, str):
            raise TypeError("Ming caption must be a string")
        if any(token in prompt for token in ("<IMAGE>", "<image>", "<imagePatch>", "</image>")):
            raise ValueError("Ming captions cannot include reserved image placeholder tokens")
        with self._lock:
            return self._encode(prompt, reference)

    def _encode(self, prompt, reference):
        config = self.config
        image_features = None
        grid = None
        if reference is not None:
            pixels = self.image_processor(images=[reference.convert("RGB")], return_tensors="pt")
            grid_raw = pixels["image_grid_thw"]
            if grid_raw.shape != (1, 3):
                raise ValueError("Ming accepts exactly one reference image")
            merge = int(self.image_processor.merge_size)
            grid = (int(grid_raw[0, 0]), int(grid_raw[0, 1]) // merge, int(grid_raw[0, 2]) // merge)
            with self._stage(self.vision, self.linear_proj):
                features = self.vision(pixels["pixel_values"].to(device=self.device, dtype=self.dtype),
                    grid_thw=grid_raw.to(self.device)).pooler_output
                image_features = F.normalize(self.linear_proj(features), dim=-1)
            patch_count = grid[0] * grid[1] * grid[2]
            if len(image_features) != patch_count:
                raise ValueError("Ming vision features do not match reference tokens")
            prompt = "<image>" + "<imagePatch>" * patch_count + "</image>\n" + prompt
        else:
            patch_count = 0
        query_suffix = "<image>" + "<imagePatch>" * 256 + "</image>"
        ids = self.tokenizer(prompt + query_suffix, return_tensors="pt")["input_ids"]
        token_ids = ids[0]
        expected = [config.image_start_token] + [config.image_patch_token] * 256 + [config.image_end_token]
        if token_ids[-258:].tolist() != expected:
            raise ValueError("Ming tokenizer did not preserve the learned-query suffix")
        if ids.shape[1] > config.max_position_embeddings:
            raise ValueError(f"Ming caption/reference exceeds {config.max_position_embeddings} tokens")
        direct_end = ids.shape[1] - 258
        patch_positions = (token_ids == config.image_patch_token).nonzero().flatten()
        if len(patch_positions) != patch_count + 256:
            raise ValueError("Ming image token count does not match the reference and query")
        reference_start = int(patch_positions[0]) if patch_count else None
        positions = ming_positions(ids.shape[1], reference_start, grid).to(self.device)
        ids = ids.to(self.device)
        with self._stage(self.word_embeddings):
            hidden = self.word_embeddings(ids)
        if image_features is not None:
            hidden[0, patch_positions[:patch_count].to(self.device)] = image_features
            del image_features
        hidden[0, -257:-1] = self.query_tokens.to(device=self.device, dtype=self.dtype)
        image_mask = ids == config.image_patch_token
        rotary = video_rotary_embeddings(positions, config, self.dtype)
        captured = {}
        for index, block in enumerate(self.blocks):
            # HF hidden_states[k] is the residual stream before block k,
            # except the final index, which includes the final RMSNorm.
            if index in self.selected_layers:
                captured[index] = hidden[:, :direct_end].clone()
            with self._stage(block):
                hidden = block(hidden, rotary, image_mask)
        with self._stage(self.final_norm):
            hidden = self.final_norm(hidden)
        captured[len(self.blocks)] = hidden[:, :direct_end]
        query = hidden[:, -257:-1].contiguous()
        direct = torch.cat([captured[index] for index in self.selected_layers], dim=-1)
        del hidden, captured, rotary
        with self._stage(self.proj_in, self.connector, self.proj_out, self.direct_projector):
            query = self.proj_in(query)
            # An explicit additive all-zero mask enables bidirectional
            # attention under Transformers 5, bypassing its causal mask.
            mask = torch.zeros((1, 1, query.shape[1], query.shape[1]), device=self.device, dtype=query.dtype)
            query = self.connector(inputs_embeds=query, attention_mask=mask, use_cache=False).last_hidden_state
            query = self.proj_out(query)
            if self.connector_norm:
                query = F.normalize(query, dim=-1)
            if self.text_encoder_norm:
                query = query * 1000.0
            direct = self.direct_projector(direct)
        return query[0].detach(), direct[0].detach()
