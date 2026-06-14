import json
import os
import warnings
from importlib import metadata
from posixpath import dirname as _posix_dirname, join as _posix_join
from typing import List, Optional

import torch
import torch.nn as nn
import yaml
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.memory_management import attach_layer_offloading
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

from .src.caption_verifier import CaptionVerifier
from .src.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
)
from .src.latent_norm import get_latent_norm
from .src.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from .src.pipeline_ideogram4 import (
    Ideogram4Pipeline,
    Ideogram4PipelineConfig,
    _build_transformer,
    _load_autoencoder,
    _load_indexed_or_single_state_dict,
)
from .src.quantized_loading import (
    FP8_TEXT_ENCODER_CONFIG_FLAG,
    Fp8Linear,
    _move_non_meta_tensors_to_device,
    is_comfy_quant_state_dict,
    is_fp8_state_dict,
    is_nvfp4_state_dict,
    load_comfy_quant_state_dict,
    load_fp8_state_dict,
    swap_linears_to_comfy_quant,
    swap_linears_to_fp8,
)
from .src.sampler_configs import PRESETS
from .src.scheduler import get_schedule_for_resolution, make_step_intervals


IDEOGRAM4_NF4_REPO = "ideogram-ai/ideogram-4-nf4"
IDEOGRAM4_FP8_REPO = "ideogram-ai/ideogram-4-fp8"
IDEOGRAM4_COMFY_REPO = "Comfy-Org/Ideogram-4"
IDEOGRAM4_TORCH_MIN = (2, 11)
IDEOGRAM4_COMFY_FILES = {
    "nvfp4": {
        "conditional": "diffusion_models/ideogram4_nvfp4_mixed.safetensors",
        "unconditional": "diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors",
        "text_encoder": "text_encoders/qwen3vl_8b_nvfp4.safetensors",
        "vae": "vae/flux2-vae.safetensors",
    },
    "fp8": {
        "conditional": "diffusion_models/ideogram4_fp8_scaled.safetensors",
        "unconditional": "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
        "text_encoder": "text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
        "vae": "vae/flux2-vae.safetensors",
    },
}

scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def _version_tuple(version: str) -> tuple[int, ...]:
    pieces = []
    for part in version.split("+", 1)[0].split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits == "":
            break
        pieces.append(int(digits))
    return tuple(pieces)


def infer_ideogram4_quantization(name_or_path: Optional[str], model_kwargs=None) -> str:
    model_kwargs = model_kwargs or {}
    explicit = model_kwargs.get("quantization", None)
    if explicit is not None and str(explicit).lower() != "auto":
        quantization = str(explicit).lower()
    else:
        path = (name_or_path or "").lower()
        if "nvfp4" in path or path.rstrip("/") == IDEOGRAM4_COMFY_REPO.lower():
            quantization = "nvfp4"
        else:
            quantization = "fp8" if "fp8" in path else "nf4"
    if quantization not in {"nf4", "fp8", "nvfp4"}:
        raise ValueError(
            "Ideogram 4 quantization must be 'nf4', 'fp8', 'nvfp4', or "
            f"'auto', got {quantization!r}"
        )
    return quantization


def default_repo_for_quantization(quantization: str) -> str:
    if quantization == "fp8":
        return IDEOGRAM4_FP8_REPO
    if quantization == "nvfp4":
        return IDEOGRAM4_COMFY_REPO
    return IDEOGRAM4_NF4_REPO


def patchify_latents(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"latent height/width must be divisible by patch_size={patch_size}, "
            f"got {(height, width)}"
        )
    patch_h = patch_w = patch_size
    grid_h = height // patch_h
    grid_w = width // patch_w
    latents = latents.view(
        batch,
        channels,
        grid_h,
        patch_h,
        grid_w,
        patch_w,
    )
    latents = latents.permute(0, 3, 5, 1, 2, 4).contiguous()
    return latents.view(
        batch,
        patch_h * patch_w * channels,
        grid_h,
        grid_w,
    )


def unpatchify_latents(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    if channels % (patch_size * patch_size) != 0:
        raise ValueError(
            f"latent channels must be divisible by patch_size**2={patch_size * patch_size}, "
            f"got {channels}"
        )
    patch_h = patch_w = patch_size
    ae_channels = channels // (patch_h * patch_w)
    latents = latents.view(batch, patch_h, patch_w, ae_channels, height, width)
    latents = latents.permute(0, 3, 4, 1, 5, 2).contiguous()
    return latents.view(batch, ae_channels, height * patch_h, width * patch_w)


def pack_latent_tokens(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    return latents.view(batch, channels, height * width).permute(0, 2, 1).contiguous()


def unpack_latent_tokens(
    tokens: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    batch, token_count, channels = tokens.shape
    if token_count != height * width:
        raise ValueError(
            f"token count {token_count} does not match latent grid {height}x{width}"
        )
    return tokens.permute(0, 2, 1).contiguous().view(batch, channels, height, width)


def module_device(module: nn.Module) -> torch.device:
    fallback = None
    for tensor in list(module.parameters()) + list(module.buffers()):
        fallback = tensor.device
        if not tensor.is_meta:
            return tensor.device
    if fallback is not None:
        return fallback
    return torch.device("cpu")


def move_module_to_device(module: nn.Module, device: torch.device) -> None:
    if any(
        tensor.is_meta
        for tensor in list(module.parameters()) + list(module.buffers())
    ):
        _move_non_meta_tensors_to_device(module, device)
    else:
        module.to(device)


def dequantize_fp8_linears(
    module: nn.Module, dtype: torch.dtype, device: torch.device
) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, Fp8Linear):
            linear = nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                dtype=dtype,
                device=device,
            )
            weight = child.weight.to(device=device, dtype=dtype)
            scale = child.weight_scale.to(device=device, dtype=dtype)
            while scale.dim() < weight.dim():
                scale = scale.unsqueeze(-1)
            linear.weight.data.copy_(weight * scale)
            if child.bias is not None:
                linear.bias.data.copy_(child.bias.to(device=device, dtype=dtype))
            setattr(module, name, linear)
            replaced += 1
        else:
            replaced += dequantize_fp8_linears(child, dtype=dtype, device=device)
    return replaced


def _load_local_state_dict(
    root: str, index_filename: str
) -> dict[str, torch.Tensor]:
    if index_filename.endswith(".safetensors"):
        single_path = os.path.join(root, index_filename.replace("/", os.sep))
        if not os.path.exists(single_path):
            raise FileNotFoundError(f"Could not find Ideogram state dict at {single_path}")
        return load_file(single_path)

    index_path = os.path.join(root, index_filename.replace("/", os.sep))
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]
        shard_dir = os.path.dirname(index_path)
        state_dict: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            state_dict.update(load_file(os.path.join(shard_dir, shard)))
        return state_dict

    single_filename = index_filename.removesuffix(".index.json")
    single_path = os.path.join(root, single_filename.replace("/", os.sep))
    if not os.path.exists(single_path):
        raise FileNotFoundError(f"Could not find Ideogram state dict at {single_path}")
    return load_file(single_path)


def _load_subfolder_state_dict_local_or_hf(
    repo_or_dir: str,
    subfolder: str,
    basename: str,
    status_callback=None,
) -> dict[str, torch.Tensor]:
    prefix = f"{subfolder}/" if subfolder else ""
    index_filename = f"{prefix}{basename}.safetensors.index.json"
    shard_dir = _posix_dirname(index_filename)

    if os.path.isdir(repo_or_dir):
        index_path = os.path.join(repo_or_dir, index_filename.replace("/", os.sep))
        single_path = os.path.join(
            repo_or_dir, f"{prefix}{basename}.safetensors".replace("/", os.sep)
        )
    else:
        try:
            if status_callback is not None:
                status_callback("Looking up Qwen3-VL text encoder weight index")
            index_path = hf_hub_download(repo_id=repo_or_dir, filename=index_filename)
        except EntryNotFoundError:
            single_filename = f"{prefix}{basename}.safetensors"
            if status_callback is not None:
                status_callback(
                    "Downloading/checking Qwen3-VL text encoder weights: "
                    f"{os.path.basename(single_filename)}"
                )
            single_path = hf_hub_download(
                repo_id=repo_or_dir, filename=single_filename
            )
            if status_callback is not None:
                status_callback(
                    f"Loading Qwen3-VL text encoder weights: {os.path.basename(single_path)}"
                )
            return load_file(single_path)
        single_path = None

    if not os.path.exists(index_path):
        if not os.path.exists(single_path):
            raise FileNotFoundError(
                f"Could not find Qwen3-VL text encoder weights at {single_path}"
            )
        if status_callback is not None:
            status_callback(
                f"Loading Qwen3-VL text encoder weights: {os.path.basename(single_path)}"
            )
        return load_file(single_path)

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]
    shard_filenames = sorted(set(weight_map.values()))

    state_dict: dict[str, torch.Tensor] = {}
    for idx, shard in enumerate(shard_filenames, start=1):
        if os.path.isdir(repo_or_dir):
            shard_path = os.path.join(
                os.path.dirname(index_path), shard.replace("/", os.sep)
            )
        else:
            shard_repo_path = _posix_join(shard_dir, shard) if shard_dir else shard
            if status_callback is not None:
                status_callback(
                    f"Downloading/checking Qwen3-VL text encoder shard "
                    f"{idx}/{len(shard_filenames)}: {os.path.basename(shard)}"
                )
            shard_path = hf_hub_download(repo_id=repo_or_dir, filename=shard_repo_path)
        if status_callback is not None:
            status_callback(
                f"Loading Qwen3-VL text encoder shard {idx}/{len(shard_filenames)}: "
                f"{os.path.basename(shard_path)}"
            )
        state_dict.update(load_file(shard_path))
    return state_dict


def _load_named_state_dict_local_or_hf(
    repo_or_dir: str,
    filename: str,
    *,
    status_callback=None,
    label: str = "weights",
) -> dict[str, torch.Tensor]:
    if os.path.isdir(repo_or_dir):
        path = os.path.join(repo_or_dir, filename.replace("/", os.sep))
    else:
        if status_callback is not None:
            status_callback(f"Downloading/checking {label}: {os.path.basename(filename)}")
        path = hf_hub_download(repo_id=repo_or_dir, filename=filename)
    if status_callback is not None:
        status_callback(f"Loading {label}: {os.path.basename(path)}")
    return load_file(path)


def _load_component_state_dict(
    repo_or_dir: str, index_filename: str
) -> dict[str, torch.Tensor]:
    if os.path.isdir(repo_or_dir):
        return _load_local_state_dict(repo_or_dir, index_filename)
    return _load_indexed_or_single_state_dict(repo_or_dir, index_filename)


def _load_autoencoder_local_or_hf(
    repo_or_dir: str,
    autoencoder_filename: str,
    device: torch.device,
    dtype: torch.dtype,
):
    if os.path.isdir(repo_or_dir):
        weights_path = os.path.join(repo_or_dir, autoencoder_filename.replace("/", os.sep))
    else:
        weights_path = hf_hub_download(repo_id=repo_or_dir, filename=autoencoder_filename)
    return _load_autoencoder(weights_path, device, dtype)


def _load_qwen3_vl_local_or_hf(
    repo_or_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    tokenizer_subfolder: str,
    text_encoder_subfolder: str,
    status_callback=None,
):
    if status_callback is not None:
        status_callback("Loading Qwen3-VL tokenizer")
    if os.path.isdir(repo_or_dir):
        tokenizer_path = os.path.join(repo_or_dir, tokenizer_subfolder)
        text_encoder_path = os.path.join(repo_or_dir, text_encoder_subfolder)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        config_path = os.path.join(text_encoder_path, "config.json")
        model_source_kwargs = {"pretrained_model_name_or_path": text_encoder_path}
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            repo_or_dir, subfolder=tokenizer_subfolder
        )
        if status_callback is not None:
            status_callback("Downloading/checking Qwen3-VL text encoder config")
        config_path = hf_hub_download(
            repo_id=repo_or_dir, filename=f"{text_encoder_subfolder}/config.json"
        )
        model_source_kwargs = {
            "pretrained_model_name_or_path": repo_or_dir,
            "subfolder": text_encoder_subfolder,
        }

    if status_callback is not None:
        status_callback("Reading Qwen3-VL text encoder config")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    is_quantized = "quantization_config" in cfg_data
    is_fp8 = bool(cfg_data.get(FP8_TEXT_ENCODER_CONFIG_FLAG, False))

    if is_fp8:
        if status_callback is not None:
            status_callback("Loading Qwen3-VL FP8 text encoder state dict")
        state_dict = _load_subfolder_state_dict_local_or_hf(
            repo_or_dir,
            text_encoder_subfolder,
            "model",
            status_callback=status_callback,
        )
        if status_callback is not None:
            status_callback("Building Qwen3-VL text encoder structure")
        config = AutoConfig.from_pretrained(
            **model_source_kwargs, trust_remote_code=True
        )
        from .src.quantized_loading import load_fp8_state_dict, swap_linears_to_fp8

        with init_empty_weights():
            model = AutoModel.from_config(config, trust_remote_code=True)
        if status_callback is not None:
            status_callback("Swapping Qwen3-VL linear layers to FP8")
        swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
        if status_callback is not None:
            status_callback(f"Materializing Qwen3-VL FP8 weights on {device}")
        load_fp8_state_dict(
            model, state_dict, device=device, dtype=dtype, assign=True, strict=False
        )
        del state_dict
    elif is_quantized:
        if status_callback is not None:
            status_callback("Loading quantized Qwen3-VL text encoder with Transformers")
        model = AutoModel.from_pretrained(
            **model_source_kwargs,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map={"": device},
        )
    else:
        if status_callback is not None:
            status_callback("Loading Qwen3-VL text encoder with Transformers")
        model = AutoModel.from_pretrained(
            **model_source_kwargs, torch_dtype=dtype, trust_remote_code=True
        )
        if status_callback is not None:
            status_callback(f"Moving Qwen3-VL text encoder to {device}")
        model.to(device)
    model.eval()
    return tokenizer, model


def _remap_comfy_qwen3_vl_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model.visual."):
            key = "visual." + key[len("model.visual."):]
        elif key.startswith("model."):
            key = "language_model." + key[len("model."):]
        remapped[key] = value
    return remapped


def _load_qwen3_vl_comfy_local_or_hf(
    config_repo_or_dir: str,
    weights_repo_or_dir: str,
    weights_filename: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    tokenizer_subfolder: str,
    text_encoder_subfolder: str,
    status_callback=None,
):
    if status_callback is not None:
        status_callback("Loading Qwen3-VL tokenizer/config source")
    if os.path.isdir(config_repo_or_dir):
        tokenizer_path = os.path.join(config_repo_or_dir, tokenizer_subfolder)
        text_encoder_path = os.path.join(config_repo_or_dir, text_encoder_subfolder)
        if not os.path.exists(tokenizer_path):
            tokenizer_path = config_repo_or_dir
        if not os.path.exists(os.path.join(text_encoder_path, "config.json")):
            text_encoder_path = config_repo_or_dir
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model_source_kwargs = {"pretrained_model_name_or_path": text_encoder_path}
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config_repo_or_dir, subfolder=tokenizer_subfolder
        )
        model_source_kwargs = {
            "pretrained_model_name_or_path": config_repo_or_dir,
            "subfolder": text_encoder_subfolder,
        }

    if status_callback is not None:
        status_callback("Loading Comfy Qwen3-VL text encoder state dict")
    state_dict = _load_named_state_dict_local_or_hf(
        weights_repo_or_dir,
        weights_filename,
        status_callback=status_callback,
        label="Comfy Qwen3-VL text encoder weights",
    )

    if status_callback is not None:
        status_callback("Building Qwen3-VL text encoder structure")
    config = AutoConfig.from_pretrained(**model_source_kwargs, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModel.from_config(config, trust_remote_code=True)
    state_dict = _remap_comfy_qwen3_vl_state_dict(state_dict)

    if is_comfy_quant_state_dict(state_dict) or is_nvfp4_state_dict(state_dict):
        if status_callback is not None:
            status_callback("Swapping Qwen3-VL linear layers to Comfy quantized modules")
        swap_linears_to_comfy_quant(model, state_dict, compute_dtype=dtype)
        if status_callback is not None:
            status_callback(f"Materializing Comfy Qwen3-VL weights on {device}")
        load_comfy_quant_state_dict(
            model,
            state_dict,
            device=device,
            dtype=dtype,
            assign=True,
            strict=False,
        )
    elif is_fp8_state_dict(state_dict):
        if status_callback is not None:
            status_callback("Swapping Qwen3-VL linear layers to FP8")
        swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
        if status_callback is not None:
            status_callback(f"Materializing Qwen3-VL FP8 weights on {device}")
        load_fp8_state_dict(
            model, state_dict, device=device, dtype=dtype, assign=True, strict=False
        )
    else:
        prepared = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
            for k, v in state_dict.items()
        }
        model.load_state_dict(prepared, strict=False, assign=True)
        model.to(device)
    del state_dict
    model.eval()
    return tokenizer, model


def _warn_if_torch_below_official_requirement() -> None:
    if _version_tuple(torch.__version__) < IDEOGRAM4_TORCH_MIN:
        warnings.warn(
            "Ideogram 4's official package declares torch>=2.11. "
            f"This environment has torch {torch.__version__}; if loading or fp8 "
            "execution fails, upgrade to the torch 2.11 stack.",
            stacklevel=2,
        )


def _check_nf4_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("Ideogram 4 NF4 weights require CUDA.")
    try:
        bnb_version = metadata.version("bitsandbytes")
    except metadata.PackageNotFoundError as exc:
        raise ImportError("Ideogram 4 NF4 weights require bitsandbytes>=0.49.2.") from exc
    if _version_tuple(bnb_version) < (0, 49, 2):
        raise ImportError(
            f"Ideogram 4 NF4 weights require bitsandbytes>=0.49.2, got {bnb_version}."
        )


class Ideogram4Model(BaseModel):
    arch = "ideogram4"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["Ideogram4Transformer"]
        self.use_old_lokr_format = False
        self.te_padding_side = "left"
        self.unconditional_transformer = None
        self.caption_verifier = CaptionVerifier()
        self.latent_space_version = "ideogram4_reference_patched_norm_v1"

    @property
    def text_embedding_space_version(self):
        return f"{self.arch}_te_v2"

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def _model_kwargs(self) -> dict:
        return self.model_config.model_kwargs or {}

    def _skip_unconditional_transformer_for_training(self) -> bool:
        return bool(
            self._model_kwargs().get(
                "skip_unconditional_transformer_for_training", False
            )
        )

    def warn_if_fp8_training_without_dequantize(self) -> None:
        if getattr(self, "quantization", None) != "fp8":
            return
        if self._model_kwargs().get("dequantize_fp8_transformer", False):
            return
        if getattr(self, "_warned_fp8_training_without_dequantize", False):
            return
        warnings.warn(
            "Ideogram 4 FP8 training is using the lower-VRAM weight-only "
            "transformer path. That path dequantizes Fp8Linear weights during "
            "forward, so the first training step and sample generation can "
            "appear stuck at 0%. On 48GB GPUs such as L40, set "
            "model_kwargs.dequantize_fp8_transformer: true to dequantize the "
            "FP8 transformers once before training and generation.",
            stacklevel=2,
        )
        self._warned_fp8_training_without_dequantize = True

    def _resolve_max_text_tokens(self) -> int:
        kwargs = self._model_kwargs()
        raw_value = kwargs.get("max_text_tokens", kwargs.get("max_text_length", None))
        if raw_value is None:
            return Ideogram4PipelineConfig.max_text_tokens
        max_text_tokens = int(raw_value)
        if max_text_tokens < 1:
            raise ValueError("Ideogram 4 max_text_tokens must be greater than 0.")
        return max_text_tokens

    def _read_local_meta(self, model_path: str) -> dict:
        if not os.path.isdir(model_path):
            return {}
        meta_path = os.path.join(model_path, "aitk_meta.yaml")
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _resolve_quantization(self, model_path: str, quantization: str) -> str:
        meta_quantization = str(
            self._read_local_meta(model_path).get("ideogram4_quantization", "")
        ).lower()
        if meta_quantization in {"nf4", "fp8", "nvfp4"}:
            return meta_quantization
        return quantization

    def _is_comfy_layout(self, model_path: str, quantization: str) -> bool:
        kwargs = self._model_kwargs()
        if kwargs.get("comfy_layout", False):
            return True
        if quantization == "nvfp4" and not model_path:
            return True
        if (model_path or "").lower().rstrip("/") == IDEOGRAM4_COMFY_REPO.lower():
            return True
        if quantization not in IDEOGRAM4_COMFY_FILES or not os.path.isdir(model_path):
            return False
        files = IDEOGRAM4_COMFY_FILES[quantization]
        return os.path.exists(os.path.join(model_path, files["conditional"].replace("/", os.sep)))

    def _resolve_comfy_config_source(self, model_path: str) -> str:
        kwargs = self._model_kwargs()
        for key in (
            "text_encoder_config_name_or_path",
            "config_name_or_path",
            "tokenizer_name_or_path",
        ):
            value = kwargs.get(key)
            if value:
                return value

        extras = self.model_config.extras_name_or_path
        if extras and extras != model_path:
            return extras

        if os.path.isdir(model_path):
            if (
                os.path.exists(os.path.join(model_path, "tokenizer"))
                and os.path.exists(os.path.join(model_path, "text_encoder", "config.json"))
            ):
                return model_path
            meta = self._read_local_meta(model_path)
            for key in (
                "ideogram4_config_model",
                "ideogram4_base_model",
                "extras_name_or_path",
                "base_model",
                "base_model_name_or_path",
            ):
                if meta.get(key):
                    return meta[key]

        return IDEOGRAM4_FP8_REPO

    def _comfy_pipeline_config(
        self,
        weights_path: str,
        config_path: str,
        quantization: str,
    ) -> Ideogram4PipelineConfig:
        files = IDEOGRAM4_COMFY_FILES[quantization]
        return Ideogram4PipelineConfig(
            weights_repo=weights_path,
            text_encoder_config_repo=config_path,
            text_encoder_weights_repo=weights_path,
            conditional_index_filename=files["conditional"],
            unconditional_index_filename=files["unconditional"],
            autoencoder_filename=files["vae"],
            text_encoder_weights_filename=files["text_encoder"],
            max_text_tokens=self._resolve_max_text_tokens(),
        )

    def _resolve_base_components_path(self, model_path: str, quantization: str) -> str:
        extras = self.model_config.extras_name_or_path
        if extras and extras != model_path:
            return extras

        if os.path.isdir(model_path):
            meta = self._read_local_meta(model_path)
            for key in (
                "ideogram4_base_model",
                "extras_name_or_path",
                "base_model",
                "base_model_name_or_path",
            ):
                if meta.get(key):
                    return meta[key]
            return default_repo_for_quantization(quantization)

        return model_path or default_repo_for_quantization(quantization)

    def _load_transformer_from_path(
        self,
        component_root: str,
        index_filename: str,
        dtype: torch.dtype,
    ) -> Ideogram4Transformer:
        state_dict = _load_component_state_dict(component_root, index_filename)
        transformer = _build_transformer(
            Ideogram4Config(), state_dict, self.device_torch, dtype
        )
        del state_dict
        return transformer

    def _dequantize_fp8_transformer_if_requested(
        self,
        transformer: Ideogram4Transformer,
        *,
        dtype: torch.dtype,
        label: str,
    ) -> int:
        if not self._model_kwargs().get("dequantize_fp8_transformer", False):
            return 0
        self.print_and_status_update(f"Dequantizing FP8 {label} transformer")
        replaced = dequantize_fp8_linears(
            transformer, dtype=dtype, device=self.device_torch
        )
        if replaced == 0:
            warnings.warn(
                "dequantize_fp8_transformer was enabled but no Fp8Linear "
                f"layers were found in the {label} transformer.",
                stacklevel=2,
            )
        return replaced

    def _transformer_ignore_modules(self, transformer: Ideogram4Transformer) -> list:
        return [
            transformer.rotary_emb.inv_freq,
            transformer.input_proj,
            transformer.llm_cond_proj,
        ]

    def _apply_transformer_memory_policy(
        self,
        transformer: Ideogram4Transformer,
        *,
        component: str,
    ) -> None:
        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            attach_layer_offloading(
                self,
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                component=component,
                block_paths=self.get_transformer_block_names(),
                ignore_modules=self._transformer_ignore_modules(transformer),
            )
        elif self.model_config.low_vram:
            transformer.to("cpu")
        else:
            transformer.to(self.device_torch)

    def load_model(self):
        _warn_if_torch_below_official_requirement()

        dtype = self.torch_dtype
        model_path = self.model_config.name_or_path
        quantization = infer_ideogram4_quantization(model_path, self._model_kwargs())
        quantization = self._resolve_quantization(model_path, quantization)
        self.model_config.quantize = True
        if quantization == "nf4":
            _check_nf4_runtime(self.device_torch)

        comfy_layout = self._is_comfy_layout(model_path, quantization)
        if comfy_layout:
            weights_model_path = model_path or default_repo_for_quantization(quantization)
            config_model_path = self._resolve_comfy_config_source(model_path)
            pipeline_config = self._comfy_pipeline_config(
                weights_model_path,
                config_model_path,
                quantization,
            )
            base_model_path = weights_model_path
        else:
            base_model_path = self._resolve_base_components_path(model_path, quantization)
            weights_model_path = base_model_path
            config_model_path = base_model_path
            pipeline_config = Ideogram4PipelineConfig(
                weights_repo=base_model_path,
                max_text_tokens=self._resolve_max_text_tokens(),
            )
        local_transformer = (
            os.path.isdir(model_path)
            and os.path.exists(os.path.join(model_path, "transformer"))
        )

        self.print_and_status_update(
            f"Loading Ideogram 4 {quantization.upper()} components"
        )

        conditional_root = (
            model_path if local_transformer else weights_model_path
        )
        conditional_transformer = self._load_transformer_from_path(
            conditional_root,
            pipeline_config.conditional_index_filename,
            dtype,
        )

        if quantization == "fp8":
            self._dequantize_fp8_transformer_if_requested(
                conditional_transformer, dtype=dtype, label="conditional"
            )

        if self._skip_unconditional_transformer_for_training():
            self.print_and_status_update(
                "Skipping Ideogram 4 unconditional transformer (experimental)"
            )
            unconditional_transformer = None
        else:
            self.print_and_status_update("Loading Ideogram 4 unconditional transformer")
            unconditional_transformer = self._load_transformer_from_path(
                weights_model_path,
                pipeline_config.unconditional_index_filename,
                dtype,
            )
            if quantization == "fp8":
                self._dequantize_fp8_transformer_if_requested(
                    unconditional_transformer, dtype=dtype, label="unconditional"
                )

        self.print_and_status_update("Loading Qwen3-VL text encoder")
        if comfy_layout:
            tokenizer, text_encoder = _load_qwen3_vl_comfy_local_or_hf(
                config_model_path,
                weights_model_path,
                pipeline_config.text_encoder_weights_filename,
                self.device_torch,
                dtype,
                tokenizer_subfolder=pipeline_config.tokenizer_subfolder,
                text_encoder_subfolder=pipeline_config.text_encoder_subfolder,
                status_callback=self.print_and_status_update,
            )
        else:
            tokenizer, text_encoder = _load_qwen3_vl_local_or_hf(
                base_model_path,
                self.device_torch,
                dtype,
                tokenizer_subfolder=pipeline_config.tokenizer_subfolder,
                text_encoder_subfolder=pipeline_config.text_encoder_subfolder,
                status_callback=self.print_and_status_update,
            )

        self.print_and_status_update("Loading Ideogram 4 VAE")
        autoencoder = _load_autoencoder_local_or_hf(
            weights_model_path,
            pipeline_config.autoencoder_filename,
            self.device_torch,
            dtype,
        )

        pipe = Ideogram4Pipeline(
            conditional_transformer=conditional_transformer,
            unconditional_transformer=unconditional_transformer,
            text_encoder=text_encoder,
            text_tokenizer=tokenizer,
            autoencoder=autoencoder,
            config=pipeline_config,
            device=self.device_torch,
            dtype=dtype,
        )

        conditional_transformer.eval()
        conditional_transformer.dtype = dtype
        if unconditional_transformer is not None:
            unconditional_transformer.requires_grad_(False)
            unconditional_transformer.eval()
            unconditional_transformer.dtype = dtype
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        autoencoder.requires_grad_(False)
        autoencoder.eval()

        self._apply_transformer_memory_policy(
            conditional_transformer,
            component="transformer",
        )
        if unconditional_transformer is not None:
            self._apply_transformer_memory_policy(
                unconditional_transformer,
                component="unconditional_transformer",
            )

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            attach_layer_offloading(
                self,
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
                component="text_encoder",
            )

        self.noise_scheduler = Ideogram4Model.get_train_scheduler()
        self.vae = autoencoder
        self.text_encoder = [text_encoder]
        self.tokenizer = [tokenizer]
        self.model = conditional_transformer
        self.unconditional_transformer = unconditional_transformer
        self.pipeline = pipe
        self.base_model_path = base_model_path
        self.config_model_path = config_model_path
        self.quantization = quantization
        self.print_and_status_update("Ideogram 4 model loaded")

    def get_generation_pipeline(self):
        return Ideogram4Pipeline(
            conditional_transformer=unwrap_model(self.transformer),
            unconditional_transformer=(
                unwrap_model(self.unconditional_transformer)
                if self.unconditional_transformer is not None
                else None
            ),
            text_encoder=unwrap_model(self.text_encoder[0]),
            text_tokenizer=self.tokenizer[0],
            autoencoder=unwrap_model(self.vae),
            config=self.pipeline.config,
            device=self.device_torch,
            dtype=self.torch_dtype,
        )

    def save_device_state(self):
        unet_has_grad = self.get_model_has_grad()
        self.device_state = {
            "vae": {
                "training": self.vae.training,
                "device": module_device(self.vae),
                "requires_grad": False,
            },
            "unet": {
                "training": self.unet.training,
                "device": self.unet.device,
                "requires_grad": unet_has_grad,
            },
            "text_encoder": [],
            "unconditional_transformer": None,
        }
        if self.unconditional_transformer is not None:
            self.device_state["unconditional_transformer"] = {
                "training": self.unconditional_transformer.training,
                "device": self.unconditional_transformer.device,
                "requires_grad": False,
            }
        for encoder in self.text_encoder:
            self.device_state["text_encoder"].append(
                {
                    "training": encoder.training,
                    "device": module_device(encoder),
                    "requires_grad": False,
                }
            )

    def set_device_state(self, state):
        if state["vae"]["training"]:
            self.vae.train()
        else:
            self.vae.eval()
        self.vae.to(state["vae"]["device"])
        self.vae.requires_grad_(state["vae"].get("requires_grad", False))

        if state["unet"]["training"]:
            self.unet.train()
        else:
            self.unet.eval()
        self.unet.to(state["unet"]["device"])
        self.unet.requires_grad_(state["unet"]["requires_grad"])

        text_encoder_state = state["text_encoder"]
        if isinstance(text_encoder_state, dict):
            text_encoder_state = [text_encoder_state]
        for encoder, encoder_state in zip(self.text_encoder, text_encoder_state):
            if encoder_state["training"]:
                encoder.train()
            else:
                encoder.eval()
            move_module_to_device(encoder, encoder_state["device"])
            encoder.requires_grad_(encoder_state.get("requires_grad", False))

        unconditional_state = state.get("unconditional_transformer", None)
        if self.unconditional_transformer is not None:
            if unconditional_state is None:
                unconditional_state = {
                    "training": False,
                    "device": state["unet"]["device"]
                    if not state["unet"]["training"]
                    else torch.device("cpu"),
                    "requires_grad": False,
                }

            if unconditional_state["training"]:
                self.unconditional_transformer.train()
            else:
                self.unconditional_transformer.eval()
            self.unconditional_transformer.to(unconditional_state["device"])
            self.unconditional_transformer.requires_grad_(
                unconditional_state.get("requires_grad", False)
            )
        flush()

    def _warn_natural_caption_quality_once(self) -> None:
        if getattr(self, "_warned_natural_caption_quality", False):
            return
        warnings.warn(
            "Ideogram 4 was trained on structured JSON captions; natural-language "
            "captions/prompts are allowed, but training and sample outputs may be "
            "worse than with Ideogram JSON captions.",
            stacklevel=2,
        )
        self._warned_natural_caption_quality = True

    def _warn_conditional_only_preview_once(self) -> None:
        if getattr(self, "_warned_conditional_only_preview", False):
            return
        warnings.warn(
            "Ideogram 4 unconditional transformer is not loaded; native samples "
            "are using experimental conditional-only previews without asymmetric "
            "CFG guidance.",
            stacklevel=2,
        )
        self._warned_conditional_only_preview = True

    def _validate_caption(self, prompt: str) -> None:
        if prompt.strip() == "":
            return

        require_json = self._model_kwargs().get("require_json_captions", False)
        caption_strict = self._model_kwargs().get("caption_strict", False)

        try:
            parsed = json.loads(prompt)
        except json.JSONDecodeError as exc:
            if require_json:
                raise ValueError(
                    "Ideogram 4 training captions must be JSON strings. "
                    f"Invalid JSON: {exc}"
                ) from exc
            self._warn_natural_caption_quality_once()
            return

        if not isinstance(parsed, dict):
            if require_json:
                raise ValueError("Ideogram 4 captions must be top-level JSON objects.")
            self._warn_natural_caption_quality_once()
            return

        issues = self.caption_verifier.verify(parsed)
        ensure_ascii_issues = self.caption_verifier.check_ensure_ascii_false(prompt)
        issues = ensure_ascii_issues + issues
        if not issues:
            return
        message = "Ideogram 4 caption verifier warnings:\n" + "\n".join(issues)
        if caption_strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)

    def _sample_prompt_to_json_caption(self, prompt: Optional[str]) -> Optional[str]:
        if prompt is None:
            return None
        prompt = str(prompt)
        if prompt.strip() == "":
            return prompt

        try:
            parsed = json.loads(prompt)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            return prompt

        caption = {
            "high_level_description": prompt,
            "compositional_deconstruction": {
                "background": "",
                "elements": [
                    {
                        "type": "obj",
                        "desc": prompt,
                    }
                ],
            },
        }
        if not getattr(self, "_warned_wrapped_sample_prompt", False):
            warnings.warn(
                "Ideogram 4 sample prompt was not a JSON object; wrapping it "
                "into a minimal local JSON caption for generation. Dataset "
                "captions are still required to be JSON when "
                "model_kwargs.require_json_captions is true.",
                stacklevel=2,
            )
            self._warned_wrapped_sample_prompt = True
        return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))

    def _normalize_sample_image_config(self, image_config: GenerateImageConfig) -> None:
        image_config.prompt = self._sample_prompt_to_json_caption(image_config.prompt)
        image_config.prompt_2 = self._sample_prompt_to_json_caption(image_config.prompt_2)
        image_config.negative_prompt = self._sample_prompt_to_json_caption(
            image_config.negative_prompt
        )
        image_config.negative_prompt_2 = self._sample_prompt_to_json_caption(
            image_config.negative_prompt_2
        )

    def prepare_sample_image_config_for_encoding(
        self, image_config: GenerateImageConfig
    ) -> None:
        if self._model_kwargs().get("json_wrap_sample_prompts", False):
            self._normalize_sample_image_config(image_config)

    @torch.no_grad()
    def generate_images(self, image_configs: List[GenerateImageConfig], sampler=None, pipeline=None):
        for image_config in image_configs:
            self.prepare_sample_image_config_for_encoding(image_config)
        return super().generate_images(image_configs, sampler=sampler, pipeline=pipeline)

    def sample_memory_generate_components(self) -> tuple[str, ...]:
        return ()

    def get_prompt_embeds(self, prompt: str) -> AdvancedPromptEmbeds:
        if module_device(self.pipeline.text_encoder) != self.device_torch:
            move_module_to_device(self.pipeline.text_encoder, self.device_torch)

        prompts = [prompt] if isinstance(prompt, str) else prompt
        for p in prompts:
            self._validate_caption(p)

        text_embeds = []
        for tokens, num_text in [self.pipeline._tokenize(p) for p in prompts]:
            token_ids = tokens.to(self.device_torch).unsqueeze(0)
            text_pos = torch.arange(num_text, device=self.device_torch)
            text_position_ids = torch.stack([text_pos, text_pos, text_pos], dim=1).unsqueeze(0)
            indicator = torch.full(
                (1, num_text),
                LLM_TOKEN_INDICATOR,
                dtype=torch.long,
                device=self.device_torch,
            )
            llm_features = self.pipeline._encode_text(
                token_ids, text_position_ids, indicator
            )
            text_embeds.append(llm_features[0])

        return AdvancedPromptEmbeds(text_embeds=text_embeds)

    def _text_embedding_items(self, text_embeddings):
        if isinstance(text_embeddings, AdvancedPromptEmbeds):
            raw_items = text_embeddings.text_embeds
            if not isinstance(raw_items, (list, tuple)):
                raw_items = [raw_items]
            items = []
            for item in raw_items:
                item = item.to(self.device_torch)
                if item.dim() == 2:
                    items.append(item)
                elif item.dim() == 3:
                    items.extend(item[idx] for idx in range(item.shape[0]))
                else:
                    raise ValueError(
                        f"Ideogram 4 text embeddings must be rank 2 or 3, got {tuple(item.shape)}"
                    )
            return items

        llm_features = text_embeddings.text_embeds
        if isinstance(llm_features, (list, tuple)):
            if len(llm_features) != 1:
                raise ValueError("Ideogram 4 expected a single legacy text embedding tensor.")
            llm_features = llm_features[0]
        llm_features = llm_features.to(self.device_torch)
        if llm_features.dim() == 2:
            llm_features = llm_features.unsqueeze(0)
        if llm_features.dim() != 3:
            raise ValueError(
                f"Ideogram 4 text embeddings must be rank 3, got {tuple(llm_features.shape)}"
            )

        attention_mask = getattr(text_embeddings, "attention_mask", None)
        if attention_mask is None:
            return [llm_features[idx] for idx in range(llm_features.shape[0])]

        attention_mask = attention_mask.to(self.device_torch).bool()
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        items = []
        for feature_row, mask_row in zip(llm_features, attention_mask):
            if mask_row.any():
                items.append(feature_row[mask_row])
            else:
                items.append(feature_row[:1] * 0)
        return items

    def _build_transformer_inputs_from_embeds(
        self,
        text_embeddings: PromptEmbeds,
        latent_h: int,
        latent_w: int,
        *,
        include_text: bool = True,
    ):
        text_items = self._text_embedding_items(text_embeddings)
        batch_size = len(text_items)
        max_text_tokens = max(item.shape[0] for item in text_items)
        feature_dim = text_items[0].shape[-1]
        num_image_tokens = latent_h * latent_w
        total_seq_len = max_text_tokens + num_image_tokens if include_text else num_image_tokens

        h_idx = (
            torch.arange(latent_h, device=self.device_torch)
            .view(-1, 1)
            .expand(latent_h, latent_w)
            .reshape(-1)
        )
        w_idx = (
            torch.arange(latent_w, device=self.device_torch)
            .view(1, -1)
            .expand(latent_h, latent_w)
            .reshape(-1)
        )
        t_idx = torch.zeros_like(h_idx)
        image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

        position_ids = torch.zeros(
            batch_size, total_seq_len, 3, dtype=torch.long, device=self.device_torch
        )
        segment_ids = torch.full(
            (batch_size, total_seq_len),
            SEQUENCE_PADDING_INDICATOR,
            dtype=torch.long,
            device=self.device_torch,
        )
        indicator = torch.zeros(
            batch_size, total_seq_len, dtype=torch.long, device=self.device_torch
        )

        if include_text:
            full_llm_features = torch.zeros(
                batch_size,
                total_seq_len,
                feature_dim,
                dtype=text_items[0].dtype,
                device=self.device_torch,
            )
            for batch_idx, item in enumerate(text_items):
                item = item.to(self.device_torch)
                num_text = item.shape[0]
                offset = max_text_tokens - num_text
                full_llm_features[batch_idx, offset:max_text_tokens] = item
                text_pos = torch.arange(num_text, device=self.device_torch)
                text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
                position_ids[batch_idx, offset:max_text_tokens] = text_pos_3d
                position_ids[batch_idx, max_text_tokens:] = image_pos
                indicator[batch_idx, offset:max_text_tokens] = LLM_TOKEN_INDICATOR
                indicator[batch_idx, max_text_tokens:] = OUTPUT_IMAGE_INDICATOR
                segment_ids[batch_idx, offset:] = 1
            return full_llm_features, position_ids, segment_ids, indicator, max_text_tokens

        position_ids[:] = image_pos
        segment_ids[:] = 1
        indicator[:] = OUTPUT_IMAGE_INDICATOR
        neg_llm = torch.zeros(
            batch_size,
            num_image_tokens,
            feature_dim,
            dtype=text_items[0].dtype,
            device=self.device_torch,
        )
        return neg_llm, position_ids, segment_ids, indicator, 0

    def _latent_norm(self, latents: torch.Tensor, normalize: bool) -> torch.Tensor:
        shift, scale = get_latent_norm()
        shift = shift.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        scale = scale.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        if normalize:
            return (latents - shift) / scale
        return latents * scale + shift

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if module_device(self.vae) == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        if isinstance(image_list, torch.Tensor):
            images = image_list.to(device, dtype=dtype)
        else:
            images = torch.stack([image.to(device, dtype=dtype) for image in image_list])

        moments = self.vae.encoder(images)
        latents = moments.chunk(2, dim=1)[0]
        latents = patchify_latents(latents, self.pipeline.config.patch_size)
        latents = self._latent_norm(latents, normalize=True)
        return latents.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if module_device(self.vae) == torch.device("cpu"):
            self.vae.to(device)
        latents = latents.to(device, dtype=dtype)
        latents = self._latent_norm(latents, normalize=False)
        latents = unpatchify_latents(latents, self.pipeline.config.patch_size)
        return self.vae.decoder(latents)

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        latents = latent_model_input.to(self.device_torch, dtype=torch.float32)
        batch_size, _, latent_h, latent_w = latents.shape
        latent_tokens = pack_latent_tokens(latents)
        llm_features, position_ids, segment_ids, indicator, max_text_tokens = (
            self._build_transformer_inputs_from_embeds(
                text_embeddings, latent_h, latent_w, include_text=True
            )
        )
        text_z_padding = torch.zeros(
            batch_size,
            max_text_tokens,
            latent_tokens.shape[-1],
            dtype=torch.float32,
            device=self.device_torch,
        )
        model_input = torch.cat([text_z_padding, latent_tokens], dim=1)
        model_timestep = timestep.to(self.device_torch, dtype=torch.float32)
        if model_timestep.ndim == 0:
            model_timestep = model_timestep.unsqueeze(0)
        model_timestep = 1.0 - (model_timestep / 1000.0)

        prediction = self.transformer(
            llm_features=llm_features,
            x=model_input,
            t=model_timestep,
            position_ids=position_ids,
            segment_ids=segment_ids,
            indicator=indicator,
        )
        prediction = prediction[:, max_text_tokens:]
        return -unpack_latent_tokens(prediction, latent_h, latent_w)

    @torch.no_grad()
    def generate_single_image(
        self,
        pipeline: Ideogram4Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        sample_memory = getattr(self, "_sample_memory_coordinator", None)
        if sample_memory is None and self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)
        latent_h = gen_config.height // sc
        latent_w = gen_config.width // sc

        preset_name = extra.pop("ideogram_preset", None) or extra.pop(
            "sampler_preset", "V4_DEFAULT_20"
        )
        preset = PRESETS.get(str(preset_name), PRESETS["V4_DEFAULT_20"])
        num_steps = int(gen_config.num_inference_steps or preset.num_steps)
        if num_steps != preset.num_steps:
            guidance_schedule = tuple([float(gen_config.guidance_scale)] * num_steps)
            mu = preset.mu
            std = preset.std
        else:
            guidance_schedule = preset.guidance_schedule
            mu = preset.mu
            std = preset.std

        schedule = get_schedule_for_resolution(
            (gen_config.height, gen_config.width), known_mean=mu, std=std
        )
        step_intervals = make_step_intervals(num_steps).to(self.device_torch)
        gw_per_step = torch.as_tensor(
            guidance_schedule, dtype=torch.float32, device=self.device_torch
        )

        cond = conditional_embeds.to(self.device_torch)
        llm_features, position_ids, segment_ids, indicator, max_text_tokens = (
            self._build_transformer_inputs_from_embeds(
                cond, latent_h, latent_w, include_text=True
            )
        )
        batch_size = llm_features.shape[0]
        if pipeline.unconditional_transformer is None:
            self._warn_conditional_only_preview_once()
            neg_llm = None
            neg_position_ids = None
            neg_segment_ids = None
            neg_indicator = None
        else:
            neg_llm, neg_position_ids, neg_segment_ids, neg_indicator, _ = (
                self._build_transformer_inputs_from_embeds(
                    cond, latent_h, latent_w, include_text=False
                )
            )

        latent_dim = pipeline.conditional_transformer.config.in_channels
        sample_generator = generator
        if sample_generator is None or getattr(sample_generator, "device", None) != self.device_torch:
            sample_generator = torch.Generator(device=self.device_torch)
            sample_generator.manual_seed(gen_config.seed)
        z = torch.randn(
            batch_size,
            latent_h * latent_w,
            latent_dim,
            dtype=torch.float32,
            device=self.device_torch,
            generator=sample_generator,
        )
        text_z_padding = torch.zeros(
            batch_size,
            max_text_tokens,
            latent_dim,
            dtype=torch.float32,
            device=self.device_torch,
        )

        for step_idx in range(num_steps - 1, -1, -1):
            step_num = num_steps - step_idx
            t_val = float(schedule(step_intervals[step_idx + 1].unsqueeze(0)).item())
            s_val = float(schedule(step_intervals[step_idx].unsqueeze(0)).item())
            t = torch.full(
                (batch_size,), t_val, dtype=torch.float32, device=self.device_torch
            )
            pos_z = torch.cat([text_z_padding, z], dim=1)
            if sample_memory is not None:
                sample_memory.activate(
                    ("transformer",),
                    phase_name="ideogram conditional denoise",
                    message=(
                        f"Low-VRAM sample: Ideogram denoise step "
                        f"{step_num}/{num_steps} conditional"
                    ),
                )
            pos_out = pipeline.conditional_transformer(
                llm_features=llm_features,
                x=pos_z,
                t=t,
                position_ids=position_ids,
                segment_ids=segment_ids,
                indicator=indicator,
            )
            pos_v = pos_out[:, max_text_tokens:]
            if pipeline.unconditional_transformer is None:
                v = pos_v
            else:
                if sample_memory is not None:
                    sample_memory.activate(
                        ("unconditional_transformer",),
                        phase_name="ideogram unconditional denoise",
                        message=(
                            f"Low-VRAM sample: Ideogram denoise step "
                            f"{step_num}/{num_steps} unconditional"
                        ),
                    )
                neg_v = pipeline.unconditional_transformer(
                    llm_features=neg_llm,
                    x=z,
                    t=t,
                    position_ids=neg_position_ids,
                    segment_ids=neg_segment_ids,
                    indicator=neg_indicator,
                )
                v = gw_per_step[step_idx] * pos_v + (1.0 - gw_per_step[step_idx]) * neg_v
            z = z + v * (s_val - t_val)

        if sample_memory is not None:
            sample_memory.activate(
                ("vae",),
                phase_name="ideogram decode",
                message="Low-VRAM sample: decoding Ideogram image",
            )
        images = pipeline._decode(z, grid_h=latent_h, grid_w=latent_w)
        return images[0]

    def get_model_has_grad(self):
        return any(p.requires_grad for p in self.model.parameters())

    def get_te_has_grad(self):
        return False

    def get_model_to_train(self):
        return self.transformer

    def save_model(self, output_path, meta, save_dtype):
        if getattr(self, "quantization", None) == "nvfp4":
            raise ValueError(
                "Saving or full fine-tuning packed NVFP4 Ideogram 4 base weights is "
                "not supported. Use LoRA/adapters with the NVFP4 base frozen, or "
                "dequantize to a floating-point checkpoint before full fine-tuning."
            )

        output_dir = output_path.removesuffix(".safetensors")
        transformer_dir = os.path.join(output_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)

        transformer: Ideogram4Transformer = unwrap_model(self.model)
        save_dict = {}
        for key, value in transformer.state_dict().items():
            if isinstance(value, torch.Tensor):
                if value.is_floating_point() and value.dtype not in {
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                }:
                    value = value.to(dtype=save_dtype)
                save_dict[key] = value.detach().cpu()
        save_file(
            save_dict,
            os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors"),
        )

        meta = dict(meta or {})
        meta["ideogram4_base_model"] = getattr(
            self, "base_model_path", self.model_config.extras_name_or_path
        )
        if getattr(self, "config_model_path", None):
            meta["ideogram4_config_model"] = self.config_model_path
        meta["ideogram4_quantization"] = getattr(self, "quantization", None)
        meta["ideogram4_local_only"] = True
        meta_path = os.path.join(output_dir, "aitk_meta.yaml")
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        quantization = getattr(self, "quantization", None)
        return f"ideogram4-{quantization}" if quantization else "ideogram4"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd
