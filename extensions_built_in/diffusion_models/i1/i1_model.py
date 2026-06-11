from __future__ import annotations

import gc
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Optional, Sequence

import torch
import yaml
from diffusers import AutoencoderKL
from huggingface_hub import get_token, hf_hub_download
from huggingface_hub.errors import GatedRepoError
from optimum.quanto import freeze
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, T5GemmaModel

from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.exceptions import UserFacingError
from toolkit.memory_management import attach_layer_offloading
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.util.quantize import get_qtype, quantize, quantize_model

from .src.model import I1DiT3B
from .src.pipeline import (
    I1Pipeline,
    i1_rectified_flow_noisy_latents,
    i1_velocity_target,
    prepare_i1_image_tensor,
    prepare_i1_latent_tensor,
    reverse_scale_flux2_latents,
    scale_flux2_latents,
)

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


I1_MODEL_REPO = "zlab-princeton/i1-3B"
I1_CHECKPOINT_FILENAME = "1024_resolution_checkpoint_torch.pt"
I1_TEXT_ENCODER = "google/t5gemma-2b-2b-ul2-it"
I1_VAE = "black-forest-labs/FLUX.2-dev"
I1_REWRITER_DEFAULT = "Qwen/Qwen3-4B-Instruct-2507"
I1_REWRITER_CHOICES = {"Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-30B-A3B"}

I1_REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's text-to-image prompt into a single visually grounded "
    "75-150 word prompt. Preserve all requested objects, counts, relationships, "
    "attributes, and exact quoted text. Prefer concrete materials, lighting, "
    "spatial separation, and legible typography. Return only the rewritten prompt."
)


scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.3,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 0.3,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_dynamic_shifting": False,
}


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _from_pretrained_with_dtype(cls, *args, dtype: torch.dtype, **kwargs):
    try:
        return cls.from_pretrained(*args, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(*args, torch_dtype=dtype, **kwargs)


class I1Model(BaseModel):
    arch = "i1"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["I1DiT3B"]
        self.te_padding_side = "right"
        self.latent_space_version = self.arch
        self._warned_non_square_latents = False
        self.checkpoint_filename = self.model_config.model_kwargs.get(
            "checkpoint_filename", I1_CHECKPOINT_FILENAME
        )
        self.text_encoder_name_or_path = self.model_config.model_kwargs.get(
            "text_encoder_name_or_path", I1_TEXT_ENCODER
        )
        self.vae_name_or_path = self.model_config.model_kwargs.get(
            "vae_name_or_path", I1_VAE
        )
        self.inference_timestep_shift = float(
            self.model_config.model_kwargs.get("inference_timestep_shift", 0.3)
        )
        self.cfg_rescale = self.model_config.model_kwargs.get("cfg_rescale", 1.0)
        self.rewrite_prompt = bool(
            self.model_config.model_kwargs.get("rewrite_prompt", False)
        )
        self.rewriter_model = self.model_config.model_kwargs.get(
            "rewriter_model", I1_REWRITER_DEFAULT
        )
        self.rewrite_batch_size = int(
            self.model_config.model_kwargs.get("rewrite_batch_size", 1)
        )
        self.hf_token = self.model_config.model_kwargs.get(
            "token",
            self.model_config.model_kwargs.get("use_auth_token", None),
        )

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def _hf_token_kwargs(self) -> dict:
        token = self.hf_token
        if token is None:
            token = get_token()
        return {"token": token} if token else {}

    def _vae_load_kwargs(self) -> dict:
        if os.path.isdir(self.vae_name_or_path) and os.path.isfile(
            os.path.join(self.vae_name_or_path, "config.json")
        ):
            return {}
        return {"subfolder": "vae"}

    def _resolve_checkpoint_path(self, model_path: str) -> str:
        model_path = model_path or I1_MODEL_REPO
        if os.path.isfile(model_path):
            return model_path
        if os.path.isdir(model_path):
            for filename in (
                self.checkpoint_filename,
                self.checkpoint_filename.replace(".pt", ".safetensors"),
            ):
                direct = os.path.join(model_path, filename)
                if os.path.isfile(direct):
                    return direct
        return hf_hub_download(
            repo_id=model_path,
            filename=self.checkpoint_filename,
            repo_type="model",
            **self._hf_token_kwargs(),
        )

    def _load_transformer_checkpoint(self, checkpoint_path: str) -> I1DiT3B:
        transformer = I1DiT3B()
        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path, device="cpu")
        else:
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = (
                checkpoint["model"]
                if isinstance(checkpoint, dict) and "model" in checkpoint
                else checkpoint
            )
        transformer.load_state_dict(state_dict, strict=True)
        return transformer

    def load_model(self):
        dtype = self.torch_dtype
        model_path = self.model_config.name_or_path or I1_MODEL_REPO

        self.print_and_status_update("Loading i1 transformer")
        checkpoint_path = self._resolve_checkpoint_path(model_path)
        transformer = self._load_transformer_checkpoint(checkpoint_path)
        transformer.to(dtype=dtype)

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing i1 transformer")
            quantize_model(self, transformer)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            attach_layer_offloading(
                self,
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                component="transformer",
                block_paths=self.get_transformer_block_names(),
            )

        if self.model_config.low_vram:
            transformer.to("cpu")
        else:
            transformer.to(self.device_torch)
        transformer.eval()

        self.print_and_status_update("Loading i1 text encoder")
        hf_token_kwargs = self._hf_token_kwargs()
        tokenizer = AutoTokenizer.from_pretrained(
            self.text_encoder_name_or_path,
            **hf_token_kwargs,
        )
        text_model = _from_pretrained_with_dtype(
            T5GemmaModel,
            self.text_encoder_name_or_path,
            dtype=dtype,
            **hf_token_kwargs,
        )
        text_encoder = text_model.encoder
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        text_encoder.to(self.device_torch, dtype=dtype)
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing i1 text encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        self.print_and_status_update("Loading FLUX.2 VAE")
        try:
            vae = _from_pretrained_with_dtype(
                AutoencoderKL,
                self.vae_name_or_path,
                dtype=dtype,
                **self._vae_load_kwargs(),
                **hf_token_kwargs,
            )
        except (GatedRepoError, OSError) as exc:
            if self.vae_name_or_path == I1_VAE:
                raise UserFacingError(
                    "Access required for black-forest-labs/FLUX.2-dev.\n"
                    "i1 uses the gated FLUX.2 VAE from that repo. Request and accept "
                    "access at https://huggingface.co/black-forest-labs/FLUX.2-dev, "
                    "then run `hf auth login` with the approved account.\n"
                    "Alternatively, set `model.model_kwargs.vae_name_or_path` to a "
                    "local FLUX.2-dev folder containing `vae/`, or to the local VAE "
                    "folder itself."
                ) from exc
            raise
        vae.requires_grad_(False)
        vae.eval()
        vae.to(self.device_torch, dtype=dtype)

        self.noise_scheduler = I1Model.get_train_scheduler()
        pipe = I1Pipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            dtype=dtype,
        )

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.model = transformer
        self.pipeline = pipe
        flush()
        self.print_and_status_update("i1 model loaded")

    def get_generation_pipeline(self):
        pipeline = I1Pipeline(
            tokenizer=self.tokenizer,
            text_encoder=unwrap_model(self.text_encoder),
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
            dtype=self.torch_dtype,
        )
        if not self.model_config.low_vram:
            pipeline.to(self.device_torch)
        return pipeline

    def _prompt_embeds_to_tensor(self, prompt_embeds: PromptEmbeds) -> torch.Tensor:
        text_embeds = prompt_embeds.text_embeds
        if isinstance(text_embeds, torch.Tensor):
            return text_embeds.to(self.device_torch, dtype=self.torch_dtype)
        normalized = []
        for item in text_embeds:
            if item.dim() == 3 and item.shape[0] == 1:
                item = item[0]
            normalized.append(item.to(self.device_torch, dtype=self.torch_dtype))
        return torch.stack(normalized, dim=0)

    def _prompt_mask_to_tensor(
        self,
        prompt_embeds: PromptEmbeds,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        mask = prompt_embeds.attention_mask
        if mask is None:
            return torch.ones(
                (batch_size, seq_len), device=self.device_torch, dtype=torch.bool
            )
        if isinstance(mask, torch.Tensor):
            return mask.to(self.device_torch, dtype=torch.bool)
        normalized = []
        for item in mask:
            if item.dim() == 2 and item.shape[0] == 1:
                item = item[0]
            normalized.append(item.to(self.device_torch, dtype=torch.bool))
        return torch.stack(normalized, dim=0)

    def _validate_latent_size(self, latents: torch.Tensor) -> None:
        expected = self.transformer.input_size
        if latents.shape[-2:] != (expected, expected):
            raise UserFacingError(
                "i1-3B 1024-resolution checkpoint expects 1024x1024 images "
                f"({expected}x{expected} latents), got {tuple(latents.shape[-2:])}. "
                "Use square 1024 dataset crops for i1 and recache latents if this "
                "came from an older aspect-ratio bucket cache."
            )

    def _prepare_latents_for_i1(self, latents: torch.Tensor) -> torch.Tensor:
        expected = self.transformer.input_size
        if latents.shape[-2:] == (expected, expected):
            return latents
        if not self._warned_non_square_latents:
            self.print_and_status_update(
                "i1 received non-square latents; center-cropping/resizing to "
                f"{expected}x{expected}. For best results, use square_crop: true "
                "and recache latents."
            )
            self._warned_non_square_latents = True
        return prepare_i1_latent_tensor(latents, expected)

    def _ensure_transformer_on_device(self):
        if _module_device(self.transformer) == torch.device("cpu"):
            self.transformer.to(self.device_torch)

    def generate_single_image(
        self,
        pipeline: I1Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        del unconditional_embeds
        self._ensure_transformer_on_device()

        gen_config.width = 1024
        gen_config.height = 1024
        if self.rewrite_prompt:
            prompt = self._rewrite_prompts([gen_config.prompt])[0]
            conditional_embeds = self.get_prompt_embeds(prompt)

        prompt_embeds = self._prompt_embeds_to_tensor(conditional_embeds)
        prompt_mask = self._prompt_mask_to_tensor(
            conditional_embeds,
            prompt_embeds.shape[0],
            prompt_embeds.shape[1],
        )
        guidance_rescale = extra.pop("guidance_rescale", None)
        if guidance_rescale is None:
            sample_rescale = gen_config.guidance_rescale
            guidance_rescale = (
                sample_rescale if sample_rescale != 0.0 else self.cfg_rescale
            )

        return pipeline(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_mask,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            guidance_rescale=guidance_rescale,
            inference_timestep_shift=extra.pop(
                "inference_timestep_shift", self.inference_timestep_shift
            ),
            latents=gen_config.latents,
            generator=generator,
            **extra,
        ).images[0]

    def _format_rewrite_prompt(self, tokenizer, prompt: str) -> str:
        messages = [
            {"role": "system", "content": I1_REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Input to Rewrite:\n{prompt}"},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    @torch.inference_mode()
    def _rewrite_prompts(self, prompts: Sequence[str]) -> list[str]:
        if self.rewriter_model not in I1_REWRITER_CHOICES:
            raise ValueError(
                "i1 prompt rewriting supports "
                f"{sorted(I1_REWRITER_CHOICES)}, got {self.rewriter_model!r}."
            )
        tokenizer = AutoTokenizer.from_pretrained(self.rewriter_model)
        model = _from_pretrained_with_dtype(
            AutoModelForCausalLM,
            self.rewriter_model,
            dtype=torch.bfloat16,
        ).to(self.device_torch).eval()
        rewritten = []
        try:
            for start in range(0, len(prompts), self.rewrite_batch_size):
                batch_prompts = list(prompts[start : start + self.rewrite_batch_size])
                formatted = [
                    self._format_rewrite_prompt(tokenizer, prompt)
                    for prompt in batch_prompts
                ]
                tokenized = tokenizer(formatted, padding=True, return_tensors="pt")
                inputs = {key: value.to(self.device_torch) for key, value in tokenized.items()}
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    max_new_tokens=16384,
                    pad_token_id=tokenizer.eos_token_id,
                )
                outputs = outputs[:, inputs["input_ids"].shape[1] :]
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                rewritten.extend(
                    text.strip() or prompt
                    for text, prompt in zip(decoded, batch_prompts)
                )
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return rewritten

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: PromptEmbeds,
        batch: Optional["DataLoaderBatchDTO"] = None,
        **kwargs,
    ):
        del batch, kwargs
        self._ensure_transformer_on_device()
        latent_model_input = self._prepare_latents_for_i1(latent_model_input)
        self._validate_latent_size(latent_model_input)

        prompt_embeds = self._prompt_embeds_to_tensor(text_embeddings).detach()
        prompt_mask = self._prompt_mask_to_tensor(
            text_embeddings,
            prompt_embeds.shape[0],
            prompt_embeds.shape[1],
        )
        if prompt_embeds.shape[0] != latent_model_input.shape[0]:
            prompt_embeds = prompt_embeds.expand(latent_model_input.shape[0], -1, -1)
            prompt_mask = prompt_mask.expand(latent_model_input.shape[0], -1)

        model_timestep = timestep.to(self.device_torch, dtype=self.torch_dtype)
        if model_timestep.ndim == 0:
            model_timestep = model_timestep.unsqueeze(0)
        if model_timestep.max() > 1.0:
            model_timestep = model_timestep / 1000.0

        return self.transformer(
            latent_model_input.to(self.device_torch, dtype=self.torch_dtype),
            model_timestep,
            prompt_embeds,
            prompt_mask,
        )

    def get_prompt_embeds(self, prompt: str | Sequence[str]) -> PromptEmbeds:
        if _module_device(self.text_encoder) == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        hidden, mask = self.pipeline.encode_prompt(prompts, device=self.device_torch)
        return PromptEmbeds(hidden, attention_mask=mask)

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        original_samples = self._prepare_latents_for_i1(original_samples)
        noise = self._prepare_latents_for_i1(noise)
        return i1_rectified_flow_noisy_latents(original_samples, noise, timesteps)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        latents = self._prepare_latents_for_i1(batch.latents)
        noise = self._prepare_latents_for_i1(noise)
        return i1_velocity_target(latents, noise).detach()

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if _module_device(self.vae) == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        images = torch.stack(
            [
                prepare_i1_image_tensor(
                    image.to(device, dtype=dtype),
                    self.transformer.input_size * self.pipeline.vae_scale_factor,
                )
                for image in image_list
            ]
        )
        latents = self.vae.encode(images).latent_dist.sample()
        return scale_flux2_latents(latents).to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if _module_device(self.vae) == torch.device("cpu"):
            self.vae.to(device)
        latents = reverse_scale_flux2_latents(latents.to(device, dtype=dtype))
        return self.vae.decode(latents, return_dict=False)[0]

    def save_model(self, output_path, meta, save_dtype):
        transformer = unwrap_model(self.model)
        save_dict = {}
        for key, value in transformer.state_dict().items():
            if hasattr(value, "dequantize"):
                value = value.dequantize()
            save_dict[key] = value.detach().to("cpu", dtype=save_dtype)

        metadata = get_meta_for_safetensors(OrderedDict(meta), name=self.arch)
        if output_path.endswith(".safetensors"):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_file(save_dict, output_path, metadata=metadata)
            return

        os.makedirs(output_path, exist_ok=True)
        save_file(
            save_dict,
            os.path.join(output_path, self.checkpoint_filename.replace(".pt", ".safetensors")),
            metadata=metadata,
        )
        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.dump(meta, f)

    def get_base_model_version(self):
        return self.arch

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["in_blocks", "mid_block", "out_blocks", "text_encoder_adapter"]

    def convert_lora_weights_before_save(self, state_dict):
        return {
            key.replace("transformer.", "diffusion_model."): value
            for key, value in state_dict.items()
        }

    def convert_lora_weights_before_load(self, state_dict):
        return {
            key.replace("diffusion_model.", "transformer."): value
            for key, value in state_dict.items()
        }
