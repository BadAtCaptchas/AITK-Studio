import math
import os
from typing import TYPE_CHECKING, List, Optional

import torch
import yaml
from diffusers.models import Flux2Transformer2DModel
from optimum.quanto import QTensor, freeze
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.prompt_utils import PromptEmbeds
from toolkit.quantized_cache import quantized_cache_key
from toolkit.util.quantize import get_qtype, quantize, quantize_model

from .asymflux2_components import (
    AsymFlux2Transformer2DModel,
    FlowAdapterScheduler,
    LakonLabPixelFlux2KleinPipeline,
    OklabColorEncoder,
    PixelFlux2KleinPipeline,
)
from .flux2_klein_model import Flux2KleinModel

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


ASYMFLUX2_BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"
ASYMFLUX2_ADAPTER = "Lakonik/AsymFLUX.2-klein-9B"


class AsymFlux2Klein9BModel(Flux2KleinModel):
    arch = "asymflux2_klein_9b"
    flux2_klein_te_path: str = "Qwen/Qwen3-8B"
    flux2_te_filename: Optional[str] = None
    flux2_is_guidance_distilled: bool = False

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
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs,
        )
        self.target_lora_modules = ["AsymFlux2Transformer2DModel"]
        self.has_multiple_control_images = False
        self.use_raw_control_images = False
        self.latent_space_version = self.arch

    @staticmethod
    def get_train_scheduler():
        return AsymFlux2Klein9BModel.make_asymflow_scheduler({})

    @staticmethod
    def _is_full_finetune_path(path: Optional[str]) -> bool:
        return bool(path) and os.path.isdir(path) and os.path.isdir(os.path.join(path, "transformer"))

    @staticmethod
    def _normalize_full_finetune_output_path(output_path: str) -> str:
        if output_path.endswith(".safetensors"):
            return os.path.splitext(output_path)[0]
        return output_path

    @staticmethod
    def make_asymflow_scheduler(model_kwargs: Optional[dict]):
        model_kwargs = model_kwargs or {}
        shift = float(model_kwargs.get("asymflow_shift", 17.0))
        max_shift = float(model_kwargs.get("asymflow_max_shift", 34.0))
        base_scheduler = model_kwargs.get("asymflow_base_scheduler", "UniPCMultistep")
        return FlowAdapterScheduler(
            shift=shift,
            use_dynamic_shifting=True,
            base_seq_len=1024 ** 2,
            max_seq_len=2048 ** 2,
            base_logshift=math.log(shift),
            max_logshift=math.log(max_shift),
            dynamic_shifting_type="sqrt",
            base_scheduler=base_scheduler,
        )

    def get_base_model_version(self):
        return "asymflux2_klein_9b"

    def get_bucket_divisibility(self):
        return 16

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks", "single_transformer_blocks"]

    def _get_transformer_cache_key(self, transformer_path: str):
        model_kwargs = self.model_config.model_kwargs or {}
        base_model_name_or_path = model_kwargs.get("base_model_name_or_path", ASYMFLUX2_BASE_MODEL)
        full_finetune = self._is_full_finetune_path(self.model_config.name_or_path)
        return quantized_cache_key(
            "asymflux2_transformer",
            {
                "arch": self.arch,
                "base_model_version": self.get_base_model_version(),
                "base_model_name_or_path": base_model_name_or_path,
                "class": self.__class__.__name__,
                "dtype": str(self.torch_dtype),
                "full_finetune": full_finetune,
                "model_path": self.model_config.name_or_path,
                "qtype": self.model_config.qtype,
                "quantize_kwargs": self.model_config.quantize_kwargs,
            },
            sources=[base_model_name_or_path, transformer_path],
        )

    def _load_transformer_quantized_cache(self, transformer, transformer_path: str) -> bool:
        if not self._can_use_quantized_cache(self.model_config.qtype, "transformer"):
            return False

        cache_key, _ = self._get_transformer_cache_key(transformer_path)
        cache = self._get_quantized_cache()
        if not cache.has_entry("asymflux2_transformer", cache_key):
            return False

        try:
            self.print_and_status_update("Loading AsymFLUX transformer quantized cache")
            cache.load(
                transformer,
                "asymflux2_transformer",
                cache_key,
                device=torch.device("cpu"),
            )
            return True
        except Exception as e:
            self.print_and_status_update(
                f"Failed to load AsymFLUX transformer quantized cache, rebuilding: {e}"
            )
            return False

    def _save_transformer_quantized_cache(self, transformer, transformer_path: str):
        if not self._can_use_quantized_cache(self.model_config.qtype, "transformer"):
            return

        cache_key, key_payload = self._get_transformer_cache_key(transformer_path)
        try:
            self.print_and_status_update("Saving AsymFLUX transformer quantized cache")
            self._get_quantized_cache().save(
                transformer,
                "asymflux2_transformer",
                cache_key,
                key_payload,
                extra_metadata={"source_path": transformer_path},
            )
        except Exception as e:
            self.print_and_status_update(
                f"Failed to save AsymFLUX transformer quantized cache: {e}"
            )

    def load_te(self):
        if self.flux2_klein_te_path is None:
            raise ValueError("flux2_klein_te_path must be set for AsymFlux2Klein9BModel")
        dtype = self.torch_dtype
        text_encoder = None
        if self.model_config.quantize_te:
            text_encoder = self._load_qwen_quantized_cache()

        if text_encoder is None:
            self.print_and_status_update("Loading Qwen3")
            text_encoder: Qwen3ForCausalLM = Qwen3ForCausalLM.from_pretrained(
                self.flux2_klein_te_path,
                torch_dtype=dtype,
            )
        if self.model_config.quantize_te:
            if not getattr(text_encoder, "_aitk_loaded_from_quantized_cache", False):
                self.print_and_status_update("Quantizing Qwen3")
                quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
                freeze(text_encoder)
                self._save_qwen_quantized_cache(text_encoder)
            flush()
        elif not self.model_config.low_vram:
            text_encoder.to(self.device_torch, dtype=dtype)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        tokenizer_from_cache = getattr(text_encoder, "_aitk_loaded_from_quantized_cache", False)
        try:
            tokenizer = Qwen2TokenizerFast.from_pretrained(
                self.flux2_klein_te_path,
                local_files_only=tokenizer_from_cache,
            )
        except Exception:
            if not tokenizer_from_cache:
                raise
            tokenizer = Qwen2TokenizerFast.from_pretrained(self.flux2_klein_te_path)
        return text_encoder, tokenizer

    def _load_adapter_transformer(self, base_model_name_or_path: str, adapter_path: str):
        self.print_and_status_update("Loading FLUX.2 Klein base transformer")
        transformer = Flux2Transformer2DModel.from_pretrained(
            base_model_name_or_path,
            subfolder="transformer",
            torch_dtype=self.torch_dtype,
        )
        bridge = LakonLabPixelFlux2KleinPipeline(
            scheduler=self.make_asymflow_scheduler(self.model_config.model_kwargs),
            vae=self._make_oklab_encoder(),
            text_encoder=None,
            tokenizer=None,
            transformer=transformer,
            is_distilled=False,
        )
        self.print_and_status_update("Applying AsymFlow adapter")
        bridge.load_lakonlab_adapter(
            adapter_path,
            target_module_name="transformer",
            token=os.getenv("HF_TOKEN", None),
            use_safetensors=True,
        )
        return bridge.transformer

    def _load_full_finetune_transformer(self, model_path: str):
        self.print_and_status_update("Loading AsymFLUX full-finetune transformer")
        return AsymFlux2Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.torch_dtype,
        )

    def _make_oklab_encoder(self):
        model_kwargs = self.model_config.model_kwargs or {}
        return OklabColorEncoder(
            use_affine_norm=True,
            mean=tuple(model_kwargs.get("oklab_mean", [0.56, 0.0, 0.01])),
            std=float(model_kwargs.get("oklab_std", 0.16)),
        )

    def load_model(self):
        dtype = self.torch_dtype
        model_kwargs = self.model_config.model_kwargs or {}
        base_model_name_or_path = model_kwargs.get("base_model_name_or_path", ASYMFLUX2_BASE_MODEL)
        model_path = self.model_config.name_or_path or ASYMFLUX2_ADAPTER
        is_full_finetune = self._is_full_finetune_path(model_path)

        if is_full_finetune:
            transformer = self._load_full_finetune_transformer(model_path)
            transformer_cache_source = os.path.join(model_path, "transformer")
        else:
            transformer = self._load_adapter_transformer(base_model_name_or_path, model_path)
            transformer_cache_source = model_path

        transformer = transformer.to("cpu", dtype=dtype)

        loaded_transformer_from_cache = False
        if self.model_config.quantize:
            loaded_transformer_from_cache = self._load_transformer_quantized_cache(
                transformer,
                transformer_cache_source,
            )

        if self.model_config.quantize:
            if not loaded_transformer_from_cache:
                self.print_and_status_update("Quantizing AsymFLUX transformer")
                quantize_model(self, transformer)
                self._save_transformer_quantized_cache(transformer, transformer_cache_source)
            flush()
        elif not self.model_config.low_vram:
            transformer.to(self.device_torch, dtype=dtype)

        flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")

        text_encoder, tokenizer = self.load_te()
        vae = self._make_oklab_encoder()
        self.noise_scheduler = self.make_asymflow_scheduler(model_kwargs)

        self.print_and_status_update("Making AsymFLUX pipe")
        pipe = PixelFlux2KleinPipeline(
            scheduler=self.noise_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            transformer=transformer,
            is_distilled=False,
        )

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        flush()
        if self.model_config.low_vram:
            text_encoder[0].to("cpu")
            pipe.transformer = pipe.transformer.to("cpu")
        else:
            text_encoder[0].to(self.device_torch)
            pipe.transformer = pipe.transformer.to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        pipe.vae.requires_grad_(False)
        pipe.vae.eval()
        flush()

        self.vae = pipe.vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        pipeline = PixelFlux2KleinPipeline(
            scheduler=self.make_asymflow_scheduler(self.model_config.model_kwargs),
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
            is_distilled=False,
        )
        return pipeline.to(self.device_torch)

    def generate_single_image(
        self,
        pipeline: PixelFlux2KleinPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        gen_config.width = (gen_config.width // self.get_bucket_divisibility()) * self.get_bucket_divisibility()
        gen_config.height = (gen_config.height // self.get_bucket_divisibility()) * self.get_bucket_divisibility()

        extra = dict(extra)
        extra["negative_prompt_embeds"] = unconditional_embeds.text_embeds
        extra.setdefault("orthogonal_guidance", 1.0)
        extra.setdefault("clamp_denoised", True)

        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            **extra,
        ).images[0]
        return img

    def _prompt_text_ids(self, text_embeddings: PromptEmbeds):
        if text_embeddings.attention_mask is not None:
            return text_embeddings.attention_mask
        return self.pipeline._prepare_text_ids(text_embeddings.text_embeds).to(self.device_torch)

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: PromptEmbeds,
        guidance_embedding_scale: float,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        prompt_embeds = text_embeddings.text_embeds
        text_ids = self._prompt_text_ids(text_embeddings)
        if timestep.dim() == 0:
            timestep = timestep.expand(latent_model_input.shape[0])
        timestep = timestep.to(self.device_torch, dtype=self.torch_dtype) / 1000

        with self.transformer.cache_context("train"):
            noise_pred = self.transformer(
                x_t=latent_model_input.to(self.device_torch, dtype=self.transformer.dtype),
                timestep=timestep,
                encoder_hidden_states=prompt_embeds.to(self.device_torch, dtype=self.transformer.dtype),
                condition_latents=None,
                txt_ids=text_ids.to(self.device_torch),
                guidance=None,
            )

        if isinstance(noise_pred, QTensor):
            noise_pred = noise_pred.dequantize()
        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt,
            device=self.device_torch,
        )
        return PromptEmbeds(prompt_embeds, attention_mask=text_ids)

    def get_model_has_grad(self):
        return any(param.requires_grad for param in self.model.parameters())

    def get_te_has_grad(self):
        return any(param.requires_grad for param in self.text_encoder[0].parameters())

    def save_model(self, output_path, meta, save_dtype):
        output_path = self._normalize_full_finetune_output_path(output_path)
        os.makedirs(output_path, exist_ok=True)
        transformer: AsymFlux2Transformer2DModel = unwrap_model(self.model)

        original_dtype = next(transformer.parameters()).dtype
        transformer.to("cpu", dtype=save_dtype)
        transformer.save_pretrained(
            os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )
        transformer.to(self.device_torch if not self.model_config.low_vram else "cpu", dtype=original_dtype)

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def convert_lora_weights_before_save(self, state_dict):
        return state_dict

    def convert_lora_weights_before_load(self, state_dict):
        return state_dict

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = torch.float32

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        if isinstance(image_list, torch.Tensor):
            images = image_list.to(device, dtype=dtype)
        else:
            images = torch.stack([image.to(device, dtype=dtype) for image in image_list]).to(device, dtype=dtype)
        return self.vae.encode(images)

    def decode_latents(self, latents, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = torch.float32

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        latents = latents.to(device, dtype=dtype)
        return self.vae.decode(latents).clamp(-1, 1)
