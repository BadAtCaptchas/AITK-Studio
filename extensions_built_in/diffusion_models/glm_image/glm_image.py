import hashlib
import os
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import torch
import yaml
from diffusers import AutoencoderKL, GlmImagePipeline, GlmImageTransformer2DModel
from diffusers.pipelines.glm_image.pipeline_glm_image import (
    GlmImageForConditionalGeneration,
    GlmImageProcessor,
)
from optimum.quanto import freeze
from transformers import ByT5Tokenizer, T5EncoderModel

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.memory_management import MemoryManager
from toolkit.models.base_model import BaseModel
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.util.quantize import get_qtype, quantize, quantize_model

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.25,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 0.75,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


class GlmImageModel(BaseModel):
    arch = "glm_image"

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
        self.target_lora_modules = ["GlmImageTransformer2DModel"]
        self.te_padding_side = "left"
        self._prior_token_cache = {}

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 8 * 2  # GLM VAE scale factor, then transformer patch size.

    def _ensure_glm_transformers_available(self):
        if GlmImageForConditionalGeneration.__name__ == "PreTrainedModel":
            raise ImportError(
                "GLM-Image requires a Transformers build with "
                "GlmImageForConditionalGeneration and GlmImageProcessor. "
                "Install the repository requirements before loading glm_image."
            )

    def load_model(self):
        self._ensure_glm_transformers_available()

        dtype = self.torch_dtype
        self.print_and_status_update("Loading GLM-Image model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path

        self.print_and_status_update("Loading transformer")
        if model_path.endswith(".safetensors"):
            if base_model_path.endswith(".safetensors"):
                base_model_path = "zai-org/GLM-Image"
            transformer = GlmImageTransformer2DModel.from_single_file(
                model_path,
                config=base_model_path,
                subfolder="transformer",
                torch_dtype=dtype,
            )
            transformer.to(dtype)
        else:
            transformer_path = model_path
            transformer_subfolder = "transformer"
            if os.path.exists(transformer_path):
                transformer_subfolder = None
                transformer_path = os.path.join(transformer_path, "transformer")
                if os.path.exists(os.path.join(model_path, "text_encoder")):
                    base_model_path = model_path

            transformer = GlmImageTransformer2DModel.from_pretrained(
                transformer_path,
                subfolder=transformer_subfolder,
                torch_dtype=dtype,
                use_safetensors=True,
            )

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[transformer.image_projector],
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")

        flush()

        self.print_and_status_update("Loading text encoder")
        tokenizer = ByT5Tokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer"
        )
        text_encoder = T5EncoderModel.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
            use_safetensors=True,
        )

        self.print_and_status_update("Loading prior encoder")
        processor = GlmImageProcessor.from_pretrained(
            base_model_path, subfolder="processor"
        )
        vision_language_encoder = GlmImageForConditionalGeneration.from_pretrained(
            base_model_path,
            subfolder="vision_language_encoder",
            torch_dtype=dtype,
            use_safetensors=True,
        )

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )
            MemoryManager.attach(
                vision_language_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        text_encoder.to(self.device_torch, dtype=dtype)
        vision_language_encoder.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing text/prior encoders")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            quantize(
                vision_language_encoder, weights=get_qtype(self.model_config.qtype_te)
            )
            freeze(vision_language_encoder)
            flush()

        self.print_and_status_update("Loading VAE")
        vae = AutoencoderKL.from_pretrained(
            base_model_path,
            subfolder="vae",
            torch_dtype=dtype,
            use_safetensors=True,
        )

        self.noise_scheduler = GlmImageModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")
        pipe: GlmImagePipeline = GlmImagePipeline(
            scheduler=self.noise_scheduler,
            tokenizer=tokenizer,
            processor=processor,
            text_encoder=None,
            vision_language_encoder=None,
            vae=vae,
            transformer=None,
        )
        pipe.text_encoder = text_encoder
        pipe.vision_language_encoder = vision_language_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")
        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)

        text_encoder.requires_grad_(False)
        text_encoder.eval()
        vision_language_encoder.requires_grad_(False)
        vision_language_encoder.eval()
        flush()

        self.vae = vae
        self.text_encoder = [pipe.text_encoder, pipe.vision_language_encoder]
        self.tokenizer = [pipe.tokenizer]
        self.processor = pipe.processor
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = GlmImageModel.get_train_scheduler()

        pipeline: GlmImagePipeline = GlmImagePipeline(
            scheduler=scheduler,
            tokenizer=self.tokenizer[0],
            processor=self.processor,
            text_encoder=unwrap_model(self.text_encoder[0]),
            vision_language_encoder=unwrap_model(self.text_encoder[1]),
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )

        pipeline = pipeline.to(self.device_torch)
        return pipeline

    def generate_single_image(
        self,
        pipeline: GlmImagePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        prompt_embeds, negative_prompt_embeds = self._prepare_sampling_prompt_embeds(
            conditional_embeds,
            unconditional_embeds,
        )
        prompts = (
            gen_config.prompt
            if isinstance(gen_config.prompt, list)
            else [gen_config.prompt]
        )
        prior_token_ids = self._get_prior_tokens_for_prompts(
            prompts,
            height=gen_config.height,
            width=gen_config.width,
        )

        img = pipeline(
            prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prior_token_ids=prior_token_ids,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            **extra,
        ).images[0]
        return img

    def _prepare_sampling_prompt_embeds(
        self,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
    ):
        prompt_embeds = self._text_embeds_to_tensor(conditional_embeds)
        negative_prompt_embeds = self._text_embeds_to_tensor(unconditional_embeds)

        if (
            prompt_embeds.dim() == 3
            and negative_prompt_embeds.dim() == 3
            and prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]
        ):
            max_len = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])
            prompt_embeds = self._left_pad_prompt_embed_tensor(prompt_embeds, max_len)
            negative_prompt_embeds = self._left_pad_prompt_embed_tensor(
                negative_prompt_embeds,
                max_len,
            )

        return prompt_embeds, negative_prompt_embeds

    def _left_pad_prompt_embed_tensor(self, tensor: torch.Tensor, max_len: int):
        if tensor.shape[1] >= max_len:
            return tensor
        pad_shape = (tensor.shape[0], max_len - tensor.shape[1], *tensor.shape[2:])
        pad = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([pad, tensor], dim=1)

    def _text_embeds_to_tensor(self, prompt_embeds: AdvancedPromptEmbeds):
        text_embeds = prompt_embeds.text_embeds
        if isinstance(text_embeds, torch.Tensor):
            return text_embeds.to(self.device_torch, dtype=self.torch_dtype)

        normalized = []
        for item in text_embeds:
            if item.dim() == 3 and item.shape[0] == 1:
                item = item[0]
            normalized.append(item.to(self.device_torch, dtype=self.torch_dtype))

        max_len = max(item.shape[0] for item in normalized)
        padded = []
        for item in normalized:
            if item.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - item.shape[0],
                    item.shape[1],
                    device=item.device,
                    dtype=item.dtype,
                )
                item = torch.cat([pad, item], dim=0)
            padded.append(item.unsqueeze(0))
        return torch.cat(padded, dim=0)

    def _prior_cache_key(
        self, prompt: str, height: int, width: int, crop_coords: Tuple[int, int]
    ):
        return (prompt, int(height), int(width), int(crop_coords[0]), int(crop_coords[1]))

    def _prior_seed_for_key(self, key) -> int:
        digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**31)

    def _ensure_prior_encoder_on_device(self):
        encoder = self.pipeline.vision_language_encoder
        if getattr(encoder, "device", torch.device("cpu")) == torch.device("cpu"):
            encoder.to(self.device_torch)

    def _get_prior_tokens_for_prompts(
        self,
        prompts: Sequence[str],
        height: int,
        width: int,
        crop_coords: Tuple[int, int] = (0, 0),
    ):
        tokens = []
        for prompt in prompts:
            prompt = prompt or ""
            key = self._prior_cache_key(prompt, height, width, crop_coords)
            cached = self._prior_token_cache.get(key)
            if cached is None:
                self._ensure_prior_encoder_on_device()
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self._prior_seed_for_key(key))
                with torch.no_grad():
                    prior_token_ids, _, _ = self.pipeline.generate_prior_tokens(
                        prompt,
                        height=height,
                        width=width,
                        image=None,
                        device=self.device_torch,
                        generator=generator,
                    )
                cached = prior_token_ids[0].detach().to("cpu")
                self._prior_token_cache[key] = cached
            tokens.append(cached.to(self.device_torch, dtype=torch.long))
        return torch.stack(tokens, dim=0)

    def _build_prior_token_inputs(
        self,
        prompts: Sequence[str],
        latent_batch_size: int,
        pixel_height: int,
        pixel_width: int,
    ):
        original_batch_size = len(prompts)
        prior_token_ids = self._get_prior_tokens_for_prompts(
            prompts,
            height=pixel_height,
            width=pixel_width,
        )
        prior_token_drop = torch.zeros_like(prior_token_ids, dtype=torch.bool)

        if latent_batch_size == original_batch_size * 2:
            prior_token_ids = torch.cat([prior_token_ids, prior_token_ids], dim=0)
            prior_token_drop = torch.cat(
                [
                    torch.ones_like(prior_token_drop, dtype=torch.bool),
                    torch.zeros_like(prior_token_drop, dtype=torch.bool),
                ],
                dim=0,
            )
        elif latent_batch_size != original_batch_size:
            raise ValueError(
                "GLM prior token batch must match latent batch size or half of it "
                "for classifier-free guidance"
            )

        return prior_token_ids, prior_token_drop

    def _batch_prompts(
        self,
        batch: Optional["DataLoaderBatchDTO"],
        latent_batch_size: int,
        text_embeddings: AdvancedPromptEmbeds,
    ):
        if batch is not None:
            prompts = batch.get_caption_list()
            if latent_batch_size in (len(prompts), len(prompts) * 2):
                return prompts

        text_embeds = text_embeddings.text_embeds
        if isinstance(text_embeds, torch.Tensor):
            text_batch_size = text_embeds.shape[0]
        else:
            text_batch_size = len(text_embeds)

        if latent_batch_size == text_batch_size:
            return [""] * latent_batch_size
        if latent_batch_size * 2 == text_batch_size:
            return [""] * latent_batch_size
        if text_batch_size % 2 == 0 and latent_batch_size == text_batch_size:
            return [""] * (text_batch_size // 2)
        return [""] * latent_batch_size

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: AdvancedPromptEmbeds,
        batch: Optional["DataLoaderBatchDTO"] = None,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        batch_size, _, latent_height, latent_width = latent_model_input.shape
        pixel_height = latent_height * self.pipeline.vae_scale_factor
        pixel_width = latent_width * self.pipeline.vae_scale_factor

        prompts = self._batch_prompts(batch, batch_size, text_embeddings)
        prior_token_ids, prior_token_drop = self._build_prior_token_inputs(
            prompts,
            latent_batch_size=batch_size,
            pixel_height=pixel_height,
            pixel_width=pixel_width,
        )

        prompt_embeds = self._text_embeds_to_tensor(text_embeddings)
        target_size = torch.tensor(
            [[pixel_height, pixel_width]],
            device=self.device_torch,
            dtype=prompt_embeds.dtype,
        ).repeat(batch_size, 1)
        crop_coords = torch.zeros(
            (batch_size, 2), device=self.device_torch, dtype=prompt_embeds.dtype
        )

        model_timestep = timestep.to(self.device_torch)
        if model_timestep.ndim == 0:
            model_timestep = model_timestep.unsqueeze(0)
        model_timestep = (model_timestep - 1).to(self.device_torch)

        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(
                self.device_torch, dtype=self.torch_dtype
            ).detach(),
            encoder_hidden_states=prompt_embeds.detach(),
            prior_token_id=prior_token_ids.detach(),
            prior_token_drop=prior_token_drop,
            timestep=model_timestep.detach(),
            target_size=target_size,
            crop_coords=crop_coords,
            return_dict=False,
            **kwargs,
        )[0]

        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> AdvancedPromptEmbeds:
        if self.pipeline.text_encoder.device == torch.device("cpu"):
            self.pipeline.text_encoder.to(self.device_torch)

        prompts = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=prompts,
            do_classifier_free_guidance=False,
            device=self.device_torch,
            dtype=self.torch_dtype,
        )

        text_embeds = [prompt_embeds[i] for i in range(prompt_embeds.shape[0])]
        return AdvancedPromptEmbeds(text_embeds=text_embeds)

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer: GlmImageTransformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return self.arch

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks"]

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

    def _latent_stats(self, latents: torch.Tensor):
        latents_mean = getattr(self.vae.config, "latents_mean", None)
        latents_std = getattr(self.vae.config, "latents_std", None)
        if latents_mean is None or latents_std is None:
            return None, None

        channels = latents.shape[1]
        latents_mean = (
            torch.tensor(latents_mean)
            .view(1, channels, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(latents_std)
            .view(1, channels, 1, 1)
            .to(latents.device, latents.dtype)
        )
        return latents_mean, latents_std

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        images = torch.stack([image.to(device, dtype=dtype) for image in image_list])
        latents = self.vae.encode(images).latent_dist.sample()

        latents_mean, latents_std = self._latent_stats(latents)
        if latents_mean is not None and latents_std is not None:
            latents = (latents - latents_mean) / latents_std
        else:
            latents = latents * getattr(self.vae.config, "scaling_factor", 1.0)

        return latents.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = latents.to(device, dtype=dtype)
        latents_mean, latents_std = self._latent_stats(latents)
        if latents_mean is not None and latents_std is not None:
            latents = latents * latents_std + latents_mean
        else:
            latents = latents / getattr(self.vae.config, "scaling_factor", 1.0)

        return self.vae.decode(latents, return_dict=False)[0]
