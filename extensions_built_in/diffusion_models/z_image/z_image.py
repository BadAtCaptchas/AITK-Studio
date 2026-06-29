import os
from typing import List, Optional

import torch
import yaml
from toolkit.base_lora import fuse_base_lora_into_model, load_lora_network_for_model
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze
from toolkit.util.quantize import (
    quantize,
    get_qtype,
    quantize_model,
    dequantize_if_quantized,
)
from toolkit.memory_management import attach_layer_offloading
from toolkit.metadata import get_meta_for_safetensors
from ..flux2.sega import normalize_sega_config
from .sega import (
    apply_zimage_sega_rope_scale,
    build_zimage_sega_rope_scale,
    summarize_zimage_rope_scale,
)
from .paths import resolve_single_file_model_path
from safetensors.torch import load_file, save_file

from transformers import AutoTokenizer, Qwen3ForCausalLM
from diffusers import AutoencoderKL

try:
    from diffusers import ZImagePipeline
    from diffusers.models.transformers import ZImageTransformer2DModel
except ImportError:
    raise ImportError(
        "Diffusers is out of date. Update diffusers to the latest version by doing pip uninstall diffusers and then pip install -r requirements.txt"
    )


scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}

SINGLE_FILE_EXTRAS_REPO = "Tongyi-MAI/Z-Image-Turbo"


def convert_single_file_to_diffusers(state_dict):
    """Convert a Comfy/single-file Z-Image checkpoint to diffusers transformer keys."""
    new_sd = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.endswith(".attention.qkv.weight"):
            prefix = new_key[: -len(".attention.qkv.weight")]
            q, k_proj, v = torch.chunk(value, 3, dim=0)
            new_sd[prefix + ".attention.to_q.weight"] = q
            new_sd[prefix + ".attention.to_k.weight"] = k_proj
            new_sd[prefix + ".attention.to_v.weight"] = v
            continue
        new_key = new_key.replace(".attention.out.weight", ".attention.to_out.0.weight")
        new_key = new_key.replace(".attention.q_norm.weight", ".attention.norm_q.weight")
        new_key = new_key.replace(".attention.k_norm.weight", ".attention.norm_k.weight")
        if new_key.startswith("x_embedder."):
            new_key = "all_x_embedder.2-1." + new_key[len("x_embedder.") :]
        elif new_key.startswith("final_layer."):
            new_key = "all_final_layer.2-1." + new_key[len("final_layer.") :]
        new_sd[new_key] = value
    return new_sd


def convert_diffusers_to_single_file(state_dict):
    """Convert diffusers Z-Image transformer keys back to Comfy/single-file layout."""
    new_sd = {}
    qkv_cache = {}
    for key, value in state_dict.items():
        new_key = key
        matched_qkv = False
        for suffix in (
            ".attention.to_q.weight",
            ".attention.to_k.weight",
            ".attention.to_v.weight",
        ):
            if new_key.endswith(suffix):
                prefix = new_key[: -len(suffix)]
                cache = qkv_cache.setdefault(prefix, {})
                cache[suffix] = value
                if len(cache) == 3:
                    new_sd[prefix + ".attention.qkv.weight"] = torch.cat(
                        [
                            cache[".attention.to_q.weight"],
                            cache[".attention.to_k.weight"],
                            cache[".attention.to_v.weight"],
                        ],
                        dim=0,
                    )
                    del qkv_cache[prefix]
                matched_qkv = True
                break
        if matched_qkv:
            continue
        new_key = new_key.replace(".attention.to_out.0.weight", ".attention.out.weight")
        new_key = new_key.replace(".attention.norm_q.weight", ".attention.q_norm.weight")
        new_key = new_key.replace(".attention.norm_k.weight", ".attention.k_norm.weight")
        if new_key.startswith("all_x_embedder.2-1."):
            new_key = "x_embedder." + new_key[len("all_x_embedder.2-1.") :]
        elif new_key.startswith("all_final_layer.2-1."):
            new_key = "final_layer." + new_key[len("all_final_layer.2-1.") :]
        new_sd[new_key] = value
    return new_sd


class ZImageModel(BaseModel):
    arch = "zimage"

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
        self.target_lora_modules = ["ZImageTransformer2DModel"]
        self.is_single_file = False
        self.single_file_extras_repo = SINGLE_FILE_EXTRAS_REPO

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 8 * 2  # 8 for the VAE, 2 for patch size

    def load_training_adapter(self, transformer: ZImageTransformer2DModel):
        self.print_and_status_update("Loading assistant LoRA")
        loaded = load_lora_network_for_model(
            base_model=self,
            model_to_train=transformer,
            path=self.model_config.assistant_lora_path,
            label="Assistant LoRA",
            is_assistant_adapter=True,
            is_ara=True,
        )
        self.model_config.assistant_lora_path = loaded.path
        network = loaded.network
        self.print_and_status_update("Merging in assistant LoRA")

        network.merge_in(merge_weight=1.0)
        if not network.is_merged_in:
            raise ValueError("Assistant LoRA could not be merged into the loaded Z-Image model.")

        # mark it as not merged so inference ignores it.
        network.is_merged_in = False

        # add the assistant so sampler will activate it while sampling
        self.assistant_lora = network

        # deactivate lora during training
        self.assistant_lora.multiplier = -1.0
        self.assistant_lora.is_active = False

        # tell the model to invert assistant on inference since we want remove lora effects
        self.invert_assistant_lora = True

    @staticmethod
    def is_comfy_single_file_state_dict(state_dict):
        return any(
            key.endswith(".attention.qkv.weight")
            or key.startswith("x_embedder.")
            or key.startswith("final_layer.")
            for key in state_dict
        )

    def load_transformer(self, model_path, base_model_path, dtype):
        if model_path.endswith(".safetensors"):
            self.is_single_file = True
            if base_model_path.endswith(".safetensors"):
                base_model_path = self.single_file_extras_repo
            transformer_path = resolve_single_file_model_path(model_path)
            state_dict = load_file(transformer_path)
            if self.is_comfy_single_file_state_dict(state_dict):
                self.print_and_status_update(
                    "Model is a Comfy single-file checkpoint, loading with safetensors"
                )
                state_dict = convert_single_file_to_diffusers(state_dict)
                state_dict = {
                    key: value.to(dtype=dtype) if value.is_floating_point() else value
                    for key, value in state_dict.items()
                }
                config = ZImageTransformer2DModel.load_config(
                    base_model_path, subfolder="transformer"
                )
                with torch.device("meta"):
                    transformer = ZImageTransformer2DModel.from_config(config)
                transformer.load_state_dict(state_dict, assign=True)
                transformer.to(dtype)
                del state_dict
                flush()
            else:
                del state_dict
                transformer = ZImageTransformer2DModel.from_single_file(
                    transformer_path,
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
                # check if the path is a full checkpoint.
                te_folder_path = os.path.join(model_path, "text_encoder")
                # if we have the te, this folder is a full checkpoint, use it as the base
                if os.path.exists(te_folder_path):
                    base_model_path = model_path

            transformer = ZImageTransformer2DModel.from_pretrained(
                transformer_path, subfolder=transformer_subfolder, torch_dtype=dtype
            )

        return transformer, base_model_path

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading ZImage model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path

        self.print_and_status_update("Loading transformer")

        transformer, base_model_path = self.load_transformer(model_path, base_model_path, dtype)

        # load assistant lora if specified
        if self.model_config.base_lora_path is not None:
            self.print_and_status_update("Fusing Base LoRA")
            result = fuse_base_lora_into_model(self, transformer)
            if result is not None:
                self.print_and_status_update(
                    f"Fused Base LoRA into training base: {result.path} "
                    f"(strength={result.strength}, modules={result.num_modules})"
                )

        if self.model_config.assistant_lora_path is not None:
            self.load_training_adapter(transformer)
            # set qtype to be float8 if it is qfloat8
            if self.model_config.qtype == "qfloat8":
                self.model_config.qtype = "float8"

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
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
                ignore_modules=[
                    transformer.x_pad_token,
                    transformer.cap_pad_token,
                ]
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")

        flush()

        self.print_and_status_update("Text Encoder")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer", torch_dtype=dtype
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype
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

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        self.print_and_status_update("Loading VAE")
        vae = AutoencoderKL.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype
        )

        self.noise_scheduler = ZImageModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        kwargs = {}

        pipe: ZImagePipeline = ZImagePipeline(
            scheduler=self.noise_scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
            **kwargs,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        # leave it on cpu for now
        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # just to make sure everything is on the right device and dtype
        text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        flush()

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = ZImageModel.get_train_scheduler()

        pipeline: ZImagePipeline = ZImagePipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: ZImagePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)
        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            **extra,
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        sega_config: dict | None = None,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        self._last_sega_scale_stats = None
        sega_rope_scale = None
        normalized_sega_config = normalize_sega_config(sega_config)
        if normalized_sega_config["enabled"]:
            axes_dims = getattr(self.transformer, "axes_dims", [32, 48, 48])
            vae_scale_factor = getattr(self.pipeline, "vae_scale_factor", 8)
            sega_rope_scale = build_zimage_sega_rope_scale(
                latent_model_input,
                axes_dims,
                base_resolution=normalized_sega_config["base_resolution"],
                strength=normalized_sega_config["strength"],
                min_scale=normalized_sega_config["min_scale"],
                max_scale=normalized_sega_config["max_scale"],
                vae_scale_factor=vae_scale_factor,
            )
            self._last_sega_scale_stats = summarize_zimage_rope_scale(sega_rope_scale)

        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        timestep_model_input = (1000 - timestep) / 1000

        with apply_zimage_sega_rope_scale(self.transformer, sega_rope_scale):
            model_out_list = self.transformer(
                latent_model_input_list,
                timestep_model_input,
                text_embeddings.text_embeds,
            )[0]

        noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred

        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            device=self.device_torch,
        )
        pe = PromptEmbeds([prompt_embeds, None])
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer: ZImageTransformer2DModel = unwrap_model(self.model)
        if self.is_single_file:
            state_dict = transformer.state_dict()
            save_dict = {
                key: dequantize_if_quantized(value).clone().to("cpu", dtype=save_dtype)
                for key, value in state_dict.items()
            }
            save_dict = convert_diffusers_to_single_file(save_dict)

            if not output_path.endswith(".safetensors"):
                output_path += ".safetensors"
            meta = get_meta_for_safetensors(meta, name=self.arch)
            save_file(save_dict, output_path, metadata=meta)
        else:
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
        return "zimage"

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
