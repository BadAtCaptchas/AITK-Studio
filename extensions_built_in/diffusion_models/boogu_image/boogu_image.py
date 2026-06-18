import os
import sys
from typing import TYPE_CHECKING, List, Optional, Sequence

import torch
import torch.nn.functional as F
import yaml
from diffusers import AutoencoderKL
from optimum.quanto import freeze
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.image_io import open_static_image
from toolkit.memory_management import attach_layer_offloading
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.util.quantize import get_qtype, quantize, quantize_model

VENDORED_SRC = os.path.join(os.path.dirname(__file__), "src")
if VENDORED_SRC not in sys.path:
    sys.path.insert(0, VENDORED_SRC)

from boogu.models.transformers import BooguImageTransformer2DModel
from boogu.models.transformers.rope import BooguImageRotaryPosEmbed
from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline
from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline
from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
    FlowMatchEulerDiscreteScheduler as BooguFlowMatchEuler,
)

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


BOOGU_BASE_MODEL_PATH = "Boogu/Boogu-Image-0.1-Base"
BOOGU_EDIT_MODEL_PATH = "Boogu/Boogu-Image-0.1-Edit"
BOOGU_TURBO_MODEL_PATH = "Boogu/Boogu-Image-0.1-Turbo"

scheduler_config = {"num_train_timesteps": 1000}


def _is_fp8_repo(model_ref: Optional[str]) -> bool:
    if not model_ref:
        return False
    normalized = str(model_ref).replace("\\", "/").lower()
    return normalized.endswith("-fp8") or (
        "/boogu-image-0.1-" in normalized and "-fp8" in normalized
    )


def _as_prompt_list(prompt) -> List[str]:
    if isinstance(prompt, str):
        return [prompt]
    return list(prompt)


def _pil_from_tensor(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu()
    if tensor.dim() == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Expected a single control image tensor, got a batch.")
        tensor = tensor[0]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    return to_pil_image(tensor).convert("RGB")


def _pil_from_value(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, torch.Tensor):
        return _pil_from_tensor(value)
    return open_static_image(value, mode="RGB")


def _is_fake_text_encoder(module) -> bool:
    return module is not None and module.__class__.__name__ == "FakeTextEncoder"


class BooguImageModel(BaseModel):
    arch = "boogu_image"
    default_model_path = BOOGU_BASE_MODEL_PATH
    encode_control_in_text_embeddings = False
    has_multiple_control_images = False
    use_raw_control_images = False
    boogu_task_type = "t2i"

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
        self.target_lora_modules = ["BooguImageTransformer2DModel"]
        self._control_latent = None
        self.freqs_cis = None
        self.mllm = None

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def _assert_supported_model_ref(self, model_ref: Optional[str]):
        if _is_fp8_repo(model_ref):
            raise ValueError(
                "Boogu fp8 repositories are inference-oriented and are not supported for "
                "AITK training. Use the non-fp8 Boogu repo with model.quantize: true instead."
            )

    def get_pipeline_cls(self):
        return BooguImagePipeline

    def _resolve_component_paths(self):
        model_path = self.model_config.name_or_path or self.default_model_path
        extras_path = self.model_config.extras_name_or_path or self.default_model_path

        base_model_path = extras_path
        transformer_path = model_path
        transformer_subfolder = "transformer"

        if model_path.endswith(".safetensors") or model_path.endswith(".bin"):
            raise ValueError(
                "Boogu-Image expects a Diffusers-format model directory or Hugging Face repo. "
                "Single-file transformer checkpoints are not supported yet."
            )

        if os.path.isdir(model_path):
            if os.path.isdir(os.path.join(model_path, "mllm")):
                base_model_path = model_path
            if os.path.isdir(os.path.join(model_path, "transformer")):
                transformer_subfolder = "transformer"
            else:
                transformer_subfolder = None

        return base_model_path, transformer_path, transformer_subfolder

    def _load_pipeline(self):
        (
            base_model_path,
            transformer_path,
            transformer_subfolder,
        ) = self._resolve_component_paths()
        self._assert_supported_model_ref(base_model_path)
        self._assert_supported_model_ref(transformer_path)

        self.print_and_status_update("Loading Boogu Qwen3-VL processor")
        processor = Qwen3VLProcessor.from_pretrained(
            base_model_path,
            subfolder="processor",
            use_fast=True,
        )

        self.print_and_status_update("Loading Boogu Qwen3-VL model")
        mllm = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path,
            subfolder="mllm",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )

        self.print_and_status_update("Loading Boogu transformer")
        transformer = BooguImageTransformer2DModel.from_pretrained(
            transformer_path,
            subfolder=transformer_subfolder,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )

        self.print_and_status_update("Loading Boogu VAE")
        vae = AutoencoderKL.from_pretrained(
            base_model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )

        self.print_and_status_update("Loading Boogu scheduler")
        scheduler = BooguFlowMatchEuler.from_pretrained(
            base_model_path,
            subfolder="scheduler",
        )
        self.generation_scheduler_config = scheduler.config

        pipe = self.get_pipeline_cls()(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mllm=mllm,
            processor=processor,
        )
        pipe.text_instruction_rewriter = None
        pipe.instruction_rewriter_processor = None
        return pipe

    def _prepare_freqs_cis(self, transformer: torch.nn.Module):
        if hasattr(self.pipeline, "freqs_cis"):
            return self.pipeline.freqs_cis
        return BooguImageRotaryPosEmbed.get_freqs_cis(
            transformer.config.axes_dim_rope,
            transformer.config.axes_lens,
            theta=10000,
        )

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Boogu-Image model")

        pipe = self._load_pipeline()
        pipe.scheduler = self.get_train_scheduler()

        transformer = pipe.transformer
        mllm = pipe.mllm
        vae = pipe.vae
        processor = pipe.processor

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Boogu transformer")
            quantize_model(self, transformer)
            flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Qwen3-VL model")
            quantize(mllm, weights=get_qtype(self.model_config.qtype_te))
            freeze(mllm)
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

        mllm.requires_grad_(False)
        mllm.eval()
        vae.requires_grad_(False)
        vae.eval()

        if self.low_vram:
            transformer.to("cpu")
            mllm.to("cpu")
        else:
            transformer.to(self.device_torch, dtype=dtype)
            mllm.to(self.device_torch, dtype=dtype)

        vae.to(self.device_torch, dtype=dtype)

        self.vae = vae
        self.mllm = mllm
        self.text_encoder = [mllm]
        self.tokenizer = [processor]
        self.model = transformer
        self.pipeline = pipe
        self.noise_scheduler = pipe.scheduler
        self.freqs_cis = self._prepare_freqs_cis(transformer)

        self.print_and_status_update("Model Loaded")

    def _get_mllm_for_generation(self):
        if self.sample_prompts_cache is not None:
            return None
        if isinstance(self.text_encoder, list) and len(self.text_encoder) > 0:
            encoder = self.text_encoder[0]
            if not _is_fake_text_encoder(encoder):
                return unwrap_model(encoder)

        pipeline_mllm = getattr(self.pipeline, "mllm", None)
        if pipeline_mllm is not None and not _is_fake_text_encoder(pipeline_mllm):
            return unwrap_model(pipeline_mllm)
        if self.mllm is not None and not _is_fake_text_encoder(self.mllm):
            return unwrap_model(self.mllm)

        raise RuntimeError(
            "Boogu-Image text encoder is unloaded and sample prompt embeddings "
            "are not cached. Enable train.cache_text_embeddings before unloading "
            "the text encoder, or disable train.unload_text_encoder for sampling."
        )

    def get_generation_pipeline(self):
        scheduler = BooguFlowMatchEuler.from_config(self.generation_scheduler_config)
        pipeline = self.get_pipeline_cls()(
            transformer=unwrap_model(self.model),
            vae=unwrap_model(self.vae),
            scheduler=scheduler,
            mllm=self._get_mllm_for_generation(),
            processor=self.tokenizer[0],
        )
        pipeline.text_instruction_rewriter = None
        pipeline.instruction_rewriter_processor = None
        return pipeline.to(self.device_torch)

    def _sample_control_images(self, gen_config: GenerateImageConfig):
        input_images = None
        if gen_config.ctrl_img is not None:
            control_img = open_static_image(gen_config.ctrl_img, mode="RGB")
            if control_img.size != (gen_config.width, gen_config.height):
                control_img = control_img.resize(
                    (gen_config.width, gen_config.height), Image.BILINEAR
                )
            input_images = [[control_img]]

        multi_ctrl_imgs = getattr(gen_config, "multi_ctrl_imgs", None)
        if multi_ctrl_imgs:
            input_images = []
            image_group = []
            for control_path in multi_ctrl_imgs:
                control_img = open_static_image(control_path, mode="RGB")
                if control_img.size != (gen_config.width, gen_config.height):
                    control_img = control_img.resize(
                        (gen_config.width, gen_config.height), Image.BILINEAR
                    )
                image_group.append(control_img)
            input_images.append(image_group)
        return input_images

    def generate_single_image(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        input_images = self._sample_control_images(gen_config)
        instruction = gen_config.prompt or ""
        negative_instruction = gen_config.negative_prompt or ""

        text_guidance_scale = gen_config.guidance_scale
        image_guidance_scale = extra.pop("image_guidance_scale", 1.0)
        if self.arch == "boogu_image_turbo" and gen_config.guidance_scale == 0:
            text_guidance_scale = extra.pop("text_guidance_scale", 1.0)
            image_guidance_scale = 1.0
            extra.setdefault("empty_instruction_guidance_scale", 0.0)
            extra.setdefault("use_dmd_student_inference", True)

        call_kwargs = {
            "instruction": instruction,
            "negative_instruction": negative_instruction,
            "instruction_embeds": conditional_embeds.text_embeds,
            "instruction_attention_mask": conditional_embeds.attention_mask,
            "negative_instruction_embeds": unconditional_embeds.text_embeds,
            "negative_instruction_attention_mask": unconditional_embeds.attention_mask,
            "height": gen_config.height,
            "width": gen_config.width,
            "num_inference_steps": gen_config.num_inference_steps,
            "text_guidance_scale": text_guidance_scale,
            "image_guidance_scale": image_guidance_scale,
            "latents": gen_config.latents,
            "generator": generator,
            "input_images": input_images,
            "device": str(self.device_torch),
            "use_rewrite_text_instruction": False,
            **extra,
        }

        try:
            return pipeline(**call_kwargs).images[0]
        except TypeError as exc:
            message = str(exc)
            unexpected_device_kwarg = (
                "unexpected keyword argument 'device'" in message
                or 'unexpected keyword argument "device"' in message
            )
            if not unexpected_device_kwarg:
                raise
            call_kwargs.pop("device", None)
            return pipeline(**call_kwargs).images[0]

    def _control_images_to_pil_nested(self, control_images, batch_size: int):
        if control_images is None:
            return None
        if isinstance(control_images, torch.Tensor):
            if control_images.dim() == 3:
                return [[_pil_from_tensor(control_images)]]
            return [
                [_pil_from_tensor(control_images[idx])]
                for idx in range(control_images.shape[0])
            ]
        if isinstance(control_images, Sequence) and not isinstance(
            control_images, (str, bytes)
        ):
            if not control_images:
                return None
            first = control_images[0]
            if isinstance(first, Sequence) and not isinstance(
                first, (str, bytes, Image.Image, torch.Tensor)
            ):
                return [
                    [_pil_from_value(item) for item in group]
                    for group in control_images
                ]
            if batch_size == len(control_images) and all(
                isinstance(item, torch.Tensor) and item.dim() == 3 for item in control_images
            ):
                return [[_pil_from_value(item)] for item in control_images]
            return [[_pil_from_value(item) for item in control_images]]
        return [[_pil_from_value(control_images)]]

    def _get_prompt_embeds_with_pipeline(self, prompt: List[str], control_images=None):
        input_images = self._control_images_to_pil_nested(control_images, len(prompt))
        self.text_encoder_to(self.device_torch, dtype=self.torch_dtype)

        max_sequence_length = self.model_config.model_kwargs.get(
            "max_sequence_length", 1280
        )
        task_type = self.model_config.model_kwargs.get(
            "task_type",
            "ti2i" if input_images else self.boogu_task_type,
        )

        if hasattr(self.pipeline, "_get_instruction_feature_embeds"):
            (
                prompt_embeds,
                prompt_attention_mask,
            ) = self.pipeline._get_instruction_feature_embeds(
                instruction=prompt,
                input_pil_images=input_images,
                device=self.device_torch,
                max_sequence_length=max_sequence_length,
                truncate_instruction_sequence=False,
                task_type=task_type,
            )
            return prompt_embeds, prompt_attention_mask

        if hasattr(self.pipeline, "encode_prompt"):
            prompt_embeds, prompt_attention_mask, _, _ = self.pipeline.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
                device=self.device_torch,
                max_sequence_length=max_sequence_length,
            )
            return prompt_embeds, prompt_attention_mask

        raise AttributeError("Boogu pipeline does not expose prompt encoding helpers.")

    def get_prompt_embeds(self, prompt: str, control_images=None) -> PromptEmbeds:
        prompt_list = _as_prompt_list(prompt)
        prompt_embeds, prompt_attention_mask = self._get_prompt_embeds_with_pipeline(
            prompt_list, control_images=control_images
        )
        pe = PromptEmbeds(prompt_embeds)
        pe.attention_mask = prompt_attention_mask
        return pe

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        try:
            timestep = timestep.expand(latent_model_input.shape[0]).to(
                latent_model_input.dtype
            )
        except Exception:
            pass

        timestep = 1 - (timestep / 1000)
        model_pred = self.model(
            latent_model_input,
            timestep,
            text_embeddings.text_embeds,
            self.freqs_cis,
            text_embeddings.attention_mask,
            ref_image_hidden_states=self._control_latent,
        )

        if hasattr(model_pred, "sample"):
            model_pred = model_pred.sample
        if isinstance(model_pred, (tuple, list)):
            model_pred = model_pred[0]
        return model_pred

    def _get_target_hw(self, batch: "DataLoaderBatchDTO"):
        if batch.tensor is not None:
            return batch.tensor.shape[2], batch.tensor.shape[3]
        return batch.file_items[0].crop_height, batch.file_items[0].crop_width

    def _prepare_control_tensor(
        self, control_tensor: torch.Tensor, target_h: int, target_w: int
    ):
        if control_tensor.dim() == 3:
            control_tensor = control_tensor.unsqueeze(0)
        control_tensor = control_tensor.to(self.vae_device_torch, dtype=self.torch_dtype)
        if control_tensor.shape[2] != target_h or control_tensor.shape[3] != target_w:
            control_tensor = F.interpolate(
                control_tensor, size=(target_h, target_w), mode="bilinear"
            )
        return control_tensor * 2 - 1

    def condition_noisy_latents(
        self, latents: torch.Tensor, batch: "DataLoaderBatchDTO"
    ):
        self._control_latent = None
        target_h, target_w = self._get_target_hw(batch)

        with torch.no_grad():
            multi_controls = getattr(batch, "control_tensor_list", None)
            if multi_controls:
                self.vae.to(self.device_torch)
                ref_latents = []
                for control_group in multi_controls:
                    group_latents = []
                    for control_tensor in control_group:
                        prepared = self._prepare_control_tensor(
                            control_tensor, target_h, target_w
                        )
                        encoded = self.encode_images(list(prepared)).to(
                            latents.device, latents.dtype
                        )
                        group_latents.extend([latent for latent in encoded])
                    ref_latents.append(group_latents)
                self._control_latent = ref_latents
                return latents.detach()

            control_tensor = batch.control_tensor
            if control_tensor is not None:
                self.vae.to(self.device_torch)
                prepared = self._prepare_control_tensor(
                    control_tensor, target_h, target_w
                )
                control_latent = self.encode_images(list(prepared)).to(
                    latents.device, latents.dtype
                )
                self._control_latent = [[x] for x in control_latent]

        return latents.detach()

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer = unwrap_model(self.model)
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
        return (batch.latents - noise).detach()

    def get_transformer_block_names(self) -> Optional[List[str]]:
        blocks = [
            "noise_refiner",
            "context_refiner",
            "double_stream_layers",
            "single_stream_layers",
        ]
        if self.model_config.model_kwargs.get("use_image_refiner", False):
            blocks.insert(2, "ref_image_refiner")
        return blocks

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_sd[key.replace("transformer.", "diffusion_model.")] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_sd[key.replace("diffusion_model.", "transformer.")] = value
        return new_sd

    def get_base_model_version(self):
        return self.arch


class BooguImageEditModel(BooguImageModel):
    arch = "boogu_image_edit"
    default_model_path = BOOGU_EDIT_MODEL_PATH
    encode_control_in_text_embeddings = True
    has_multiple_control_images = True
    use_raw_control_images = True
    boogu_task_type = "ti2i"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return [
            "noise_refiner",
            "context_refiner",
            "ref_image_refiner",
            "double_stream_layers",
            "single_stream_layers",
        ]


class BooguImageTurboModel(BooguImageModel):
    arch = "boogu_image_turbo"
    default_model_path = BOOGU_TURBO_MODEL_PATH
    boogu_task_type = "t2i"

    def get_pipeline_cls(self):
        return BooguImageTurboPipeline
