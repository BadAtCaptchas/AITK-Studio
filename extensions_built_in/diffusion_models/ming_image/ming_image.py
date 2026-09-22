"""Native, denoiser-only LoRA adapters for Ming Image Design 0.1.

Inference semantics: vLLM-Omni PR 8021, a62d2ec999ae8fa669e8c22194c67575c0f2d3dc.
This module deliberately does not depend on the vLLM inference runtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.image_io import open_static_image
from toolkit.layered_output import LayeredImageOutput
from toolkit.models.base_model import BaseModel
from toolkit.sample_controls import sample_control_paths
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

from .src.transformer import MingImageTransformer2DModel


DESIGN_REPO = 'inclusionAI/Ming-Image-0.1-Design'
LAYER_REPO = 'inclusionAI/Ming-Image-0.1-Design-Layer'


class MingTrainScheduler(CustomFlowMatchEulerDiscreteScheduler):
    """Uniform training draws with the checkpoint's static flow shift of six."""

    def set_train_timesteps(self, num_timesteps, device, timestep_type='linear', **kwargs):
        if timestep_type not in ('linear', 'shift'):
            raise ValueError('Ming training supports timestep_type: linear or shift')
        sigma = torch.linspace(1.0, 0.001, num_timesteps, device=device, dtype=torch.float32)
        sigma = self.config.shift * sigma / (1 + (self.config.shift - 1) * sigma)
        self.timesteps = sigma * self.config.num_train_timesteps
        self.sigmas = torch.cat((sigma, torch.zeros(1, device=device)))
        self.timestep_type = timestep_type
        return self.timesteps


def resolve_ming_checkpoint(name_or_path: str, revision: str | None = None) -> str:
    """Materialize only model components in the configured global model store."""
    if Path(name_or_path).is_dir():
        return str(Path(name_or_path).resolve())
    from huggingface_hub import snapshot_download
    from toolkit.paths import MODELS_PATH
    patterns = ['mllm/*', 'mlp/*', 'connector/*', 'vae/*', 'transformer/*', 'scheduler/*']
    options = dict(repo_id=name_or_path, revision=revision, cache_dir=MODELS_PATH, allow_patterns=patterns)
    return snapshot_download(**options)


def ming_component_identity(root: str | Path, components: tuple[str, ...]) -> str:
    """Fingerprint local component revisions without rereading giant weights.

    Snapshot paths pin Hub revisions; file metadata also invalidates caches when
    components in a local checkout are replaced or updated at the same path.
    """
    root = Path(root).resolve(strict=True)
    digest = hashlib.sha256(str(root).encode('utf-8'))
    for component in sorted(components):
        for path in sorted((root / component).rglob('*')):
            if not path.is_file():
                continue
            stat = path.stat()
            identity = [path.relative_to(root).as_posix(), stat.st_size,
                        stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]
            digest.update(json.dumps(identity, separators=(',', ':')).encode('utf-8'))
    return digest.hexdigest()


def tensor_image(image: torch.Tensor, *, normalized: bool = False) -> Image.Image:
    image = image.detach().float().cpu()
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[0] not in (3, 4):
        raise ValueError('Ming reference must be a CHW RGB or RGBA image')
    if normalized:
        image = image / 2 + 0.5
    pixels = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(pixels)


class MingImageDesignModel(BaseModel):
    arch = 'ming_image_design'
    preserve_image_alpha = True
    layered = False

    def __init__(self, device, model_config, dtype='bf16', **kwargs):
        super().__init__(device, model_config, dtype=dtype, **kwargs)
        self.is_flow_matching = True
        self.is_transformer = True
        self.use_old_lokr_format = False
        self.target_lora_modules = ['ZImageTransformerBlock']
        self.vae_scale_factor = 8
        self.encode_control_in_text_embeddings = True
        self.has_multiple_control_images = False
        self._component_identity = model_config.name_or_path
        self._vae_identity = model_config.name_or_path
        self._conditioner = None

    @staticmethod
    def get_train_scheduler():
        return MingTrainScheduler(num_train_timesteps=1000, shift=6.0, use_dynamic_shifting=False)

    def get_bucket_divisibility(self):
        return 16

    def get_transformer_block_names(self):
        return MingImageTransformer2DModel.get_transformer_block_names()

    def get_latent_space_version(self):
        digest = hashlib.sha256(self._vae_identity.encode()).hexdigest()[:16]
        return super().get_latent_space_version() + f'_ming_rgba_mode_scale80064_v1_{digest}'

    def get_text_embedding_space_version(self):
        digest = hashlib.sha256(self._component_identity.encode()).hexdigest()[:16]
        return f'{self.arch}_conditioning_pr8021_v1_{digest}'

    def component_load_kwargs(self, role='transformer', dtype=None):
        result = super().component_load_kwargs(role, dtype)
        result['use_comfy_weights'] = False
        # The generic component mixin attaches the legacy manager. Attach the
        # requested block manager after quantization, with all three block lists.
        if role == 'transformer' and self.model_config.layer_offloading:
            result.update(offload=0.0, device='cpu')
        if role == 'vae':
            result['device'] = 'cpu'
        return result

    def load_model(self):
        from .src.conditioning import MingConditioner
        from .src.vae import MingImageVAE

        self.print_and_status_update('Loading Ming Image components')
        root = resolve_ming_checkpoint(self.model_config.name_or_path, self.model_config.model_kwargs.get('revision'))
        config_path = Path(root) / 'transformer' / 'config.json'
        if config_path.is_file():
            metadata = json.loads(config_path.read_text(encoding='utf-8'))
            expected = ('learned', True) if self.layered else ('zero_masked', False)
            actual = (metadata.get('alignment_padding_mode'), metadata.get('multi_frame_output'))
            if actual != expected:
                raise ValueError(f'{self.arch} checkpoint has incompatible padding/output metadata: {actual}')
        self._component_identity = ming_component_identity(root, ('mllm', 'mlp', 'connector', 'vae'))
        self._vae_identity = ming_component_identity(root, ('vae',))
        # The conditioner owns CPU master weights and stages modules on demand.
        self._conditioner = MingConditioner.load(root, device=self.device_torch, dtype=self.torch_dtype, offload=True)
        self.text_encoder = [self._conditioner]
        self.tokenizer = [self._conditioner.tokenizer]
        self.model = MingImageTransformer2DModel.load(root, **self.component_load_kwargs('transformer'))
        self.model.requires_grad_(False)
        if self.model_config.layer_offloading:
            from toolkit.memory_management.offload import attach_layer_offloading
            attach_layer_offloading(
                self, self.model, self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                block_paths=self.model.get_transformer_block_names(),
            )
        self.vae = MingImageVAE.load(root, **self.component_load_kwargs('vae'))
        self.vae.requires_grad_(False).eval().to('cpu')
        self.noise_scheduler = self.get_train_scheduler()
        self.pipeline = self.get_generation_pipeline()
        self.print_and_status_update('Ming Image loaded (experimental)')

    @torch.no_grad()
    def encode_images(self, image_list, device=None, dtype=None):
        device = device or self.vae_device_torch
        dtype = dtype or self.vae_torch_dtype
        images = image_list if isinstance(image_list, torch.Tensor) else torch.stack(list(image_list))
        grouped = images.ndim == 5  # B,F,C,H,W, independent images, never video compression
        if grouped:
            batch, frames = images.shape[:2]
            images = images.flatten(0, 1)
        if images.ndim != 4 or images.shape[1] not in (3, 4):
            raise ValueError('Ming expects BCHW images or BFCHW layer groups')
        if images.shape[-2] % 16 or images.shape[-1] % 16:
            raise ValueError('Ming image dimensions must be divisible by 16')
        self.vae.to(device=device, dtype=dtype).eval()
        encoded = []
        try:
            for image in images:
                image = image.to(device=device, dtype=dtype)
                if image.shape[0] == 3:
                    image = torch.cat((image, torch.ones_like(image[:1])), dim=0)
                latent = self.vae.encode(image[None, :, None]).latent_dist.mode()
                encoded.append(((latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor)[:, :, 0])
        finally:
            if self.model_config.low_vram:
                self.vae.to('cpu')
        latents = torch.cat(encoded, dim=0)
        if grouped:
            latents = latents.reshape(batch, frames, *latents.shape[1:]).transpose(1, 2).contiguous()
        return latents.to(device=device, dtype=dtype)

    @torch.no_grad()
    def decode_latents(self, latents, device=None, dtype=None):
        device = device or self.vae_device_torch
        dtype = dtype or self.vae_torch_dtype
        grouped = latents.ndim == 5
        if grouped:
            batch, _, frames, _, _ = latents.shape
            latents = latents.transpose(1, 2).flatten(0, 1)
        self.vae.to(device=device, dtype=dtype).eval()
        if self.model_config.low_vram:
            self.vae.enable_tiling()
        decoded = []
        try:
            for latent in latents:
                latent = latent[None, :, None].to(device=device, dtype=dtype)
                latent = latent / self.vae.config.scaling_factor + self.vae.config.shift_factor
                decoded.append(self.vae.decode(latent).sample[:, :, 0].cpu())
        finally:
            self.vae.disable_tiling()
            if self.model_config.low_vram:
                self.vae.to('cpu')
        images = torch.cat(decoded, dim=0)
        return images.reshape(batch, frames, *images.shape[1:]) if grouped else images

    @staticmethod
    def _references(control_images, count):
        if control_images is None:
            return [None] * count
        if isinstance(control_images, torch.Tensor):
            items = list(control_images)
        else:
            items = list(control_images)
        if len(items) != count:
            raise ValueError('Ming supports exactly one reference per prompt')
        output = []
        for item in items:
            if isinstance(item, list):
                if len(item) != 1:
                    raise ValueError('Ming supports exactly one reference per prompt')
                item = item[0]
            output.append(item if isinstance(item, Image.Image) else tensor_image(item))
        return output

    def get_prompt_embeds(self, prompt, control_images=None):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        references = self._references(control_images, len(prompts))
        queries, direct, reference_latents = [], [], []
        for caption, reference in zip(prompts, references):
            if reference is not None:
                pixels = torch.from_numpy(np.asarray(reference.convert('RGBA')).copy()).permute(2, 0, 1).float() / 127.5 - 1
                latent = self.encode_images([pixels])[0].detach().cpu()
            else:
                latent = torch.empty(0)
            if caption:
                query, details = self._conditioner.encode(caption, reference=reference)
            else:
                # Ming's unconditional distribution is zero conditioning, not an encoded negative prompt.
                query, details = torch.zeros(256, 2560), torch.zeros(1, 3840)
            queries.append(query.detach().cpu())
            direct.append(details.detach().cpu())
            reference_latents.append(latent)
        return AdvancedPromptEmbeds(text_embeds=queries, direct_embeds=direct, reference_latents=reference_latents)

    def load_cached_control_images(self, file_item):
        file_item.load_control_image()
        try:
            value = file_item.control_tensor
            return [value.unsqueeze(0)] if value is not None else None
        finally:
            file_item.cleanup_control()

    def load_sample_control_images(self, gen_config):
        paths = sample_control_paths(gen_config)
        if len(paths) > 1:
            raise ValueError('Ming supports one sample reference image')
        if not paths:
            return None
        image = open_static_image(paths[0], mode='RGBA').resize((gen_config.width, gen_config.height), Image.Resampling.LANCZOS)
        return [torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255]

    def get_noise_prediction(self, latent_model_input, timestep, text_embeddings, batch=None, **kwargs):
        value = latent_model_input.to(self.device_torch, self.torch_dtype)
        single = value.ndim == 4
        if single:
            value = value.unsqueeze(2)
        refs = []
        for reference in text_embeddings.reference_latents:
            refs.append(reference.to(value).unsqueeze(1) if reference.numel() else None)
        if self.layered and any(reference is None for reference in refs):
            raise ValueError('Ming Design-Layer requires a flattened reference image')
        t = 1 - timestep.to(device=value.device, dtype=torch.float32).reshape(-1) / 1000
        if t.numel() == 1:
            t = t.expand(value.shape[0])
        result = self.model(
            x=list(value.unbind()), t=t,
            cap_feats=[item.to(value) for item in text_embeddings.text_embeds],
            cap_feats_2=[item.to(value) for item in text_embeddings.direct_embeds],
            ref_x=refs if any(ref is not None for ref in refs) else None,
            return_dict=False,
        )[0]
        prediction = -torch.stack(result)
        if prediction.shape != value.shape:
            raise ValueError(f'Ming prediction shape {tuple(prediction.shape)} differs from targets {tuple(value.shape)}')
        return prediction.squeeze(2) if single else prediction

    def get_loss_target(self, *, noise, batch, **kwargs):
        return (noise - batch.latents).detach()

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def sample_memory_text_encode_components(self):
        # The conditioner and VAE stage themselves; never move the entire MLLM to CUDA.
        return ()

    def sample_memory_generate_components(self):
        return ('unet',)

    def get_generation_pipeline(self):
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, use_dynamic_shifting=True)
        return SimpleNamespace(scheduler=scheduler, transformer=self.model, vae=self.vae,
                               text_encoder=self.text_encoder[0] if self.text_encoder else None)

    def prepare_sample_prompt_context(self, gen_config):
        if self.layered:
            count = gen_config.num_layers or 2
            instruction = f'Decompose the image into {count} layers.'
            if not gen_config.prompt.startswith(instruction):
                gen_config.prompt = instruction + '\n' + gen_config.prompt

    def prepare_sample_image_config_for_encoding(self, gen_config):
        if gen_config.negative_prompt and gen_config.negative_prompt.strip():
            raise ValueError('Ming uses zero negative conditioning; negative prompts are unsupported')
        if gen_config.width % 16 or gen_config.height % 16:
            raise ValueError('Ming sample dimensions must be divisible by 16')
        if self.layered and not sample_control_paths(gen_config):
            raise ValueError('Ming Design-Layer samples require ctrl_img')
        self.prepare_sample_prompt_context(gen_config)

    def generate_images(self, image_configs, sampler=None, pipeline=None):
        for item in image_configs:
            self.prepare_sample_image_config_for_encoding(item)
            if item.output_ext != 'png':
                raise ValueError('Ming RGBA samples require sample.format: png')
        return super().generate_images(image_configs, sampler=sampler, pipeline=pipeline)

    @torch.no_grad()
    def generate_single_image(self, pipeline, gen_config, conditional_embeds, unconditional_embeds, generator, extra):
        frames = (gen_config.num_layers or 2) + 1 if self.layered else 1
        shape = (1, 16, frames, gen_config.height // 8, gen_config.width // 8)
        latents = torch.randn(shape, generator=generator, device='cpu', dtype=torch.float32).to(self.device_torch)
        if gen_config.latents is not None:
            if tuple(gen_config.latents.shape) != shape:
                raise ValueError('Ming supplied sample latents have the wrong shape')
            latents = gen_config.latents.to(self.device_torch, torch.float32)
        scheduler = pipeline.scheduler
        spatial_tokens = (shape[-2] // 2) * (shape[-1] // 2)
        mu = 0.5 + (spatial_tokens - 256) * (1.15 - 0.5) / (4096 - 256)
        scheduler.set_timesteps(gen_config.num_inference_steps, device=self.device_torch, mu=mu)
        negative = AdvancedPromptEmbeds(
            text_embeds=[torch.zeros_like(item) for item in conditional_embeds.text_embeds],
            direct_embeds=[torch.zeros_like(item) for item in conditional_embeds.direct_embeds],
            reference_latents=conditional_embeds.reference_latents,
        )
        for index, timestep in enumerate(scheduler.timesteps):
            check = getattr(self, 'sample_cancel_check', None)
            if callable(check):
                check()
            positive = self.get_noise_prediction(latents, timestep[None], conditional_embeds).float()
            prediction = positive
            if gen_config.guidance_scale > 0:
                unconditioned = self.get_noise_prediction(latents, timestep[None], negative).float()
                prediction = positive + gen_config.guidance_scale * (positive - unconditioned)
            latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0].float()
            self._emit_sample_step(latents, index, len(scheduler.timesteps))
        # Release transformer residency before the VAE decoder peaks.
        self.model.to('cpu')
        flush()
        images = self.decode_latents(latents)[0]
        pictures = [tensor_image(item, normalized=True) for item in images]
        return LayeredImageOutput(pictures[0], pictures[1:]) if self.layered else pictures[0]


class MingImageDesignLayerModel(MingImageDesignModel):
    arch = 'ming_image_design_layer'
    layered = True
    supports_layered_images = True
