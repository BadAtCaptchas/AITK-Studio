# Minimal LakonLab-derived components needed for AsymFLUX.2 Klein support.
# Derived from LakonLab by Hansheng Chen, Copyright (c) 2026.

import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
from diffusers.models import AutoModel
from diffusers.models.modeling_utils import (
    _LOW_CPU_MEM_USAGE_DEFAULT,
    ContextManagers,
    ModelMixin,
    load_state_dict,
    no_init_weights,
)
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Modulation,
    Flux2PosEmbed,
    Flux2SingleTransformerBlock,
    Flux2TimestepGuidanceEmbeddings,
    Flux2Transformer2DModel,
    Flux2TransformerBlock,
)
from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    Flux2ImageProcessor,
    Flux2KleinPipeline,
    Flux2PipelineOutput,
)
from diffusers.quantizers import DiffusersAutoQuantizer
from diffusers.schedulers import SchedulerMixin
from diffusers.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    BaseOutput,
    _add_variant,
    _get_model_file,
    is_accelerate_available,
    is_torch_version,
    logging,
)
from diffusers.utils.torch_utils import empty_device_cache, randn_tensor
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

try:
    import accelerate
except Exception:  # pragma: no cover - accelerate is an install dependency in normal use
    accelerate = None

try:
    from diffusers.utils import apply_lora_scale
except Exception:  # pragma: no cover - older diffusers fallback
    def apply_lora_scale(_kwarg_name):
        def decorator(fn):
            return fn

        return decorator


logger = logging.get_logger(__name__)


class OklabColorEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        use_affine_norm: bool = True,
        mean: Tuple[float, float, float] = (0.5, 0.0, 0.0),
        std: float = 0.21,
    ):
        super().__init__()
        self.use_affine_norm = use_affine_norm
        self.register_buffer(
            "lrgb_to_lms",
            torch.tensor(
                [
                    [0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005],
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "lms_to_oklab",
            torch.tensor(
                [
                    [0.2104542553, 0.7936177850, -0.0040720468],
                    [1.9779984951, -2.4285922050, 0.4505937099],
                    [0.0259040371, 0.7827717662, -0.8086757660],
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer("oklab_to_lms", torch.linalg.inv(self.lms_to_oklab))
        self.register_buffer("lms_to_lrgb", torch.linalg.inv(self.lrgb_to_lms))
        if self.use_affine_norm:
            self.register_buffer("affine_mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("affine_std", torch.tensor(std, dtype=torch.float32))

    @property
    def dtype(self):
        return self.lrgb_to_lms.dtype

    def _buffer_like(self, name: str, reference: torch.Tensor):
        return getattr(self, name).to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def srgb_to_lrgb(srgb):
        a = 0.055
        return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + a) / (1 + a)) ** 2.4)

    @staticmethod
    def lrgb_to_srgb(lrgb):
        lrgb = lrgb.clamp(min=0)
        a = 0.055
        return torch.where(lrgb <= 0.0031308, lrgb * 12.92, (1 + a) * (lrgb ** (1 / 2.4)) - a)

    def lrgb_to_oklab(self, lrgb):
        lms = torch.einsum("ij,bj...->bi...", self._buffer_like("lrgb_to_lms", lrgb), lrgb).clamp(min=0)
        return torch.einsum("ij,bj...->bi...", self._buffer_like("lms_to_oklab", lms), lms.pow(1 / 3))

    def oklab_to_lrgb(self, oklab):
        lms = torch.einsum("ij,bj...->bi...", self._buffer_like("oklab_to_lms", oklab), oklab).pow(3)
        lrgb = torch.einsum("ij,bj...->bi...", self._buffer_like("lms_to_lrgb", lms), lms)
        return lrgb.clamp(0, 1)

    def encode(self, img):
        rgb = img / 2 + 0.5
        lrgb = self.srgb_to_lrgb(rgb)
        oklab = self.lrgb_to_oklab(lrgb)
        if self.use_affine_norm:
            n_dim = img.dim() - 2
            mean = self._buffer_like("affine_mean", oklab).reshape(-1, *([1] * n_dim))
            std = self._buffer_like("affine_std", oklab).reshape(-1, *([1] * n_dim))
            oklab = (oklab - mean) / std
        return oklab

    def decode(self, oklab):
        if self.use_affine_norm:
            n_dim = oklab.dim() - 2
            mean = self._buffer_like("affine_mean", oklab).reshape(-1, *([1] * n_dim))
            std = self._buffer_like("affine_std", oklab).reshape(-1, *([1] * n_dim))
            oklab = oklab * std + mean
        lrgb = self.oklab_to_lrgb(oklab)
        rgb = self.lrgb_to_srgb(lrgb)
        return rgb * 2 - 1


class AsymFlowCalibration(tuple):
    __slots__ = ()
    _fields = ("s", "k", "timestep", "sigma")

    def __new__(cls, s, k, timestep, sigma):
        return tuple.__new__(cls, (s, k, timestep, sigma))

    @property
    def s(self):
        return self[0]

    @property
    def k(self):
        return self[1]

    @property
    def timestep(self):
        return self[2]

    @property
    def sigma(self):
        return self[3]


class AsymFlowMixin:
    train_sigma_min = 1e-6

    def init_asymflow_buffers(self, patch_dim: int, base_rank: int):
        if patch_dim < base_rank:
            raise ValueError(f"patch_dim ({patch_dim}) must be >= base_rank ({base_rank})")
        eye = torch.eye(base_rank)
        self.register_buffer("proj_buffer", F.pad(eye, (0, 0, 0, patch_dim - base_rank)))
        self.register_buffer("scale_buffer", torch.tensor(1.0))

    def asymflow_calibration(self, timestep, batch_size: int, ndim: int):
        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            timestep = timestep.float()
            s = self.scale_buffer.float()
            sigma = timestep / self.num_timesteps
            k = 1 / (s + (1 - s) * sigma)
            cal_timestep = timestep * k
            sigma = sigma.expand(batch_size).reshape(batch_size, *((ndim - 1) * [1])).float()
            k = k.reshape(batch_size, *((ndim - 1) * [1]))
            return AsymFlowCalibration(s=s, k=k, timestep=cal_timestep, sigma=sigma)

    @staticmethod
    def orthogonal_decomposition(full_rank_state, proj_buffer):
        subspace = full_rank_state @ proj_buffer @ proj_buffer.T
        return subspace, full_rank_state - subspace

    def asymflow_velocity(self, u_a_packed, x_t_packed, calibration: AsymFlowCalibration):
        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            sigma_min = self.train_sigma_min if self.training else self.sigma_min
            u_a_packed = u_a_packed.float()
            x_t_packed = x_t_packed.float()
            proj_buffer = self.proj_buffer.float()
            u_a_subspace, u_a_complement = self.orthogonal_decomposition(u_a_packed, proj_buffer)
            x_t_subspace, x_t_complement = self.orthogonal_decomposition(x_t_packed, proj_buffer)
            sk = calibration.s * calibration.k
            sigma_clamped = calibration.sigma.clamp(min=sigma_min)
            u_subspace = sk * u_a_subspace + (1 - sk) / sigma_clamped * x_t_subspace
            u_complement = (x_t_complement + calibration.s * u_a_complement) / sigma_clamped
            return u_subspace + u_complement


class AsymFlux2Transformer2DModel(AsymFlowMixin, Flux2Transformer2DModel):
    @register_to_config
    def __init__(
        self,
        patch_size=16,
        in_channels: int = 3,
        base_rank: int = 128,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        sigma_min: float = 1e-4,
        num_timesteps=1,
        guidance_embeds: bool = True,
    ):
        super(Flux2Transformer2DModel, self).__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=guidance_embeds,
        )
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)
        self.x_embedder = nn.Linear(in_channels * (patch_size ** 2), self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)
        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False
        )
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels * (patch_size ** 2), bias=False)
        self.base_rank = base_rank
        self.sigma_min = sigma_min
        self.num_timesteps = num_timesteps
        self.init_asymflow_buffers(self.in_channels * (patch_size ** 2), self.base_rank)
        self.gradient_checkpointing = False

    @staticmethod
    def patchify(latents, patch_size, pack_channels=True):
        bs, c, h, w = latents.size()
        latents = latents.reshape(bs, c, h // patch_size, patch_size, w // patch_size, patch_size)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        if pack_channels:
            return latents.reshape(bs, c * patch_size * patch_size, h // patch_size, w // patch_size)
        return latents.reshape(bs, c, patch_size * patch_size, h // patch_size, w // patch_size)

    @staticmethod
    def unpatchify(latents, patch_size, packed_channels=True):
        if packed_channels:
            bs, c, h, w = latents.size()
            return latents.reshape(bs, c // (patch_size * patch_size), patch_size, patch_size, h, w).permute(
                0, 1, 4, 2, 5, 3
            ).reshape(bs, c // (patch_size * patch_size), h * patch_size, w * patch_size)
        bs, c, _, h, w = latents.size()
        return latents.reshape(bs, c, patch_size, patch_size, h, w).permute(0, 1, 4, 2, 5, 3).reshape(
            bs, c, h * patch_size, w * patch_size
        )

    @staticmethod
    def pack(latents):
        bs, c, h, w = latents.shape
        return latents.reshape(bs, c, h * w).permute(0, 2, 1)

    @staticmethod
    def unpack(latents, h, w):
        bs, _, c = latents.shape
        return latents.permute(0, 2, 1).reshape(bs, c, h, w)

    @staticmethod
    def _prepare_latent_ids(latents):
        batch_size, _, height, width = latents.shape
        latent_ids = torch.cartesian_prod(
            torch.arange(1),
            torch.arange(height),
            torch.arange(width),
            torch.arange(1),
        )
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return latent_ids.to(device=latents.device)

    @staticmethod
    def _prepare_condition_latent_ids(image_latents: List[torch.Tensor], scale: int = 10):
        if not isinstance(image_latents, list):
            raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            _, _, h, w = x.shape
            x_ids = torch.cartesian_prod(t.view(-1), torch.arange(h), torch.arange(w), torch.arange(1))
            image_latent_ids.append(x_ids)
        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0).expand(image_latents[0].size(0), -1, -1)
        return image_latent_ids.to(device=image_latents[0].device)

    def _get_rotary_emb(self, img_ids, txt_ids):
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            image_rotary_emb = self.pos_embed(img_ids.float())
            text_rotary_emb = self.pos_embed(txt_ids.float())
            return (
                torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
                torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
            )

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        condition_latents: Optional[List[torch.Tensor]] = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
    ):
        x_t = self.patchify(x_t, self.patch_size)
        img_ids = self._prepare_latent_ids(x_t)

        bs, _, h, w = x_t.size()
        x_t_packed = self.pack(x_t)
        num_x_tokens = x_t_packed.size(1)
        packed_ndim = x_t_packed.dim()

        calibration = self.asymflow_calibration(timestep, bs, packed_ndim)
        hidden_states = x_t_packed * calibration.k.to(x_t_packed.dtype)

        input_img_ids = img_ids
        if condition_latents is not None:
            condition_hidden_states = [self.patchify(z, self.patch_size) for z in condition_latents]
            condition_latent_ids = self._prepare_condition_latent_ids(condition_hidden_states)
            condition_hidden_states = [self.pack(z) / calibration.s for z in condition_hidden_states]
            hidden_states = torch.cat([hidden_states] + condition_hidden_states, dim=1)
            input_img_ids = torch.cat([img_ids, condition_latent_ids], dim=1)

        hidden_states = self.x_embedder(hidden_states)
        num_txt_tokens = encoder_hidden_states.shape[1]

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(calibration.timestep.to(hidden_states.dtype) * 1000, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        concat_rotary_emb = self._get_rotary_emb(input_img_ids, txt_ids)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=double_stream_mod_img,
                    temb_mod_txt=double_stream_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for block in self.single_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        hidden_states = hidden_states[:, num_txt_tokens : num_txt_tokens + num_x_tokens]
        hidden_states = self.norm_out(hidden_states, temb)
        u_a_packed = self.proj_out(hidden_states)
        output_packed = self.asymflow_velocity(u_a_packed, x_t_packed, calibration)
        output = self.unpack(output_packed.to(hidden_states.dtype), h, w)
        return self.unpatchify(output, self.patch_size)


_SET_ADAPTER_SCALE_FN_MAPPING.update(
    AsymFlux2Transformer2DModel=lambda model_cls, weights: weights,
)


@dataclass
class FlowWrapperSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowAdapterScheduler(SchedulerMixin, ConfigMixin):
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting=False,
        dynamic_shifting_type="exp",
        base_seq_len=256,
        max_seq_len=4096,
        base_logshift=0.5,
        max_logshift=1.15,
        terminal_sigma=None,
        base_scheduler="UniPCMultistep",
        eps=1e-4,
        **kwargs,
    ):
        sigmas = torch.from_numpy(1 - np.linspace(0, 1, num_train_timesteps, dtype=np.float32, endpoint=False))
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * num_train_timesteps
        alphas = 1 - self.sigmas

        base_scheduler_class = getattr(diffusers.schedulers, base_scheduler + "Scheduler", None)
        if base_scheduler_class is None:
            raise AttributeError(f"Cannot find base_scheduler [{base_scheduler}].")
        if base_scheduler in ["EulerDiscrete", "EulerAncestralDiscrete"]:
            if kwargs.get("prediction_type", "epsilon") != "epsilon":
                raise ValueError(f"{base_scheduler} requires prediction_type='epsilon'.")
            kwargs["prediction_type"] = "epsilon"
            self.scales = (
                (alphas ** 2 + self.sigmas ** 2) / (1 + (self.sigmas / alphas.clamp(min=eps)) ** 2)
            ).sqrt()
        elif base_scheduler in ["UniPCMultistep", "DPMSolverSinglestep", "DPMSolverMultistep", "DEISMultistep", "SASolver"]:
            self.scales = torch.ones_like(alphas)
            if kwargs.get("prediction_type", "flow_prediction") != "flow_prediction":
                raise ValueError(f"{base_scheduler} requires prediction_type='flow_prediction'.")
            kwargs["prediction_type"] = "flow_prediction"
            kwargs["use_flow_sigmas"] = True
        else:
            raise AttributeError(f"Unsupported base_scheduler [{base_scheduler}].")

        signatures = inspect.signature(base_scheduler_class).parameters.keys()
        if "final_sigmas_type" in signatures:
            kwargs["final_sigmas_type"] = "zero"
        if "lower_order_final" in signatures:
            kwargs["lower_order_final"] = True

        self.base_scheduler = base_scheduler_class(num_train_timesteps=num_train_timesteps, **kwargs)
        self.base_scheduler.timesteps = self.timesteps
        if self.config.base_scheduler in ["EulerDiscrete", "EulerAncestralDiscrete"]:
            self.base_scheduler.sigmas = self.sigmas / alphas.clamp(min=self.config.eps)
        else:
            self.base_scheduler.sigmas = self.sigmas

        self._step_index = None
        self._begin_index = None

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def get_shift(self, seq_len=None):
        if self.config.use_dynamic_shifting and seq_len is not None:
            if self.config.dynamic_shifting_type == "exp":
                m = (self.config.max_logshift - self.config.base_logshift) / (
                    self.config.max_seq_len - self.config.base_seq_len
                )
                logshift = (seq_len - self.config.base_seq_len) * m + self.config.base_logshift
                shift = torch.exp(logshift) if isinstance(logshift, torch.Tensor) else np.exp(logshift)
            elif self.config.dynamic_shifting_type == "sqrt":
                max_shift = np.exp(self.config.max_logshift)
                base_shift = np.exp(self.config.base_logshift)
                sqrt_max_seq_len = np.sqrt(self.config.max_seq_len)
                sqrt_base_seq_len = np.sqrt(self.config.base_seq_len)
                m = (max_shift - base_shift) / (sqrt_max_seq_len - sqrt_base_seq_len)
                shift = (np.sqrt(seq_len) - sqrt_base_seq_len) * m + base_shift
            else:
                raise ValueError(f"Unsupported dynamic_shifting_type [{self.config.dynamic_shifting_type}].")
        else:
            shift = self.config.shift
        return shift

    def stretch_to_terminal(self, sigma):
        one_minus_sigma = 1 - sigma
        return 1 - (one_minus_sigma * (1 - self.config.terminal_sigma) / one_minus_sigma[-1])

    def _shift_sigmas(self, sigmas, seq_len=None):
        shift = self.get_shift(seq_len=seq_len)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        if self.config.terminal_sigma is not None:
            sigmas = self.stretch_to_terminal(sigmas)
        return sigmas

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        sigmas: Optional[List[float]] = None,
        seq_len=None,
        device=None,
    ):
        if sigmas is None:
            if num_inference_steps is None:
                raise ValueError("Either num_inference_steps or sigmas must be provided.")
            self.num_inference_steps = num_inference_steps
            sigmas = np.linspace(1, 0, num_inference_steps, dtype=np.float32, endpoint=False)
        else:
            if num_inference_steps is not None and len(sigmas) != num_inference_steps:
                raise ValueError("len(sigmas) must match num_inference_steps.")
            self.num_inference_steps = len(sigmas)
            sigmas = np.array(sigmas, dtype=np.float32)

        sigmas = self._shift_sigmas(torch.from_numpy(sigmas), seq_len=seq_len)
        target_device = torch.device(device) if device is not None else sigmas.device
        sigmas = sigmas.to(device=target_device, dtype=torch.float32)
        self.timesteps = (sigmas * self.config.num_train_timesteps).to(device=target_device, dtype=torch.float32)
        if self.config.base_scheduler in ["DEISMultistep", "SASolver"]:
            self.sigmas = torch.cat([sigmas, torch.tensor([self.config.eps], dtype=sigmas.dtype, device=sigmas.device)])
        else:
            self.sigmas = torch.cat([sigmas, torch.zeros(1, dtype=sigmas.dtype, device=sigmas.device)])
        alphas = 1 - self.sigmas

        self.base_scheduler.set_timesteps(num_inference_steps, device=target_device)
        self.base_scheduler.timesteps = self.timesteps
        if self.config.base_scheduler in ["EulerDiscrete", "EulerAncestralDiscrete"]:
            self.base_scheduler.sigmas = self.sigmas / alphas.clamp(min=self.config.eps)
            self.scales = (
                (alphas ** 2 + self.sigmas ** 2) / (1 + (self.sigmas / alphas.clamp(min=self.config.eps)) ** 2)
            ).sqrt()
        elif self.config.base_scheduler in ["UniPCMultistep", "DPMSolverSinglestep", "DPMSolverMultistep", "DEISMultistep", "SASolver"]:
            self.base_scheduler.sigmas = self.sigmas.clamp(max=1 - self.config.eps)
            self.scales = torch.ones_like(alphas)
        else:
            raise AttributeError(f"Unsupported base_scheduler [{self.config.base_scheduler}].")

        self._step_index = None
        self._begin_index = None

    def set_train_timesteps(
        self,
        num_timesteps,
        device,
        timestep_type="linear",
        latents=None,
        patch_size=1,
    ):
        self.timestep_type = timestep_type
        if timestep_type in ["linear", "weighted"]:
            sigmas = torch.linspace(1, 0.001, num_timesteps, device=device)
        elif timestep_type == "sigmoid":
            t = torch.sigmoid(torch.randn((num_timesteps,), device=device))
            sigmas, _ = torch.sort(1 - t, descending=True)
        elif timestep_type in ["flux_shift", "lumina2_shift", "shift", "next_sample"]:
            if latents is None:
                raise ValueError("latents must be provided when using shifted AsymFlow timesteps.")
            h = latents.shape[2]
            w = latents.shape[3]
            # FlowAdapter shifts on pixel sequence length; the transformer patch size is separate.
            seq_len = h * w
            sigmas = torch.linspace(1, 0.001, num_timesteps, device=device)
            shifted = self._shift_sigmas(sigmas.detach().cpu(), seq_len=seq_len)
            sigmas = shifted.to(device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Invalid timestep type: {timestep_type}")

        timesteps = sigmas * self.config.num_train_timesteps
        self.timesteps = timesteps.to(device=device, dtype=torch.float32)
        self.sigmas = sigmas.to(device=device, dtype=torch.float32)
        return self.timesteps

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sigmas = (timesteps.float() / self.config.num_train_timesteps).to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        while sigmas.dim() < original_samples.dim():
            sigmas = sigmas.unsqueeze(-1)
        return (1.0 - sigmas) * original_samples + sigmas * noise

    def get_sigmas(self, timesteps: torch.Tensor, n_dim, dtype, device) -> torch.Tensor:
        sigma = (timesteps.float() / self.config.num_train_timesteps).to(device=device, dtype=dtype)
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        prediction_type="u",
        eps=1e-6,
    ) -> Union[FlowWrapperSchedulerOutput, Tuple]:
        if prediction_type not in ["u", "x0"]:
            raise ValueError("prediction_type must be 'u' or 'x0'.")
        if self.step_index is None:
            self._init_step_index(timestep)

        ori_dtype = model_output.dtype
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        alpha = 1 - sigma
        scale = self.scales[self.step_index]
        next_scale = self.scales[self.step_index + 1]

        if hasattr(self.base_scheduler, "is_scale_input_called"):
            self.base_scheduler.is_scale_input_called = True
        kwargs = {"return_dict": False}
        if generator is not None:
            kwargs["generator"] = generator

        if self.config.base_scheduler in ["UniPCMultistep", "DPMSolverSinglestep", "DPMSolverMultistep", "DEISMultistep", "SASolver"]:
            if prediction_type == "x0":
                model_output = (sample - model_output) / sigma.clamp(min=eps)
        else:
            if prediction_type == "u":
                model_output = sample + alpha * model_output
            else:
                model_output = (sample - alpha * model_output) / sigma.clamp(min=eps)
        prev_sample = self.base_scheduler.step(model_output, timestep, sample / scale, **kwargs)[0] * next_scale
        prev_sample = prev_sample.to(ori_dtype)
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return FlowWrapperSchedulerOutput(prev_sample=prev_sample)


def guidance_jit(pos_mean, neg_mean, guidance_scale, orthogonal: float = 1.0, parallel_dir: Optional[torch.Tensor] = None):
    bias = (pos_mean - neg_mean) * (guidance_scale - 1)
    if orthogonal:
        dim = list(range(1, pos_mean.dim()))
        if parallel_dir is None:
            parallel_dir = pos_mean
        bias = bias - (
            (bias * parallel_dir).mean(dim=dim, keepdim=True)
            / (parallel_dir * parallel_dir).mean(dim=dim, keepdim=True).clamp(min=1e-6)
            * parallel_dir
        ).mul(orthogonal)
    return bias


class PixelFlux2KleinPipeline(Flux2KleinPipeline):
    model_cpu_offload_seq = "text_encoder->transformer"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowAdapterScheduler,
        vae: OklabColorEncoder,
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        transformer: Flux2Transformer2DModel,
        is_distilled: bool = False,
    ):
        super(Flux2KleinPipeline, self).__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )
        self.register_to_config(is_distilled=is_distilled)
        self.vae_scale_factor = 1
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 16)
        self.tokenizer_max_length = 512
        self.default_sample_size = 1024

    def prepare_latents(
        self,
        batch_size,
        height,
        width,
        dtype,
        device,
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        height = 16 * (int(height) // (self.vae_scale_factor * 16))
        width = 16 * (int(width) // (self.vae_scale_factor * 16))
        shape = (batch_size, 3, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You passed {len(generator)} generators but requested batch size {batch_size}."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents

    def prepare_image_latents(self, images: list[torch.Tensor], batch_size, device, dtype):
        image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            image_latent = self.vae.encode(image).to(self.transformer.dtype)
            image_latents.append(image_latent.repeat(batch_size, 1, 1, 1))
        return image_latents

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[Any] = None,
        prompt: Union[str, list[str], None] = None,
        negative_prompt: Union[str, list[str], None] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[list[float]] = None,
        guidance_scale: float = 4.0,
        orthogonal_guidance: float = 1.0,
        clamp_denoised: bool = True,
        num_images_per_prompt: int = 1,
        generator: Union[torch.Generator, list[torch.Generator], None] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, dict], None]] = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int] = (9, 18, 27),
    ):
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if self.do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ""
            if prompt is not None and isinstance(prompt, list) and not isinstance(negative_prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )
            guidance_scale = torch.tensor(guidance_scale, device=device, dtype=torch.float32)

        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            condition_images = []
            for img in image:
                self.image_processor.check_image_input(img)
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size
                multiple_of = self.vae_scale_factor * 16
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            height=height,
            width=width,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        if condition_images is not None:
            image_latents = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                device=device,
                dtype=self.vae.dtype,
            )

        image_seq_len = latents.shape[2:].numel()
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, seq_len=image_seq_len, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)
        self.scheduler.set_begin_index(0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                _t = t / 1000
                timestep = _t.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = latents.to(self.transformer.dtype)

                with self.transformer.cache_context("cond"):
                    denoising_output = self.transformer(
                        x_t=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        condition_latents=image_latents,
                        txt_ids=text_ids,
                        guidance=None,
                        joint_attention_kwargs=self.attention_kwargs,
                    ).float()

                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_denoising_output = self.transformer(
                            x_t=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            condition_latents=image_latents,
                            txt_ids=negative_text_ids,
                            guidance=None,
                            joint_attention_kwargs=self._attention_kwargs,
                        ).float()
                    cfg_bias = guidance_jit(
                        denoising_output,
                        neg_denoising_output,
                        guidance_scale,
                        orthogonal_guidance,
                        latents - denoising_output * _t,
                    )
                    denoising_output = denoising_output + cfg_bias

                if clamp_denoised:
                    denoised = latents - denoising_output * _t
                    image_tensor = self.vae.decode(denoised.to(self.vae.dtype)).clamp(-1, 1)
                    denoised = self.vae.encode(image_tensor).to(latents.dtype)
                    denoising_output = (latents - denoised) / _t.clamp(min=1e-4)

                latents = self.scheduler.step(denoising_output, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                progress_bar.update()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents.to(self.vae.dtype))
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return Flux2PipelineOutput(images=image)


LOCAL_CLASS_MAPPING = {
    "AsymFlux2Transformer2DModel": AsymFlux2Transformer2DModel,
}


def _assign_tensor(module: nn.Module, tensor_name: str, tensor: torch.Tensor):
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            module = getattr(module, split)
        tensor_name = splits[-1]
    if tensor_name in module._parameters:
        old_param = module._parameters[tensor_name]
        requires_grad = False if old_param is None else old_param.requires_grad
        module._parameters[tensor_name] = nn.Parameter(tensor, requires_grad=requires_grad)
    elif tensor_name in module._buffers:
        module._buffers[tensor_name] = tensor
    else:
        setattr(module, tensor_name, tensor)


class LakonLabMixin:
    def load_lakonlab_adapter(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        target_module_name: str = "transformer",
        adapter_name: Optional[str] = None,
        **kwargs,
    ):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        disable_mmap = kwargs.pop("disable_mmap", False)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError("Low memory initialization requires torch >= 1.9.0.")
        if low_cpu_mem_usage and accelerate is None:
            low_cpu_mem_usage = False

        user_agent = {"diffusers": diffusers.__version__, "file_type": "model", "framework": "pytorch"}
        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
        }
        config = AutoModel.load_config(pretrained_model_name_or_path, subfolder=subfolder, **load_config_kwargs)
        orig_class_name = config["_class_name"]
        if orig_class_name in LOCAL_CLASS_MAPPING:
            model_cls = LOCAL_CLASS_MAPPING[orig_class_name]
        else:
            from diffusers.pipelines.pipeline_loading_utils import ALL_IMPORTABLE_CLASSES, get_class_obj_and_candidates

            model_cls, _ = get_class_obj_and_candidates(
                library_name="diffusers",
                class_name=orig_class_name,
                importable_classes=ALL_IMPORTABLE_CLASSES,
                pipelines=None,
                is_pipeline_module=False,
            )
        if model_cls is None:
            raise ValueError(f"Cannot find a model class for {orig_class_name}.")

        model_file = None
        if use_safetensors:
            try:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
            except IOError:
                if not allow_pickle:
                    raise
        if model_file is None:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=_add_variant(WEIGHTS_NAME, variant),
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
        if model_file is None:
            raise FileNotFoundError(f"Could not find adapter weights for {pretrained_model_name_or_path}.")

        base_module = getattr(self, target_module_name)
        torch_dtype = base_module.dtype
        device = base_module.device
        dtype_orig = model_cls._set_default_torch_dtype(torch_dtype)

        overwrite_state_dict = {}
        lora_state_dict = {}
        adapter_state_dict = load_state_dict(model_file, disable_mmap=disable_mmap)
        for key, value in adapter_state_dict.items():
            value = value.to(dtype=torch_dtype, device=device)
            key = key.removeprefix(f"{target_module_name}.")
            if "lora" in key:
                lora_state_dict[key] = value
            else:
                overwrite_state_dict[key] = value

        pre_quantized = "quantization_config" in base_module.config and base_module.config["quantization_config"] is not None
        if pre_quantized:
            config["quantization_config"] = base_module.config.quantization_config
            hf_quantizer = DiffusersAutoQuantizer.from_config(config["quantization_config"], pre_quantized=True)
            hf_quantizer.validate_environment(torch_dtype=torch_dtype)
            torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)
            user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError("low_cpu_mem_usage cannot be False with quantization.")
        else:
            hf_quantizer = None

        use_keep_in_fp32_modules = model_cls._keep_in_fp32_modules is not None and (
            hf_quantizer is None or getattr(hf_quantizer, "use_keep_in_fp32_modules", False)
        )
        if use_keep_in_fp32_modules:
            keep_in_fp32_modules = model_cls._keep_in_fp32_modules
            if not isinstance(keep_in_fp32_modules, list):
                keep_in_fp32_modules = [keep_in_fp32_modules]
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError("low_cpu_mem_usage cannot be False when keep_in_fp32_modules is used.")
        else:
            keep_in_fp32_modules = []

        for key in overwrite_state_dict.keys():
            module_name = key.rsplit(".", 1)[0]
            if module_name and module_name not in keep_in_fp32_modules:
                keep_in_fp32_modules.append(module_name)

        init_contexts = [no_init_weights()]
        if low_cpu_mem_usage:
            init_contexts.append(accelerate.init_empty_weights())
        with ContextManagers(init_contexts):
            asymflow_module = model_cls.from_config(config).eval()

        torch.set_default_dtype(dtype_orig)

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(model=asymflow_module, device_map=None, keep_in_fp32_modules=keep_in_fp32_modules)

        base_state_dict = base_module.state_dict()
        base_state_dict.update(overwrite_state_dict)
        empty_state_dict = asymflow_module.state_dict()
        for param_name, param in base_state_dict.items():
            if param_name not in empty_state_dict:
                continue
            if hf_quantizer is not None and hf_quantizer.check_if_quantized_param(
                asymflow_module, param, param_name, base_state_dict, param_device=device
            ):
                hf_quantizer.create_quantized_param(
                    asymflow_module, param, param_name, device, base_state_dict, unexpected_keys=[], dtype=torch_dtype
                )
            else:
                _assign_tensor(asymflow_module, param_name, param)

        empty_device_cache()

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(asymflow_module)
            asymflow_module.hf_quantizer = hf_quantizer

        if lora_state_dict:
            if adapter_name is None:
                adapter_name = f"{target_module_name}_lakonlab"
            asymflow_module.load_lora_adapter(
                lora_state_dict,
                prefix=None,
                adapter_name=adapter_name,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
        else:
            adapter_name = None

        setattr(self, target_module_name, asymflow_module)
        return adapter_name


class LakonLabPixelFlux2KleinPipeline(PixelFlux2KleinPipeline, LakonLabMixin):
    pass
