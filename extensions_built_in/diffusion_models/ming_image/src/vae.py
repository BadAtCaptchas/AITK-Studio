"""RGBA Qwen VAE with Ming's scalar latent normalization.

Layers are independent images: never encode them as a video, because that
would introduce temporal compression and causal convolution across layers.
"""

from typing import Optional

import torch
from diffusers import AutoencoderKLQwenImage
from diffusers.configuration_utils import register_to_config

from toolkit.models.v2._mixin import OstrisModelMixin


class MingImageVAE(AutoencoderKLQwenImage, OstrisModelMixin):
    aitk_subfolder = "vae"
    aitk_config_repo = "inclusionAI/Ming-Image-0.1-Design"

    @register_to_config
    def __init__(
        self,
        base_dim=96,
        z_dim=16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(False, True, True),
        dropout=0.0,
        input_channels=4,
        latents_mean=None,
        latents_std=None,
        scaling_factor=8.0064,
        shift_factor=0.0,
    ):
        if input_channels != 4:
            raise ValueError("Ming requires its four-channel RGBA VAE")
        if not isinstance(scaling_factor, (int, float)) or scaling_factor <= 0:
            raise ValueError("Ming VAE scaling_factor must be positive")
        super().__init__(
            base_dim=base_dim,
            z_dim=z_dim,
            dim_mult=list(dim_mult),
            num_res_blocks=num_res_blocks,
            attn_scales=list(attn_scales),
            temperal_downsample=list(temperal_downsample),
            dropout=dropout,
            input_channels=input_channels,
            latents_mean=latents_mean,
            latents_std=latents_std,
        )


def _as_images(images: torch.Tensor | list[torch.Tensor]) -> tuple[torch.Tensor, bool]:
    if isinstance(images, list):
        images = torch.stack(images)
    if images.ndim not in (4, 5) or images.shape[1] not in (3, 4):
        raise ValueError("Ming images must have shape [B,3|4,H,W] or [B,3|4,F,H,W]")
    if min(images.shape) < 1:
        raise ValueError("Ming images must have positive dimensions")
    if images.shape[1] == 3:
        # Input tensors use [-1,1], so opaque alpha is +1.
        images = torch.cat((images, torch.ones_like(images[:, :1])), dim=1)
    return (images.unsqueeze(2), False) if images.ndim == 4 else (images, True)


@torch.no_grad()
def encode_rgba_frames(
    vae: torch.nn.Module,
    images: torch.Tensor | list[torch.Tensor],
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    images, has_frames = _as_images(images)
    device = vae.device if device is None else device
    dtype = vae.dtype if dtype is None else dtype
    vae.eval().requires_grad_(False)
    encoded = []
    for sample in images:
        frames = []
        for frame in sample.unbind(dim=1):
            pixels = frame.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(2)
            latent = vae.encode(pixels).latent_dist.mode()
            if latent.shape[2] != 1:
                raise ValueError("Ming VAE encoded an independent image into multiple frames")
            frames.append(((latent - vae.config.shift_factor) * vae.config.scaling_factor)[0, :, 0])
        encoded.append(torch.stack(frames, dim=1))
    result = torch.stack(encoded)
    return result if has_frames else result[:, :, 0]


@torch.no_grad()
def decode_rgba_frames(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if latents.ndim not in (4, 5) or min(latents.shape) < 1:
        raise ValueError("Ming latents must have shape [B,C,H,W] or [B,C,F,H,W]")
    has_frames = latents.ndim == 5
    if not has_frames:
        latents = latents.unsqueeze(2)
    device = vae.device if device is None else device
    dtype = vae.dtype if dtype is None else dtype
    decoded = []
    for sample in latents:
        frames = []
        for frame in sample.unbind(dim=1):
            latent = frame.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(2)
            pixels = vae.decode(latent / vae.config.scaling_factor + vae.config.shift_factor).sample
            if pixels.ndim != 5 or pixels.shape[1] != 4 or pixels.shape[2] != 1:
                raise ValueError("Ming VAE must return one RGBA image for each latent frame")
            frames.append(pixels[0, :, 0])
        decoded.append(torch.stack(frames, dim=1))
    result = torch.stack(decoded)
    return result if has_frames else result[:, :, 0]
