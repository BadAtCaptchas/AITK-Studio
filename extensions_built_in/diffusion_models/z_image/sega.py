from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Iterator

import torch
import torch.nn.functional as F

def _resize_pairs(values: torch.Tensor, pair_count: int) -> torch.Tensor:
    if values.shape[1] == pair_count:
        return values
    return F.interpolate(
        values.unsqueeze(1),
        size=pair_count,
        mode="linear",
        align_corners=False,
    ).squeeze(1)


def _radial_energy(latents: torch.Tensor, bins: int) -> torch.Tensor:
    if bins <= 0:
        raise ValueError("SEGA radial energy bins must be greater than 0.")

    x = latents.detach().float()
    if x.ndim == 5:
        b, c, t, h, w = x.shape
        x = x.reshape(b, c * t, h, w)
    elif x.ndim != 4:
        raise ValueError(f"SEGA distillation expects 4D or 5D latents, got shape {tuple(latents.shape)}.")

    _, _, h, w = x.shape
    magnitude = torch.fft.rfft2(x, norm="ortho").abs().mean(dim=1)

    fy = torch.fft.fftfreq(h, device=x.device).abs()
    fx = torch.fft.rfftfreq(w, device=x.device).abs()
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    radius = radius / radius.max().clamp_min(1e-6)
    bin_index = torch.clamp((radius * bins).long(), max=bins - 1).reshape(-1)

    flat_magnitude = magnitude.reshape(magnitude.shape[0], -1)
    energy = torch.zeros((flat_magnitude.shape[0], bins), device=x.device, dtype=torch.float32)
    energy.scatter_add_(1, bin_index.unsqueeze(0).expand(flat_magnitude.shape[0], -1), flat_magnitude)

    counts = torch.bincount(bin_index, minlength=bins).to(device=x.device, dtype=torch.float32).clamp_min(1.0)
    energy = energy / counts.unsqueeze(0)
    return energy / energy.mean(dim=1, keepdim=True).clamp_min(1e-6)


def _set_axis_pair_scale(
    scale: torch.Tensor,
    offset: int,
    axis_pair_count: int,
    pair_scale: torch.Tensor,
) -> None:
    if axis_pair_count <= 0:
        return
    scale[:, offset : offset + axis_pair_count] = pair_scale[:, :axis_pair_count]


def build_zimage_sega_rope_scale(
    latents: torch.Tensor,
    axes_dims: list[int] | tuple[int, ...],
    *,
    base_resolution: float = 1024,
    strength: float = 1.0,
    min_scale: float = 0.5,
    max_scale: float = 2.0,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    if len(axes_dims) < 3:
        raise ValueError("Z-Image SEGA scaling expects three RoPE axes.")
    if base_resolution <= 0:
        raise ValueError("sega_distill_base_resolution must be greater than 0.")
    if strength < 0:
        raise ValueError("sega_distill_strength must be greater than or equal to 0.")
    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("SEGA min/max scales must be greater than 0.")
    if min_scale > max_scale:
        raise ValueError("SEGA min scale must be less than or equal to max scale.")

    axis_pairs = [int(dim) // 2 for dim in axes_dims]
    head_pair_dim = sum(axis_pairs)
    batch_size = latents.shape[0]
    scale = torch.ones((batch_size, head_pair_dim), device=latents.device, dtype=torch.float32)

    h_pairs = max(1, axis_pairs[1])
    w_pairs = max(1, axis_pairs[2])
    bins = max(h_pairs, w_pairs)

    latent_h = latents.shape[-2]
    latent_w = latents.shape[-1]
    pixel_resolution = max(latent_h, latent_w) * vae_scale_factor
    resolution_gain = max((float(pixel_resolution) / float(base_resolution)) - 1.0, 0.0)
    if strength == 0 or resolution_gain == 0:
        return scale.to(dtype=latents.dtype)

    normalized_energy = _radial_energy(latents, bins)
    pair_delta = torch.tanh(normalized_energy - 1.0)
    pair_scale = torch.clamp(
        1.0 - float(strength) * float(resolution_gain) * pair_delta,
        min=float(min_scale),
        max=float(max_scale),
    )

    offset_h = axis_pairs[0]
    offset_w = offset_h + axis_pairs[1]
    _set_axis_pair_scale(scale, offset_h, axis_pairs[1], _resize_pairs(pair_scale, h_pairs))
    _set_axis_pair_scale(scale, offset_w, axis_pairs[2], _resize_pairs(pair_scale, w_pairs))

    return scale.to(dtype=latents.dtype)


def summarize_zimage_rope_scale(scale: torch.Tensor) -> dict[str, torch.Tensor]:
    values = scale.detach().float()
    return {
        "mean": values.mean(),
        "min": values.amin(),
        "max": values.amax(),
    }


@contextmanager
def apply_zimage_sega_rope_scale(transformer, rope_scale: torch.Tensor | None) -> Iterator[None]:
    if rope_scale is None:
        yield
        return

    original_prepare_sequence = transformer._prepare_sequence

    def patched_prepare_sequence(
        self,
        feats,
        pos_ids,
        inner_pad_mask,
        pad_token,
        noise_mask=None,
        device=None,
    ):
        prepared = original_prepare_sequence(
            feats,
            pos_ids,
            inner_pad_mask,
            pad_token,
            noise_mask,
            device,
        )
        feats_out, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor = prepared

        if pad_token is self.x_pad_token:
            scale = rope_scale.to(device=freqs_cis.device, dtype=torch.float32)
            if scale.ndim == 2:
                scale = scale[:, None, :]
            if scale.shape[0] != freqs_cis.shape[0]:
                raise ValueError(
                    f"Z-Image SEGA scale batch size {scale.shape[0]} does not match RoPE batch size {freqs_cis.shape[0]}."
                )
            if scale.shape[-1] != freqs_cis.shape[-1]:
                raise ValueError(
                    f"Z-Image SEGA scale dim {scale.shape[-1]} does not match RoPE dim {freqs_cis.shape[-1]}."
                )
            freqs_cis = freqs_cis * scale

        return feats_out, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor

    transformer._prepare_sequence = MethodType(patched_prepare_sequence, transformer)
    try:
        yield
    finally:
        transformer._prepare_sequence = original_prepare_sequence
