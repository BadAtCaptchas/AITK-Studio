from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_SEGA_DISTILL_CONFIG = {
    "enabled": False,
    "base_resolution": 1024,
    "strength": 1.0,
    "min_scale": 0.5,
    "max_scale": 2.0,
}


def normalize_sega_config(config: dict | None) -> dict:
    normalized = dict(DEFAULT_SEGA_DISTILL_CONFIG)
    if config:
        normalized.update(config)
    normalized["enabled"] = bool(normalized.get("enabled", False))
    normalized["base_resolution"] = float(normalized.get("base_resolution", 1024))
    normalized["strength"] = float(normalized.get("strength", 1.0))
    normalized["min_scale"] = float(normalized.get("min_scale", 0.5))
    normalized["max_scale"] = float(normalized.get("max_scale", 2.0))
    return normalized


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


def _set_axis_scale(
    scale: torch.Tensor,
    offset: int,
    axis_dim: int,
    pair_scale: torch.Tensor,
) -> None:
    if axis_dim <= 0:
        return
    axis_values = pair_scale.repeat_interleave(2, dim=1)
    if axis_values.shape[1] < axis_dim:
        axis_values = F.pad(axis_values, (0, axis_dim - axis_values.shape[1]), value=1.0)
    scale[:, offset : offset + axis_dim] = axis_values[:, :axis_dim]


def build_flux2_sega_rope_scale(
    latents: torch.Tensor,
    token_count: int,
    axes_dim: list[int] | tuple[int, ...],
    *,
    base_resolution: float = 1024,
    strength: float = 1.0,
    min_scale: float = 0.5,
    max_scale: float = 2.0,
    vae_scale_factor: int = 16,
) -> torch.Tensor:
    if len(axes_dim) < 4:
        raise ValueError("Flux2 SEGA scaling expects four RoPE axes.")
    if token_count <= 0:
        raise ValueError("Flux2 SEGA scaling expects at least one image token.")
    if base_resolution <= 0:
        raise ValueError("sega_distill_base_resolution must be greater than 0.")
    if strength < 0:
        raise ValueError("sega_distill_strength must be greater than or equal to 0.")
    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("SEGA min/max scales must be greater than 0.")
    if min_scale > max_scale:
        raise ValueError("SEGA min scale must be less than or equal to max scale.")

    head_dim = int(sum(axes_dim))
    batch_size = latents.shape[0]
    scale = torch.ones((batch_size, head_dim), device=latents.device, dtype=torch.float32)

    spatial_h = int(axes_dim[1])
    spatial_w = int(axes_dim[2])
    h_pairs = max(1, spatial_h // 2)
    w_pairs = max(1, spatial_w // 2)
    bins = max(h_pairs, w_pairs)

    latent_h = latents.shape[-2]
    latent_w = latents.shape[-1]
    pixel_resolution = max(latent_h, latent_w) * vae_scale_factor
    resolution_gain = max((float(pixel_resolution) / float(base_resolution)) - 1.0, 0.0)
    if strength == 0 or resolution_gain == 0:
        return scale[:, None, :].expand(-1, token_count, -1).to(dtype=latents.dtype)

    normalized_energy = _radial_energy(latents, bins)
    pair_delta = torch.tanh(normalized_energy - 1.0)
    pair_scale = torch.clamp(
        1.0 - float(strength) * float(resolution_gain) * pair_delta,
        min=float(min_scale),
        max=float(max_scale),
    )

    offset_h = int(axes_dim[0])
    offset_w = offset_h + spatial_h
    _set_axis_scale(scale, offset_h, spatial_h, _resize_pairs(pair_scale, h_pairs))
    _set_axis_scale(scale, offset_w, spatial_w, _resize_pairs(pair_scale, w_pairs))

    return scale[:, None, :].expand(-1, token_count, -1).to(dtype=latents.dtype)


def summarize_rope_scale(scale: torch.Tensor) -> dict[str, torch.Tensor]:
    values = scale.detach().float()
    return {
        "mean": values.mean(),
        "min": values.amin(),
        "max": values.amax(),
    }
