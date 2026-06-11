from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_1d_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.outer(pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_interpolated_pos_embed(
    embed_dim: int,
    grid_size: int,
    image_resolution: int,
    base_image_resolution: int = 256,
) -> np.ndarray:
    scale = float(base_image_resolution) / float(image_resolution)
    grid_h = np.arange(grid_size, dtype=np.float32) * scale
    grid_w = np.arange(grid_size, dtype=np.float32) * scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1).astype(np.float32)


def _default_rope_axes_dims(head_dim: int) -> tuple[int, int, int]:
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even for RoPE.")
    time_dim = head_dim // 2
    if time_dim % 2 != 0:
        time_dim -= 1
    remaining = head_dim - time_dim
    row_dim = remaining // 2
    col_dim = remaining - row_dim
    if row_dim % 2 != 0:
        row_dim -= 1
        col_dim += 1
    if col_dim % 2 != 0:
        col_dim -= 1
        row_dim += 1
    if min(time_dim, row_dim, col_dim) <= 0:
        raise ValueError("Each RoPE axis must receive at least two dimensions.")
    return time_dim, row_dim, col_dim


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        x_float = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (x_float * self.scale.float()).to(dtype)


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        mean = x_float.mean(dim=-1, keepdim=True)
        var = (x_float - mean).square().mean(dim=-1, keepdim=True)
        x_float = (x_float - mean) * torch.rsqrt(var + self.eps)
        return (x_float * self.scale.float() + self.bias.float()).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, hidden_size: int, in_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.linear1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.linear2(F.silu(self.linear1(x)))


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, hidden_features: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(hidden_size, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class MlpBlock(nn.Module):
    def __init__(self, hidden_size: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Attention(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, qk_norm: bool, use_rmsnorm: bool
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.q_norm = norm(self.head_dim) if qk_norm else None
        self.k_norm = norm(self.head_dim) if qk_norm else None
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        return self.proj(out)


class TextEncoderAdapterTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        drop_text_prob: float,
        num_heads: int,
        mlp_ratio: float,
        use_qknorm: bool,
        use_swiglu: bool,
        use_rmsnorm: bool,
        token_len: int,
    ) -> None:
        super().__init__()
        del drop_text_prob
        self.learnable_null_caption = nn.Parameter(torch.empty(1, token_len, in_channels))
        self.connector_in = nn.Linear(in_channels, hidden_size)
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.connector_norm1 = norm(hidden_size)
        self.connector_norm2 = norm(hidden_size)
        self.connector_attn = Attention(
            hidden_size, num_heads, use_qknorm, use_rmsnorm
        )
        hidden_features = (
            int(2 / 3 * int(hidden_size * mlp_ratio))
            if use_swiglu
            else int(hidden_size * mlp_ratio)
        )
        self.connector_mlp = (
            SwiGLUFFN(hidden_size, hidden_features)
            if use_swiglu
            else MlpBlock(hidden_size, hidden_features)
        )
        self.connector_norm3 = norm(hidden_size)
        self.connector_norm4 = norm(hidden_size)
        self.connector_attn2 = Attention(
            hidden_size, num_heads, use_qknorm, use_rmsnorm
        )
        self.connector_mlp2 = (
            SwiGLUFFN(hidden_size, hidden_features)
            if use_swiglu
            else MlpBlock(hidden_size, hidden_features)
        )

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        x = self.connector_in(caption)
        x = x + self.connector_attn(self.connector_norm1(x))
        x = x + self.connector_mlp(self.connector_norm2(x))
        x = x + self.connector_attn2(self.connector_norm3(x))
        return x + self.connector_mlp2(self.connector_norm4(x))


class MultimodalRopeEmbedder(nn.Module):
    def __init__(
        self,
        axes_dims: tuple[int, ...],
        axes_lens: tuple[int, ...],
        axes_scales: tuple[float, ...],
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        cos_tables = []
        sin_tables = []
        for dim, axis_len, axis_scale in zip(axes_dims, axes_lens, axes_scales):
            steps = torch.arange(0, dim, 2, dtype=torch.float32)
            base = 1.0 / (theta ** (steps / dim))
            positions = torch.arange(axis_len, dtype=torch.float32) * axis_scale
            angles = positions[:, None] * base[None, :]
            cos_tables.append(angles.cos())
            sin_tables.append(angles.sin())
        self.cos_tables = nn.ParameterList(
            [nn.Parameter(t, requires_grad=False) for t in cos_tables]
        )
        self.sin_tables = nn.ParameterList(
            [nn.Parameter(t, requires_grad=False) for t in sin_tables]
        )

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = []
        sin = []
        for axis_idx, (cos_table, sin_table) in enumerate(
            zip(self.cos_tables, self.sin_tables)
        ):
            pos = position_ids[:, :, axis_idx].clamp(0, cos_table.shape[0] - 1)
            cos.append(cos_table[pos])
            sin.append(sin_table[pos])
        return torch.cat(cos, dim=-1), torch.cat(sin, dim=-1)


def _apply_multimodal_rope(
    x: torch.Tensor,
    freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if freqs is None:
        return x
    cos, sin = freqs
    dtype = x.dtype
    x_pair = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0, x1 = x_pair.unbind(dim=-1)
    cos = cos[:, None].float()
    sin = sin[:, None].float()
    out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return out.reshape_as(x).to(dtype)


@dataclass(frozen=True)
class I1DiTForwardCache:
    text_tokens: torch.Tensor
    text_mask: Optional[torch.Tensor]
    image_freqs: tuple[torch.Tensor, torch.Tensor]
    text_freqs: tuple[torch.Tensor, torch.Tensor]


class MMDiTAttention(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, qk_norm: bool, use_rmsnorm: bool
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_image = nn.Linear(hidden_size, 3 * hidden_size)
        self.qkv_text = nn.Linear(hidden_size, 3 * hidden_size)
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.q_norm = norm(self.head_dim) if qk_norm else None
        self.k_norm = norm(self.head_dim) if qk_norm else None
        self.proj_image = nn.Linear(hidden_size, hidden_size)
        self.proj_text = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
        text_freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
        text_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, image_len, _ = image_tokens.shape
        text_len = text_tokens.shape[1]

        def project(
            linear: nn.Linear, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            qkv = linear(x).reshape(bsz, x.shape[1], 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q_image, k_image, v_image = project(self.qkv_image, image_tokens)
        q_text, k_text, v_text = project(self.qkv_text, text_tokens)
        if self.q_norm is not None:
            q_image = self.q_norm(q_image)
            k_image = self.k_norm(k_image)
            q_text = self.q_norm(q_text)
            k_text = self.k_norm(k_text)
        q_image = _apply_multimodal_rope(q_image, image_freqs)
        k_image = _apply_multimodal_rope(k_image, image_freqs)
        q_text = _apply_multimodal_rope(q_text, text_freqs)
        k_text = _apply_multimodal_rope(k_text, text_freqs)
        q = torch.cat([q_image, q_text], dim=2)
        k = torch.cat([k_image, k_text], dim=2)
        v = torch.cat([v_image, v_text], dim=2)
        key_mask = None
        attn_mask = None
        if text_mask is not None:
            image_mask = torch.ones(
                (bsz, image_len), dtype=torch.bool, device=text_tokens.device
            )
            key_mask = torch.cat([image_mask, text_mask.bool()], dim=1)
            attn_mask = key_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(
            bsz, image_len + text_len, self.hidden_size
        )
        if key_mask is not None:
            out = out * key_mask[:, :, None].to(out.dtype)
        return self.proj_image(out[:, :image_len]), self.proj_text(out[:, image_len:])


class I1DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        use_qknorm: bool,
        use_swiglu: bool,
        use_rmsnorm: bool,
        use_skip: bool = False,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip
        if use_skip:
            self.skip_linear_image = nn.Linear(2 * hidden_size, hidden_size)
            self.skip_linear_text = nn.Linear(2 * hidden_size, hidden_size)
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.norm1 = norm(hidden_size)
        self.norm2 = norm(hidden_size)
        self.norm3 = norm(hidden_size)
        self.norm4 = norm(hidden_size)
        self.attn = MMDiTAttention(hidden_size, num_heads, use_qknorm, use_rmsnorm)
        hidden_features = (
            int(2 / 3 * int(hidden_size * mlp_ratio))
            if use_swiglu
            else int(hidden_size * mlp_ratio)
        )
        self.mlp_image = (
            SwiGLUFFN(hidden_size, hidden_features)
            if use_swiglu
            else MlpBlock(hidden_size, hidden_features)
        )
        self.mlp_text = (
            SwiGLUFFN(hidden_size, hidden_features)
            if use_swiglu
            else MlpBlock(hidden_size, hidden_features)
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
        text_freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
        text_mask: Optional[torch.Tensor],
        skip: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_skip:
            if skip is None:
                raise ValueError("Skip connection is required.")
            image_tokens = self.skip_linear_image(
                torch.cat([image_tokens, skip[0]], dim=-1)
            )
            text_tokens = self.skip_linear_text(
                torch.cat([text_tokens, skip[1]], dim=-1)
            )
        image_attn, text_attn = self.attn(
            self.norm1(image_tokens),
            self.norm1(text_tokens),
            image_freqs,
            text_freqs,
            text_mask,
        )
        image_tokens = image_tokens + self.norm3(image_attn)
        text_tokens = text_tokens + self.norm3(text_attn)
        image_tokens = image_tokens + self.norm4(
            self.mlp_image(self.norm2(image_tokens))
        )
        text_tokens = text_tokens + self.norm4(self.mlp_text(self.norm2(text_tokens)))
        if text_mask is not None:
            text_tokens = text_tokens * text_mask[:, :, None].to(text_tokens.dtype)
        return image_tokens, text_tokens


class FinalLayerNoAdaLN(nn.Module):
    def __init__(
        self, hidden_size: int, patch_size: int, out_channels: int, use_rmsnorm: bool
    ) -> None:
        super().__init__()
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.norm_final = norm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class I1DiT3B(nn.Module):
    def __init__(
        self,
        input_size: int = 1024 // 8,
        image_resolution: int = 1024,
        patch_size: int = 2,
        in_channels: int = 32,
        hidden_size: int = 2016,
        depth: int = 29,
        num_heads: int = 28,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 2304,
        text_num_tokens: int = 256,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.x_embedder = PatchEmbed(patch_size, hidden_size, in_channels)
        hw = input_size // patch_size
        self.hw = hw
        pos = _get_interpolated_pos_embed(hidden_size, hw, image_resolution)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos.reshape(1, hw * hw, hidden_size))
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.text_encoder_adapter = TextEncoderAdapterTransformer(
            text_embed_dim,
            hidden_size,
            0.1,
            num_heads,
            mlp_ratio,
            True,
            True,
            True,
            text_num_tokens,
        )
        head_dim = hidden_size // num_heads
        axes_dims = _default_rope_axes_dims(head_dim)
        axes_lens = (text_num_tokens + 1, hw, hw)
        image_scale = 256.0 / image_resolution
        self.rope_embedder = MultimodalRopeEmbedder(
            axes_dims,
            axes_lens,
            (1.0, image_scale, image_scale),
            theta=rope_theta,
        )
        self.register_buffer(
            "image_row_ids", torch.repeat_interleave(torch.arange(hw), hw), persistent=False
        )
        self.register_buffer(
            "image_col_ids", torch.tile(torch.arange(hw), (hw,)), persistent=False
        )
        num_in_blocks = depth // 2
        self.in_blocks = nn.ModuleList(
            [
                I1DiTBlock(hidden_size, num_heads, mlp_ratio, True, True, True)
                for _ in range(num_in_blocks)
            ]
        )
        self.mid_block = I1DiTBlock(
            hidden_size, num_heads, mlp_ratio, True, True, True
        )
        self.out_blocks = nn.ModuleList(
            [
                I1DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    True,
                    True,
                    True,
                    use_skip=True,
                )
                for _ in range(num_in_blocks)
            ]
        )
        self.final_layer = FinalLayerNoAdaLN(
            hidden_size, patch_size, self.out_channels, True
        )

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _build_position_ids(
        self,
        text_mask: torch.Tensor,
        text_lengths: torch.Tensor,
        num_image_tokens: int,
    ) -> torch.Tensor:
        bsz, text_len = text_mask.shape
        caption_positions = torch.arange(
            text_len, dtype=torch.long, device=text_mask.device
        )[None].expand(bsz, text_len)
        caption_positions = torch.where(
            text_mask.bool(), caption_positions, torch.zeros_like(caption_positions)
        )
        zeros = torch.zeros_like(caption_positions)
        caption_ids = torch.stack((caption_positions, zeros, zeros), dim=-1)
        row_ids = self.image_row_ids[:num_image_tokens][None].expand(
            bsz, num_image_tokens
        )
        col_ids = self.image_col_ids[:num_image_tokens][None].expand(
            bsz, num_image_tokens
        )
        image_time = text_lengths[:, None].expand(bsz, num_image_tokens)
        image_ids = torch.stack((image_time, row_ids, col_ids), dim=-1)
        return torch.cat([caption_ids, image_ids], dim=1)

    def prepare_forward_cache(
        self,
        caption: torch.Tensor,
        mask: Optional[torch.Tensor],
        num_image_tokens: int,
    ) -> I1DiTForwardCache:
        text_tokens = self.text_encoder_adapter(caption)
        text_mask = mask.bool() if mask is not None else None
        seq_text = text_tokens.shape[1]
        pos_mask = (
            text_mask
            if text_mask is not None
            else torch.ones(
                (text_tokens.shape[0], seq_text),
                dtype=torch.bool,
                device=text_tokens.device,
            )
        )
        text_lengths = pos_mask.to(torch.int32).sum(dim=1)
        position_ids = self._build_position_ids(
            pos_mask, text_lengths, num_image_tokens
        )
        cos, sin = self.rope_embedder(position_ids)
        text_freqs = (cos[:, :seq_text], sin[:, :seq_text])
        image_freqs = (
            cos[:, seq_text : seq_text + num_image_tokens],
            sin[:, seq_text : seq_text + num_image_tokens],
        )
        return I1DiTForwardCache(text_tokens, text_mask, image_freqs, text_freqs)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        caption: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        forward_cache: Optional[I1DiTForwardCache] = None,
    ) -> torch.Tensor:
        del t
        tokens = self.x_embedder(x) + self.pos_embed.to(dtype=x.dtype, device=x.device)
        cache = (
            forward_cache
            if forward_cache is not None
            else self.prepare_forward_cache(caption, mask, tokens.shape[1])
        )
        text_tokens = cache.text_tokens
        text_mask = cache.text_mask
        text_freqs = cache.text_freqs
        image_freqs = cache.image_freqs
        image_tokens = tokens
        skips = []
        for block in self.in_blocks:
            image_tokens, text_tokens = block(
                image_tokens, text_tokens, image_freqs, text_freqs, text_mask
            )
            skips.append((image_tokens, text_tokens))
        image_tokens, text_tokens = self.mid_block(
            image_tokens, text_tokens, image_freqs, text_freqs, text_mask
        )
        for block in self.out_blocks:
            image_tokens, text_tokens = block(
                image_tokens,
                text_tokens,
                image_freqs,
                text_freqs,
                text_mask,
                skips.pop(),
            )
        tokens = self.final_layer(image_tokens)
        bsz = x.shape[0]
        h = w = self.input_size // self.patch_size
        p = self.patch_size
        tokens = tokens.reshape(bsz, h, w, p, p, self.out_channels)
        tokens = tokens.permute(0, 1, 3, 2, 4, 5).reshape(
            bsz, h * p, w * p, self.out_channels
        )
        return tokens.permute(0, 3, 1, 2)
