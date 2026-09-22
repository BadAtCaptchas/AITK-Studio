# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
# Adapted from inclusionAI/Ming, modeling_bailing_moe_v2.py, revision
# 2a0c02ae3130190160c215f89fce7de3005db483. See UPSTREAM.md.
"""Frozen Bailing-MoE-v2 blocks without legacy Transformers or CUDA dependencies.

This is the image conditioner's forward-only backbone, not an autoregressive
language model. Native SDPA replaces the original eager attention; original
checkpoint names, FP32 routing, partial video RoPE, and expert math are kept.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BailingRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        hidden = hidden.float()
        hidden = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * hidden.to(dtype)


def video_rotary_embeddings(positions: torch.Tensor, config, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """PR-exact H/W interleaving then T frequencies, with an unrotated tail."""
    width = int(config.head_dim * config.partial_rotary_factor)
    if positions.ndim != 2 or positions.shape[0] != 3 or width % 2:
        raise ValueError("Bailing video positions must be [3,N] with an even rotary width")
    sections = getattr(config, "mrope_section", (8, 12, 12))
    if sum(sections) != width // 2:
        raise ValueError("Bailing mrope sections must cover half the rotary width")
    # Always construct frequencies in FP32. Casting the model must not quantize
    # these values before the trigonometric functions.
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, width, 2, device=positions.device).float() / width))
    frequencies = positions.float()[:, :, None] * inv_freq[None, None, :]
    spatial = sections[1] + sections[2]
    indices = torch.arange(width // 2, device=positions.device)
    axes = torch.where(indices < spatial, 1 + indices.remainder(2), 0)
    frequencies = frequencies[axes, :, indices].transpose(0, 1)
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return frequencies.cos().to(dtype), frequencies.sin().to(dtype)


def apply_video_rotary(query, key, cos, sin):
    def rotate(value):
        width = cos.shape[-1]
        rotated, tail = value[..., :width], value[..., width:]
        half = torch.cat((-rotated[..., width // 2:], rotated[..., :width // 2]), dim=-1)
        return torch.cat((rotated * cos[None, None] + half * sin[None, None], tail), dim=-1)
    return rotate(query), rotate(key)


class BailingAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        if self.num_heads % self.num_key_value_heads:
            raise ValueError("Bailing attention heads must be divisible by KV heads")
        self.query_key_value = nn.Linear(config.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias)
        self.q_norm = BailingRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = BailingRMSNorm(self.head_dim, config.rms_norm_eps)
        self.dense = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.use_bias)

    def forward(self, hidden, positions):
        batch, length, _ = hidden.shape
        qkv = self.query_key_value(hidden).reshape(batch, length,
            self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query, key, value = qkv.split((self.num_heads, self.num_key_value_heads, self.num_key_value_heads), dim=2)
        query = self.q_norm(query.transpose(1, 2))
        key = self.k_norm(key.transpose(1, 2))
        value = value.transpose(1, 2)
        query, key = apply_video_rotary(query, key, *positions)
        repeats = self.num_heads // self.num_key_value_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        output = F.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=0.0)
        return self.dense(output.transpose(1, 2).reshape(batch, length, -1))


class BailingMLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class BailingGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        self.expert_bias = nn.Parameter(torch.zeros(config.num_experts), requires_grad=False)
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.top_k = config.num_experts_per_tok
        self.scaling = config.routed_scaling_factor

    def forward(self, hidden):
        scores = F.linear(hidden.float(), self.weight.float()).sigmoid()
        selection = scores + self.expert_bias.float()
        grouped = selection.reshape(len(hidden), self.n_group, -1)
        group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
        groups = group_scores.topk(self.topk_group, dim=-1, sorted=False).indices
        keep = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, groups, True)
        keep = keep[:, :, None].expand_as(grouped).reshape_as(selection)
        indices = selection.masked_fill(~keep, float("-inf")).topk(self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        if self.top_k > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return indices, weights * self.scaling


class BailingMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.router_type != "MultiRouter":
            raise ValueError("Ming requires the MultiRouter Bailing checkpoint")
        self.experts = nn.ModuleList([BailingMLP(config, config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.gate = BailingGate(config)
        self.image_gate = BailingGate(config)
        # Keep checkpoint-owned audio weights, but no audio is accepted here.
        self.audio_gate = BailingGate(config)
        self.shared_experts = BailingMLP(config, config.moe_intermediate_size * config.num_shared_experts)

    def forward(self, hidden, image_mask):
        flat = hidden.reshape(-1, hidden.shape[-1])
        ids, weights = self.gate(flat)
        image_ids, image_weights = self.image_gate(flat)
        mask = image_mask.reshape(-1, 1)
        ids = torch.where(mask, image_ids, ids)
        weights = torch.where(mask, image_weights, weights)
        counts = torch.zeros(len(flat), len(self.experts), dtype=torch.int64, device=flat.device)
        counts.scatter_(1, ids, 1)
        counts = counts.sum(dim=0).cpu().tolist()
        ordering = ids.reshape(-1).argsort()
        sorted_tokens = flat[ordering // ids.shape[1]]
        results, offset = [], 0
        for expert, count in zip(self.experts, counts):
            if count:
                results.append(expert(sorted_tokens[offset:offset + count]))
                offset += count
        sorted_output = torch.cat(results)
        output = torch.empty_like(sorted_output)
        output[ordering] = sorted_output
        output = (output.reshape(*ids.shape, -1).float() * weights[:, :, None]).sum(dim=1).to(hidden.dtype)
        return output.reshape_as(hidden) + self.shared_experts(hidden)


class BailingBlock(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.attention = BailingAttention(config)
        self.input_layernorm = BailingRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = BailingRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = BailingMoE(config) if index >= config.first_k_dense_replace else BailingMLP(config, config.intermediate_size)

    def forward(self, hidden, positions, image_mask):
        hidden = hidden + self.attention(self.input_layernorm(hidden), positions)
        normalized = self.post_attention_layernorm(hidden)
        output = self.mlp(normalized, image_mask) if isinstance(self.mlp, BailingMoE) else self.mlp(normalized)
        return hidden + output
