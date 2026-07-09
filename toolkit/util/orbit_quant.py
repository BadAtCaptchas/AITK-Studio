"""OrbitQuant scalar weight-only quantization backends."""

from __future__ import annotations

import math
from threading import RLock
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from toolkit.print import print_acc
from toolkit.util.ostris_quant import OstrisQuantizer


ORBIT_QTYPES = {"orbit2": 2, "orbit3": 3, "orbit4": 4}
MIN_HADAMARD_BLOCK = 32

_normal_codebook_cache: Dict[int, torch.Tensor] = {}
_rotation_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
_skip_warned: set[int] = set()
_cache_lock = RLock()


def gaussian_lloyd_max(bits: int, iters: int = 200) -> torch.Tensor:
    """Return deterministic MSE-optimal centroids for a standard normal."""
    with _cache_lock:
        cached = _normal_codebook_cache.get(bits)
        if cached is not None:
            return cached

        levels = 2 ** bits
        q = (torch.arange(levels, dtype=torch.float64) + 0.5) / levels
        centroids = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)
        inf = torch.tensor([math.inf], dtype=torch.float64)
        for _ in range(iters):
            edges = (centroids[:-1] + centroids[1:]) / 2.0
            lo = torch.cat([-inf, edges])
            hi = torch.cat([edges, inf])
            phi_lo = torch.exp(-lo * lo / 2.0) / math.sqrt(2.0 * math.pi)
            phi_hi = torch.exp(-hi * hi / 2.0) / math.sqrt(2.0 * math.pi)
            cdf_lo = 0.5 * (1.0 + torch.erf(lo / math.sqrt(2.0)))
            cdf_hi = 0.5 * (1.0 + torch.erf(hi / math.sqrt(2.0)))
            centroids = (phi_lo - phi_hi) / (cdf_hi - cdf_lo)

        result = centroids.to(torch.float32)
        _normal_codebook_cache[bits] = result
        return result


def hadamard_block_size(dimension: int) -> int:
    return dimension & (-dimension)


def rpbh_params(dimension: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the deterministic CPU permutation and signs for a dimension."""
    with _cache_lock:
        cached = _rotation_cache.get(dimension)
        if cached is not None:
            return cached
        generator = torch.Generator().manual_seed(0x0EB17 + dimension)
        permutation = torch.randperm(dimension, generator=generator)
        signs = (
            torch.randint(0, 2, (dimension,), generator=generator, dtype=torch.int8)
            * 2
            - 1
        )
        result = (permutation, signs)
        _rotation_cache[dimension] = result
        return result


def _fwht(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Apply an orthonormal Walsh-Hadamard transform to contiguous blocks."""
    shape = x.shape
    x = x.reshape(-1, block_size)
    rows = x.shape[0]
    step = 1
    while step < block_size:
        pairs = x.view(rows, block_size // (2 * step), 2, step)
        x = torch.stack(
            (pairs[:, :, 0] + pairs[:, :, 1], pairs[:, :, 0] - pairs[:, :, 1]),
            dim=2,
        ).view(rows, block_size)
        step *= 2
    return (x * block_size ** -0.5).view(shape)


def rpbh_forward(
    x: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    rotated = torch.index_select(x, -1, permutation) * signs.to(x.dtype)
    return _fwht(rotated, block_size)


def rpbh_inverse(
    y: torch.Tensor,
    inverse_permutation: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    unrotated = _fwht(y, block_size) * signs.to(y.dtype)
    return torch.index_select(unrotated, -1, inverse_permutation)


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    flat = codes.flatten().to(torch.uint8)
    padding = (-flat.numel()) % 8
    if padding:
        flat = torch.cat([flat, flat.new_zeros(padding)])
    shifts = torch.arange(bits - 1, -1, -1, device=flat.device, dtype=torch.uint8)
    bit_matrix = (flat.unsqueeze(-1) >> shifts) & 1
    byte_matrix = bit_matrix.view(-1, 8)
    weights = torch.tensor(
        [1 << index for index in range(7, -1, -1)],
        device=flat.device,
        dtype=torch.uint8,
    )
    return (byte_matrix * weights).sum(-1, dtype=torch.uint8)


def unpack_codes(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bit_matrix = ((packed.unsqueeze(-1) >> shifts) & 1).view(-1, bits)
    weights = torch.tensor(
        [1 << index for index in range(bits - 1, -1, -1)],
        device=packed.device,
        dtype=torch.uint8,
    )
    return (bit_matrix * weights).sum(-1, dtype=torch.uint8)[:numel]


@torch.no_grad()
def _quantize_rows(
    weight_fp32: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
    codebook: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotated = rpbh_forward(weight_fp32, permutation, signs, block_size)
    row_norms = rotated.norm(dim=1)
    unit = rotated / (row_norms + 1e-10).unsqueeze(1)
    edges = (codebook[:-1] + codebook[1:]) / 2
    codes = torch.bucketize(unit, edges, out_int32=True).to(torch.uint8)
    return pack_codes(codes, bits), row_norms


class OrbitQuantizer(OstrisQuantizer):
    def __init__(self, bits: int):
        self.bits = bits

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        dimension = module.in_features
        block_size = hadamard_block_size(dimension)
        if block_size >= MIN_HADAMARD_BLOCK:
            return True

        should_warn = False
        with _cache_lock:
            if dimension not in _skip_warned:
                _skip_warned.add(dimension)
                should_warn = True
        if should_warn:
            print_acc(
                f"OrbitQuant: skipping linears with in_features={dimension} "
                f"(power-of-two block {block_size} is too small for the rotation)"
            )
        return False

    def quantize_(self, module: torch.nn.Linear, weight_fp32: torch.Tensor) -> None:
        dimension = module.in_features
        block_size = hadamard_block_size(dimension)
        device = weight_fp32.device
        permutation_cpu, signs_cpu = rpbh_params(dimension)
        permutation = permutation_cpu.to(device=device, dtype=torch.int32)
        inverse_permutation = torch.argsort(permutation_cpu).to(
            device=device,
            dtype=torch.int32,
        )
        signs = signs_cpu.to(device)
        codebook = (gaussian_lloyd_max(self.bits) * dimension ** -0.5).to(device)
        packed, row_norms = _quantize_rows(
            weight_fp32,
            permutation,
            signs,
            block_size,
            codebook,
            self.bits,
        )
        module.register_buffer("orbit_packed", packed, persistent=False)
        module.register_buffer(
            "orbit_row_norms",
            row_norms.to(module.weight.dtype),
            persistent=False,
        )
        module.register_buffer("orbit_codebook", codebook, persistent=False)
        module.register_buffer("orbit_perm", permutation, persistent=False)
        module.register_buffer("orbit_inv_perm", inverse_permutation, persistent=False)
        module.register_buffer("orbit_signs", signs, persistent=False)
        module.orbit_bits = self.bits
        module.orbit_block = block_size

    def _dequantize_rotated(
        self,
        module,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        numel = module.out_features * module.in_features
        codes = unpack_codes(module.orbit_packed, module.orbit_bits, numel)
        weight = torch.index_select(
            module.orbit_codebook.to(dtype),
            0,
            codes.to(torch.int32),
        )
        weight = weight.view(module.out_features, module.in_features)
        return weight * module.orbit_row_norms.to(dtype).unsqueeze(1)

    def dequantize(self, module) -> torch.Tensor:
        rotated = self._dequantize_rotated(module, torch.float32)
        return rpbh_inverse(
            rotated,
            module.orbit_inv_perm,
            module.orbit_signs,
            module.orbit_block,
        )

    def requantize_(self, module, fp_weight: torch.Tensor) -> None:
        weight = fp_weight.to(device=module.orbit_packed.device, dtype=torch.float32)
        packed, row_norms = _quantize_rows(
            weight,
            module.orbit_perm,
            module.orbit_signs,
            module.orbit_block,
            module.orbit_codebook,
            module.orbit_bits,
        )
        module.orbit_packed = packed
        module.orbit_row_norms = row_norms.to(module.ostris_orig_dtype)

    def forward(self, module, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight = self._dequantize_rotated(module, x.dtype)
        rotated = rpbh_forward(
            x,
            module.orbit_perm,
            module.orbit_signs,
            module.orbit_block,
        )
        return F.linear(rotated, weight, module.bias)
