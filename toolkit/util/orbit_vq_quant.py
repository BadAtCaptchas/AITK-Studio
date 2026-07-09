"""OrbitVQ lattice-vector weight-only quantization backends."""

from __future__ import annotations

from threading import RLock
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from toolkit.print import print_acc
from toolkit.util.orbit_quant import (
    MIN_HADAMARD_BLOCK,
    hadamard_block_size,
    rpbh_forward,
    rpbh_inverse,
    rpbh_params,
)
from toolkit.util.ostris_quant import OstrisQuantizer


ORBIT_VQ_QTYPES = {
    "orbitvq2": {
        "bits": 2,
        "vec_dim": 8,
        "lattice": "E8",
        "codebook_size": 2 ** 16,
    },
    "orbitvq3": {
        "bits": 3,
        "vec_dim": 4,
        "lattice": "D4",
        "codebook_size": 2 ** 12,
    },
    "orbitvq4": {
        "bits": 4,
        "vec_dim": 4,
        "lattice": "D4",
        "codebook_size": 2 ** 16,
    },
}

GROUP_SIZE = 128
LS_REFIT_ROUNDS = 1

BETA = {
    ("E8", 2 ** 16): 0.9800,
    ("D4", 2 ** 12): 0.4722,
    ("D4", 2 ** 16): 0.2617,
}

_KEY_BITS = 6
_KEY_OFFSET = 32

_skip_warned: set[int] = set()
_master_tables: Dict[Tuple[str, int], "_VQTables"] = {}
_device_tables: Dict[Tuple[str, int, str], "_VQTables"] = {}
_cache_lock = RLock()


def enumerate_lattice_codebook(lattice: str, size: int) -> torch.Tensor:
    """Return the closest lattice points with deterministic tie breaking."""
    if lattice == "D4":
        dimension, reach = 4, 27
        values_by_parity = [torch.arange(-(reach - 1), reach, 2, dtype=torch.int32)]
    elif lattice == "E8":
        dimension = 8
        values_by_parity = [
            torch.arange(-6, 7, 2, dtype=torch.int32),
            torch.arange(-5, 6, 2, dtype=torch.int32),
        ]
    else:
        raise ValueError(f"unknown lattice {lattice}")

    kept = []
    for values in values_by_parity:
        for first_value in values.tolist():
            rest = torch.cartesian_prod(*([values] * (dimension - 1)))
            points = torch.cat(
                [
                    torch.full(
                        (rest.shape[0], 1),
                        first_value,
                        dtype=torch.int32,
                    ),
                    rest,
                ],
                dim=1,
            )
            points = points[points.sum(dim=1).remainder(4) == 0]
            norm_squared = (points.to(torch.int64) ** 2).sum(dim=1)
            radius_squared = 48 if lattice == "E8" else 26 ** 2 + 1
            kept.append(points[norm_squared <= radius_squared])

    points = torch.cat(kept)
    norm_squared = (points.to(torch.int64) ** 2).sum(dim=1)
    key = _point_keys(points)
    order = torch.argsort(
        norm_squared * (1 << (_KEY_BITS * dimension)) + key
    )
    points = points[order[:size]]
    if points.shape[0] < size:
        raise RuntimeError(f"lattice enumeration too small for {lattice}/{size}")
    return points.to(torch.float32) / 2.0


def _point_keys(doubled_points: torch.Tensor) -> torch.Tensor:
    digits = doubled_points.to(torch.int64) + _KEY_OFFSET
    key = torch.zeros(
        doubled_points.shape[0],
        dtype=torch.int64,
        device=doubled_points.device,
    )
    for index in range(doubled_points.shape[1]):
        key = key | (digits[:, index] << (_KEY_BITS * index))
    return key


def _round_dn(x: torch.Tensor) -> torch.Tensor:
    rounded = x.round()
    odd = rounded.to(torch.int64).sum(dim=-1).remainder(2) != 0
    error = x - rounded
    index = error.abs().argmax(dim=-1, keepdim=True)
    step = torch.where(
        error.gather(-1, index) >= 0,
        1.0,
        -1.0,
    ).to(x.dtype)
    adjusted = rounded.scatter(
        -1,
        index,
        rounded.gather(-1, index) + step,
    )
    return torch.where(odd.unsqueeze(-1), adjusted, rounded)


def _round_lattice(x: torch.Tensor, lattice: str) -> torch.Tensor:
    integer_coset = _round_dn(x)
    if lattice == "D4":
        return integer_coset
    half_coset = _round_dn(x - 0.5) + 0.5
    integer_distance = (x - integer_coset).square().sum(dim=-1)
    half_distance = (x - half_coset).square().sum(dim=-1)
    return torch.where(
        (integer_distance <= half_distance).unsqueeze(-1),
        integer_coset,
        half_coset,
    )


class _VQTables:
    def __init__(self, lattice: str, size: int):
        self.lattice = lattice
        self.size = size
        self.beta = BETA[(lattice, size)]
        points = enumerate_lattice_codebook(lattice, size)
        self.codebook = points * self.beta
        keys = _point_keys((points * 2).to(torch.int32))
        self.sorted_keys, order = torch.sort(keys)
        self.key_to_index = order.to(torch.int32)
        self.half_sq_norms = self.codebook.square().sum(dim=1) / 2
        self.codebook_t = self.codebook.T.contiguous()

    def to(self, device: torch.device) -> "_VQTables":
        result = object.__new__(_VQTables)
        result.lattice = self.lattice
        result.size = self.size
        result.beta = self.beta
        result.codebook = self.codebook.to(device)
        result.sorted_keys = self.sorted_keys.to(device)
        result.key_to_index = self.key_to_index.to(device)
        result.half_sq_norms = self.half_sq_norms.to(device)
        result.codebook_t = self.codebook_t.to(device)
        return result


def get_vq_tables(lattice: str, size: int, device) -> _VQTables:
    """Return one immutable table instance per lattice, size, and device."""
    device = torch.device(device)
    master_key = (lattice, size)
    device_key = (lattice, size, str(device))
    with _cache_lock:
        master = _master_tables.get(master_key)
        if master is None:
            master = _VQTables(lattice, size)
            _master_tables[master_key] = master

        tables = _device_tables.get(device_key)
        if tables is None:
            tables = master.to(device)
            _device_tables[device_key] = tables
        return tables


def encode_vectors(z: torch.Tensor, tables: _VQTables) -> torch.Tensor:
    """Return exact nearest-codeword indices for rows of ``z``."""
    dimension = z.shape[-1]
    points = _round_lattice(z / tables.beta, tables.lattice)
    digits = (points * 2).round().to(torch.int64) + _KEY_OFFSET
    in_range = ((digits >= 0) & (digits < (1 << _KEY_BITS))).all(dim=-1)
    key = torch.zeros(z.shape[0], dtype=torch.int64, device=z.device)
    for index in range(dimension):
        key = key | (
            digits[:, index].clamp(0, (1 << _KEY_BITS) - 1)
            << (_KEY_BITS * index)
        )
    position = torch.searchsorted(tables.sorted_keys, key).clamp(
        max=tables.size - 1
    )
    hit = in_range & (tables.sorted_keys.gather(0, position) == key)
    indices = tables.key_to_index.gather(0, position.to(torch.int64))

    miss = ~hit
    miss_count = int(miss.sum())
    if miss_count:
        dtype = torch.float16 if z.device.type == "cuda" else torch.float32
        missing_vectors = z[miss].to(dtype)
        codebook_t = tables.codebook_t.to(dtype)
        half_norms = tables.half_sq_norms.to(dtype)
        found = torch.empty(miss_count, dtype=torch.int32, device=z.device)
        chunk_size = max(256, (2 ** 26) // tables.size)
        for start in range(0, miss_count, chunk_size):
            scores = (
                missing_vectors[start : start + chunk_size] @ codebook_t
                - half_norms
            )
            found[start : start + chunk_size] = scores.argmax(dim=1).to(
                torch.int32
            )
        indices[miss] = found
    return indices


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    flat = indices.flatten().to(torch.int32)
    padding = (-flat.numel()) % 8
    if padding:
        flat = torch.cat([flat, flat.new_zeros(padding)])
    shifts = torch.arange(bits - 1, -1, -1, device=flat.device, dtype=torch.int32)
    bit_matrix = ((flat.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)
    byte_matrix = bit_matrix.view(-1, 8)
    weights = torch.tensor(
        [1 << index for index in range(7, -1, -1)],
        device=flat.device,
        dtype=torch.uint8,
    )
    return (byte_matrix * weights).sum(-1, dtype=torch.uint8)


def unpack_indices(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bit_matrix = ((packed.unsqueeze(-1) >> shifts) & 1).view(-1, bits)
    indices = None
    for start in range(0, bits, 8):
        chunk_width = min(8, bits - start)
        weights = torch.tensor(
            [1 << index for index in range(chunk_width - 1, -1, -1)],
            device=packed.device,
            dtype=torch.uint8,
        )
        part = (
            bit_matrix[:, start : start + chunk_width] * weights
        ).sum(-1, dtype=torch.uint8).to(torch.int32)
        part = part << (bits - start - chunk_width)
        indices = part if indices is None else indices | part
    return indices[:numel]


class OrbitVQQuantizer(OstrisQuantizer):
    def __init__(
        self,
        bits: int,
        vec_dim: int,
        lattice: str,
        codebook_size: int,
        group_size: int = GROUP_SIZE,
    ):
        self.bits = bits
        self.vec_dim = vec_dim
        self.lattice = lattice
        self.codebook_size = codebook_size
        self.group_size = group_size
        self.index_bits = bits * vec_dim

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
                f"OrbitVQ: skipping linears with in_features={dimension} "
                f"(power-of-two block {block_size} is too small for the rotation)"
            )
        return False

    def _encode_rotated(
        self,
        rotated_weight: torch.Tensor,
        group_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tables = get_vq_tables(
            self.lattice,
            self.codebook_size,
            rotated_weight.device,
        )
        rows, dimension = rotated_weight.shape
        grouped = rotated_weight.view(
            rows,
            dimension // group_size,
            group_size,
        )
        scale = grouped.norm(dim=-1, keepdim=True) / group_size ** 0.5 + 1e-12
        indices = None
        for _ in range(LS_REFIT_ROUNDS):
            normalized = (grouped / scale).reshape(-1, self.vec_dim)
            indices = encode_vectors(normalized, tables)
            codewords = tables.codebook.index_select(0, indices).view(
                rows,
                dimension // group_size,
                group_size,
            )
            numerator = (grouped * codewords).sum(dim=-1, keepdim=True)
            denominator = codewords.square().sum(dim=-1, keepdim=True) + 1e-12
            scale = numerator / denominator
        return (
            pack_indices(indices, self.index_bits),
            scale.view(rows, dimension // group_size),
        )

    def quantize_(self, module: torch.nn.Linear, weight_fp32: torch.Tensor) -> None:
        dimension = module.in_features
        block_size = hadamard_block_size(dimension)
        group_size = min(self.group_size, block_size)
        device = weight_fp32.device
        permutation_cpu, signs_cpu = rpbh_params(dimension)
        permutation = permutation_cpu.to(device=device, dtype=torch.int32)
        inverse_permutation = torch.argsort(permutation_cpu).to(
            device=device,
            dtype=torch.int32,
        )
        signs = signs_cpu.to(device)
        rotated = rpbh_forward(weight_fp32, permutation, signs, block_size)
        packed, scales = self._encode_rotated(rotated, group_size)
        module.register_buffer("ovq_packed", packed, persistent=False)
        module.register_buffer(
            "ovq_scales",
            scales.to(module.weight.dtype),
            persistent=False,
        )
        module.register_buffer("ovq_perm", permutation, persistent=False)
        module.register_buffer("ovq_inv_perm", inverse_permutation, persistent=False)
        module.register_buffer("ovq_signs", signs, persistent=False)
        module.ovq_block = block_size
        module.ovq_group = group_size

    def _dequantize_rotated(
        self,
        module,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tables = get_vq_tables(
            self.lattice,
            self.codebook_size,
            module.ovq_packed.device,
        )
        rows, dimension = module.out_features, module.in_features
        group_size = module.ovq_group
        indices = unpack_indices(
            module.ovq_packed,
            self.index_bits,
            rows * dimension // self.vec_dim,
        )
        weight = tables.codebook.to(dtype).index_select(0, indices).view(
            rows,
            dimension // group_size,
            group_size,
        )
        weight = weight * module.ovq_scales.to(dtype).unsqueeze(-1)
        return weight.view(rows, dimension)

    def dequantize(self, module) -> torch.Tensor:
        rotated = self._dequantize_rotated(module, torch.float32)
        return rpbh_inverse(
            rotated,
            module.ovq_inv_perm,
            module.ovq_signs,
            module.ovq_block,
        )

    def requantize_(self, module, fp_weight: torch.Tensor) -> None:
        weight = fp_weight.to(device=module.ovq_packed.device, dtype=torch.float32)
        rotated = rpbh_forward(
            weight,
            module.ovq_perm,
            module.ovq_signs,
            module.ovq_block,
        )
        packed, scales = self._encode_rotated(rotated, module.ovq_group)
        module.ovq_packed = packed
        module.ovq_scales = scales.to(module.ostris_orig_dtype)

    def forward(self, module, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight = self._dequantize_rotated(module, x.dtype)
        rotated = rpbh_forward(
            x,
            module.ovq_perm,
            module.ovq_signs,
            module.ovq_block,
        )
        return F.linear(rotated, weight, module.bias)
