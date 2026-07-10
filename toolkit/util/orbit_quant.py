"""Orbit scalar weight-only quantization backends.

Orbit stores a randomized Walsh-Hadamard rotation of each frozen linear weight.
The hot path intentionally never constructs a full floating-point weight: scalar
codes are decoded a bounded number of output rows at a time by the Torch
reference implementation, or directly inside the optional Triton matmul.
"""

from __future__ import annotations

import math
import warnings
from threading import RLock
from typing import Dict, Iterator, Literal, Tuple

import torch
import torch.nn.functional as F

from toolkit.print import print_acc
from toolkit.util.ostris_quant import OstrisQuantizer


ORBIT_QTYPES = {"orbit2": 2, "orbit3": 3, "orbit4": 4}
MIN_HADAMARD_BLOCK = 32
DEFAULT_MAX_WORKSPACE_MB = 64
_VALID_KERNELS = {"auto", "triton", "torch"}

OrbitKernel = Literal["auto", "triton", "torch"]

_normal_codebook_cache: Dict[int, torch.Tensor] = {}
_rotation_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
_skip_warned: set[int] = set()
_triton_warned: set[str] = set()
_cache_lock = RLock()


def gaussian_lloyd_max(bits: int, iters: int = 200) -> torch.Tensor:
    """Return deterministic MSE-optimal centroids for a standard normal."""
    with _cache_lock:
        cached = _normal_codebook_cache.get(bits)
        if cached is not None:
            return cached

        levels = 2**bits
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
    return (x * block_size**-0.5).view(shape)


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
    """Pack scalar codes into the legacy MSB-first Orbit bitstream.

    Four-bit Orbit is the production path.  Its codes are packed directly as
    nibbles, avoiding the old ``numel x bits`` temporary bit matrix.  The lower
    bit experimental formats retain their original byte representation.
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"Orbit supports 2, 3, or 4 bits, got {bits}")
    flat = codes.flatten().to(torch.uint8)
    if bits == 4:
        if flat.numel() % 2:
            flat = torch.cat((flat, flat.new_zeros(1)))
        return (flat[0::2] << 4) | (flat[1::2] & 0x0F)

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
    """Unpack an MSB-first Orbit bitstream."""
    if bits not in (2, 3, 4):
        raise ValueError(f"Orbit supports 2, 3, or 4 bits, got {bits}")
    if bits == 4:
        codes = torch.empty(
            packed.numel() * 2,
            dtype=torch.uint8,
            device=packed.device,
        )
        codes[0::2] = packed >> 4
        codes[1::2] = packed & 0x0F
        return codes[:numel]

    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bit_matrix = ((packed.unsqueeze(-1) >> shifts) & 1).view(-1, bits)
    weights = torch.tensor(
        [1 << index for index in range(bits - 1, -1, -1)],
        device=packed.device,
        dtype=torch.uint8,
    )
    return (bit_matrix * weights).sum(-1, dtype=torch.uint8)[:numel]


def _row_chunks(total_rows: int, rows_per_chunk: int) -> Iterator[Tuple[int, int]]:
    for start in range(0, total_rows, rows_per_chunk):
        yield start, min(start + rows_per_chunk, total_rows)


def _bounded_rows(
    total_rows: int,
    dimension: int,
    workspace_bytes: int,
    bytes_per_element: int,
) -> int:
    """Choose a row tile using a conservative peak-live-tensor estimate."""
    row_bytes = max(1, dimension * bytes_per_element)
    return max(1, min(total_rows, workspace_bytes // row_bytes))


def _autocast_activation(x: torch.Tensor) -> torch.Tensor:
    """Mirror Linear's autocast output dtype at the custom-op boundary."""
    try:
        enabled = torch.is_autocast_enabled(x.device.type)
    except TypeError:  # Older supported PyTorch stacks use the global query.
        enabled = torch.is_autocast_enabled()
    if not enabled:
        return x
    try:
        dtype = torch.get_autocast_dtype(x.device.type)
    except (AttributeError, TypeError):
        dtype = (
            torch.get_autocast_gpu_dtype()
            if x.device.type == "cuda"
            else torch.get_autocast_cpu_dtype()
        )
    if dtype not in (torch.float16, torch.bfloat16) or x.dtype == dtype:
        return x
    return x.to(dtype)


@torch.no_grad()
def _quantize_rows(
    weight: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
    codebook: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize one already-bounded row tile in float32."""
    weight_fp32 = weight.to(device=permutation.device, dtype=torch.float32)
    rotated = rpbh_forward(weight_fp32, permutation, signs, block_size)
    row_norms = rotated.norm(dim=1)
    unit = rotated / (row_norms + 1e-10).unsqueeze(1)
    edges = (codebook[:-1] + codebook[1:]) / 2
    codes = torch.bucketize(unit, edges, out_int32=True).to(torch.uint8)
    return pack_codes(codes, bits), row_norms


class OrbitQuantizer(OstrisQuantizer):
    """Randomized Hadamard scalar weight quantizer.

    ``orbit4`` is the production W4A16 format.  ``orbit2`` and ``orbit3`` stay
    readable and usable as experimental formats, but do not use the Triton path.
    """

    def __init__(
        self,
        bits: int,
        *,
        kernel: OrbitKernel = "auto",
        max_workspace_mb: int = DEFAULT_MAX_WORKSPACE_MB,
    ):
        if bits not in (2, 3, 4):
            raise ValueError(f"Orbit supports 2, 3, or 4 bits, got {bits}")
        if kernel not in _VALID_KERNELS:
            raise ValueError(
                f"Orbit kernel must be one of {sorted(_VALID_KERNELS)}, got {kernel!r}"
            )
        if isinstance(max_workspace_mb, bool) or not isinstance(max_workspace_mb, int):
            raise TypeError("Orbit max_workspace_mb must be an integer")
        if max_workspace_mb <= 0:
            raise ValueError("Orbit max_workspace_mb must be greater than zero")
        self.bits = bits
        self.kernel: OrbitKernel = kernel
        self.max_workspace_mb = max_workspace_mb

    @property
    def workspace_bytes(self) -> int:
        return self.max_workspace_mb * 1024 * 1024

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

    def _quantization_rows(self, rows: int, dimension: int) -> int:
        # Live buffers include the FP32 source tile, permutation, FWHT ping-pong,
        # normalized values, bucketize output, and packed destination tile.
        return _bounded_rows(rows, dimension, self.workspace_bytes, 32)

    def _runtime_rows(
        self,
        rows: int,
        dimension: int,
        dtype: torch.dtype,
    ) -> int:
        element_size = torch.empty((), dtype=dtype).element_size()
        # Codes/int indices and the decoded weight are simultaneously live.
        return _bounded_rows(
            rows,
            dimension,
            self.workspace_bytes,
            10 + 2 * element_size,
        )

    @staticmethod
    def _packed_bytes_per_row(dimension: int, bits: int) -> int:
        row_bits = dimension * bits
        if row_bits % 8:
            raise ValueError("Orbit row bit count must be byte aligned")
        return row_bits // 8

    @torch.no_grad()
    def _quantize_weight(
        self,
        weight: torch.Tensor,
        permutation: torch.Tensor,
        signs: torch.Tensor,
        block_size: int,
        codebook: torch.Tensor,
        row_norm_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, dimension = weight.shape
        target_device = permutation.device
        bytes_per_row = self._packed_bytes_per_row(dimension, self.bits)
        packed = torch.empty(
            rows * bytes_per_row,
            dtype=torch.uint8,
            device=target_device,
        )
        row_norms = torch.empty(rows, dtype=row_norm_dtype, device=target_device)
        rows_per_chunk = self._quantization_rows(rows, dimension)
        for start, end in _row_chunks(rows, rows_per_chunk):
            packed_chunk, norm_chunk = _quantize_rows(
                weight[start:end],
                permutation,
                signs,
                block_size,
                codebook,
                self.bits,
            )
            packed[start * bytes_per_row : end * bytes_per_row].copy_(packed_chunk)
            row_norms[start:end].copy_(norm_chunk)
        return packed, row_norms

    def quantize_(self, module: torch.nn.Linear, weight: torch.Tensor) -> None:
        dimension = module.in_features
        block_size = hadamard_block_size(dimension)
        device = weight.device
        permutation_cpu, signs_cpu = rpbh_params(dimension)
        # Cached rotation parameters are immutable templates. Every converted
        # module must own distinct registered tensors: block offload mutates
        # ``buffer.data`` in place, so aliases would couple otherwise-independent
        # CPU and CUDA residency decisions.
        permutation = permutation_cpu.to(
            device=device,
            dtype=torch.int32,
            copy=True,
        )
        inverse_permutation = torch.argsort(permutation_cpu).to(
            device=device,
            dtype=torch.int32,
            copy=True,
        )
        signs = signs_cpu.to(device=device, copy=True)
        codebook = (gaussian_lloyd_max(self.bits) * dimension**-0.5).to(
            device=device,
            copy=True,
        )
        packed, row_norms = self._quantize_weight(
            weight,
            permutation,
            signs,
            block_size,
            codebook,
            weight.dtype,
        )
        module.register_buffer("orbit_packed", packed, persistent=False)
        module.register_buffer("orbit_row_norms", row_norms, persistent=False)
        module.register_buffer("orbit_codebook", codebook, persistent=False)
        module.register_buffer("orbit_perm", permutation, persistent=False)
        module.register_buffer("orbit_inv_perm", inverse_permutation, persistent=False)
        module.register_buffer("orbit_signs", signs, persistent=False)
        module.orbit_bits = self.bits
        module.orbit_block = block_size
        module.orbit_kernel = self.kernel
        module.orbit_max_workspace_mb = self.max_workspace_mb
        module.orbit_packed_layout = (
            "nibbles_v1" if self.bits == 4 else "bitstream_msb_v1"
        )

    def _decode_rotated_rows(
        self,
        module,
        start: int,
        end: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        dimension = module.in_features
        bytes_per_row = self._packed_bytes_per_row(dimension, module.orbit_bits)
        packed = module.orbit_packed[
            start * bytes_per_row : end * bytes_per_row
        ]
        codes = unpack_codes(
            packed,
            module.orbit_bits,
            (end - start) * dimension,
        )
        weight = torch.index_select(
            module.orbit_codebook.to(dtype),
            0,
            codes.to(torch.int32),
        )
        weight = weight.view(end - start, dimension)
        return weight * module.orbit_row_norms[start:end].to(dtype).unsqueeze(1)

    def _dequantize_rotated(
        self,
        module,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compatibility helper; explicit callers request a logical-size tensor."""
        weight = torch.empty(
            (module.out_features, module.in_features),
            dtype=dtype,
            device=module.orbit_packed.device,
        )
        rows_per_chunk = self._runtime_rows(
            module.out_features,
            module.in_features,
            dtype,
        )
        for start, end in _row_chunks(module.out_features, rows_per_chunk):
            weight[start:end].copy_(
                self._decode_rotated_rows(module, start, end, dtype)
            )
        return weight

    def dequantize_to(
        self,
        module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Stream a logical compatibility weight to ``device`` by row tiles."""
        target_device = torch.device(device)
        source_device = module.orbit_packed.device
        weight = torch.empty(
            (module.out_features, module.in_features),
            dtype=dtype,
            device=target_device,
        )
        rows_per_chunk = self._runtime_rows(
            module.out_features,
            module.in_features,
            torch.float32,
        )
        for start, end in _row_chunks(module.out_features, rows_per_chunk):
            rotated = self._decode_rotated_rows(
                module,
                start,
                end,
                torch.float32,
            )
            logical_rows = rpbh_inverse(
                rotated,
                module.orbit_inv_perm,
                module.orbit_signs,
                module.orbit_block,
            )
            weight[start:end].copy_(
                logical_rows.to(device=target_device, dtype=dtype),
                non_blocking=source_device.type == "cuda" and target_device.type == "cpu",
            )
        return weight

    def dequantize(self, module) -> torch.Tensor:
        # The returned dense tensor is required by this compatibility API, but a
        # second full rotated tensor is not: decode, inverse-rotate, and copy rows.
        return self.dequantize_to(
            module,
            module.orbit_packed.device,
            torch.float32,
        )

    def requantize_(self, module, fp_weight: torch.Tensor) -> None:
        packed, row_norms = self._quantize_weight(
            fp_weight,
            module.orbit_perm,
            module.orbit_signs,
            module.orbit_block,
            module.orbit_codebook,
            module.ostris_orig_dtype,
        )
        module.orbit_packed = packed
        module.orbit_row_norms = row_norms

    def _can_use_triton(self, module, x: torch.Tensor) -> bool:
        if self.kernel == "torch" or module.orbit_bits != 4:
            return False
        try:
            from toolkit.util import orbit_triton

            return orbit_triton.can_use(x)
        except Exception:
            return False

    def _warn_triton_fallback(self, reason: str) -> None:
        key = f"{self.kernel}:{reason}"
        with _cache_lock:
            if key in _triton_warned:
                return
            _triton_warned.add(key)
        warnings.warn(
            f"Orbit Triton kernel unavailable ({reason}); using bounded Torch kernel",
            RuntimeWarning,
            stacklevel=3,
        )

    def _triton_forward(self, module, x: torch.Tensor) -> torch.Tensor:
        from toolkit.util import orbit_triton

        rotated = orbit_triton.rpbh_forward(
            x,
            module.orbit_perm,
            module.orbit_signs,
            module.orbit_block,
        )
        return orbit_triton.linear_forward(
            rotated,
            module.orbit_packed,
            module.orbit_row_norms,
            module.orbit_codebook,
            module.bias,
            module.out_features,
            module.in_features,
        )

    def _torch_forward(self, module, x: torch.Tensor) -> torch.Tensor:
        rotated = rpbh_forward(
            x,
            module.orbit_perm,
            module.orbit_signs,
            module.orbit_block,
        )
        output = x.new_empty((*x.shape[:-1], module.out_features))
        rows_per_chunk = self._runtime_rows(
            module.out_features,
            module.in_features,
            x.dtype,
        )
        for start, end in _row_chunks(module.out_features, rows_per_chunk):
            with torch.no_grad():
                weight = self._decode_rotated_rows(module, start, end, x.dtype)
            bias = (
                None
                if module.bias is None
                else module.bias[start:end].to(device=x.device, dtype=x.dtype)
            )
            output[..., start:end] = F.linear(rotated, weight, bias)
        return output

    def forward(self, module, x: torch.Tensor) -> torch.Tensor:
        compute_x = _autocast_activation(x)
        if self._can_use_triton(module, compute_x):
            try:
                return self._triton_forward(module, compute_x)
            except Exception as exc:
                self._warn_triton_fallback(type(exc).__name__)
        elif self.kernel == "triton":
            self._warn_triton_fallback("unsupported device, dtype, or installation")
        return self._torch_forward(module, compute_x)

    def _triton_backward_input(
        self,
        module,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        from toolkit.util import orbit_triton

        rotated_gradient = orbit_triton.linear_backward_input(
            grad_output,
            module.orbit_packed,
            module.orbit_row_norms,
            module.orbit_codebook,
            module.out_features,
            module.in_features,
        )
        return orbit_triton.rpbh_inverse(
            rotated_gradient,
            module.orbit_inv_perm,
            module.orbit_signs,
            module.orbit_block,
        )

    def _torch_backward_input(
        self,
        module,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        # y = R(x) W_R^T, hence grad_x = R^-1(grad_y W_R).  Decode only
        # output-row tiles of W_R and accumulate directly in the rotated basis.
        rotated_gradient = grad_output.new_zeros(
            (*grad_output.shape[:-1], module.in_features)
        )
        rows_per_chunk = self._runtime_rows(
            module.out_features,
            module.in_features,
            grad_output.dtype,
        )
        for start, end in _row_chunks(module.out_features, rows_per_chunk):
            with torch.no_grad():
                weight = self._decode_rotated_rows(
                    module,
                    start,
                    end,
                    grad_output.dtype,
                )
            rotated_gradient.add_(
                F.linear(grad_output[..., start:end], weight.transpose(0, 1))
            )
        return rpbh_inverse(
            rotated_gradient,
            module.orbit_inv_perm,
            module.orbit_signs,
            module.orbit_block,
        )

    def backward_input(
        self,
        module,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        if self._can_use_triton(module, grad_output):
            try:
                return self._triton_backward_input(module, grad_output)
            except Exception as exc:
                self._warn_triton_fallback(type(exc).__name__)
        elif self.kernel == "triton":
            self._warn_triton_fallback("unsupported device, dtype, or installation")
        return self._torch_backward_input(module, grad_output)
