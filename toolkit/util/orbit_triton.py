"""Optional Triton kernels for the Orbit4 W4A16 hot path.

This module is safe to import without Triton installed.  Public entry points
raise a concise error in that case; :mod:`orbit_quant` catches it and uses its
bounded Torch reference implementation.
"""

from __future__ import annotations

from typing import Optional

import torch


try:  # Triton is optional, including on CPU-only development installations.
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
    _TRITON_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on the local installation
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False
    _TRITON_IMPORT_ERROR = exc


def is_available() -> bool:
    return _TRITON_AVAILABLE


def can_use(tensor: torch.Tensor) -> bool:
    return (
        _TRITON_AVAILABLE
        and tensor.device.type == "cuda"
        and tensor.dtype in (torch.float16, torch.bfloat16)
        and tensor.ndim >= 1
        and tensor.numel() > 0
    )


def _require_triton(tensor: torch.Tensor) -> None:
    if not _TRITON_AVAILABLE:
        detail = "" if _TRITON_IMPORT_ERROR is None else f": {_TRITON_IMPORT_ERROR}"
        raise RuntimeError(f"Triton is not installed{detail}")
    if tensor.device.type != "cuda":
        raise RuntimeError("Orbit Triton kernels require CUDA tensors")
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("Orbit Triton kernels require FP16 or BF16 activations")


if _TRITON_AVAILABLE:

    @triton.jit
    def _rpbh_prepare_forward_kernel(
        x_ptr,
        permutation_ptr,
        signs_ptr,
        output_ptr,
        numel: tl.constexpr,
        dimension: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        columns = offsets % dimension
        rows = offsets // dimension
        source_columns = tl.load(permutation_ptr + columns, mask=mask, other=0)
        values = tl.load(
            x_ptr + rows * dimension + source_columns,
            mask=mask,
            other=0.0,
        )
        signs = tl.load(signs_ptr + columns, mask=mask, other=0.0)
        tl.store(output_ptr + offsets, values * signs, mask=mask)


    @triton.jit
    def _fwht_stage_kernel(
        input_ptr,
        output_ptr,
        numel: tl.constexpr,
        dimension: tl.constexpr,
        block_size: tl.constexpr,
        step: tl.constexpr,
        scale: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        columns = offsets % dimension
        position = columns % block_size
        partner_offsets = offsets + ((position ^ step) - position)
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        partners = tl.load(input_ptr + partner_offsets, mask=mask, other=0.0)
        lower = (position & step) == 0
        transformed = tl.where(lower, values + partners, partners - values)
        tl.store(output_ptr + offsets, transformed * scale, mask=mask)


    @triton.jit
    def _rpbh_finish_inverse_kernel(
        x_ptr,
        inverse_permutation_ptr,
        signs_ptr,
        output_ptr,
        numel: tl.constexpr,
        dimension: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        columns = offsets % dimension
        rows = offsets // dimension
        source_columns = tl.load(
            inverse_permutation_ptr + columns,
            mask=mask,
            other=0,
        )
        values = tl.load(
            x_ptr + rows * dimension + source_columns,
            mask=mask,
            other=0.0,
        )
        signs = tl.load(signs_ptr + source_columns, mask=mask, other=0.0)
        tl.store(output_ptr + offsets, values * signs, mask=mask)


    @triton.jit
    def _orbit4_forward_kernel(
        x_ptr,
        packed_ptr,
        row_norms_ptr,
        codebook_ptr,
        bias_ptr,
        output_ptr,
        rows,
        out_features: tl.constexpr,
        in_features: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        program_m = tl.program_id(0)
        program_n = tl.program_id(1)
        offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        packed_row_bytes: tl.constexpr = in_features // 2

        for start_k in range(0, in_features, BLOCK_K):
            offsets_k = start_k + tl.arange(0, BLOCK_K)
            x = tl.load(
                x_ptr + offsets_m[:, None] * in_features + offsets_k[None, :],
                mask=(offsets_m[:, None] < rows)
                & (offsets_k[None, :] < in_features),
                other=0.0,
            )
            packed = tl.load(
                packed_ptr
                + offsets_n[:, None] * packed_row_bytes
                + offsets_k[None, :] // 2,
                mask=(offsets_n[:, None] < out_features)
                & (offsets_k[None, :] < in_features),
                other=0,
            )
            high_nibble = (packed >> 4) & 0x0F
            low_nibble = packed & 0x0F
            codes = tl.where((offsets_k[None, :] & 1) == 0, high_nibble, low_nibble)
            weight = tl.load(codebook_ptr + codes)
            scales = tl.load(
                row_norms_ptr + offsets_n,
                mask=offsets_n < out_features,
                other=0.0,
            )
            weight = weight * scales[:, None]
            accumulator += tl.dot(x, tl.trans(weight))

        if HAS_BIAS:
            bias = tl.load(
                bias_ptr + offsets_n,
                mask=offsets_n < out_features,
                other=0.0,
            )
            accumulator += bias[None, :]
        tl.store(
            output_ptr
            + offsets_m[:, None] * out_features
            + offsets_n[None, :],
            accumulator,
            mask=(offsets_m[:, None] < rows)
            & (offsets_n[None, :] < out_features),
        )


    @triton.jit
    def _orbit4_backward_input_kernel(
        grad_output_ptr,
        packed_ptr,
        row_norms_ptr,
        codebook_ptr,
        grad_rotated_ptr,
        rows,
        out_features: tl.constexpr,
        in_features: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        program_m = tl.program_id(0)
        program_k = tl.program_id(1)
        offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_k = program_k * BLOCK_K + tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        packed_row_bytes: tl.constexpr = in_features // 2

        for start_n in range(0, out_features, BLOCK_N):
            offsets_n = start_n + tl.arange(0, BLOCK_N)
            grad_output = tl.load(
                grad_output_ptr
                + offsets_m[:, None] * out_features
                + offsets_n[None, :],
                mask=(offsets_m[:, None] < rows)
                & (offsets_n[None, :] < out_features),
                other=0.0,
            )
            packed = tl.load(
                packed_ptr
                + offsets_n[:, None] * packed_row_bytes
                + offsets_k[None, :] // 2,
                mask=(offsets_n[:, None] < out_features)
                & (offsets_k[None, :] < in_features),
                other=0,
            )
            high_nibble = (packed >> 4) & 0x0F
            low_nibble = packed & 0x0F
            codes = tl.where((offsets_k[None, :] & 1) == 0, high_nibble, low_nibble)
            weight = tl.load(codebook_ptr + codes)
            scales = tl.load(
                row_norms_ptr + offsets_n,
                mask=offsets_n < out_features,
                other=0.0,
            )
            weight = weight * scales[:, None]
            accumulator += tl.dot(grad_output, weight)

        tl.store(
            grad_rotated_ptr
            + offsets_m[:, None] * in_features
            + offsets_k[None, :],
            accumulator,
            mask=(offsets_m[:, None] < rows)
            & (offsets_k[None, :] < in_features),
        )


def _fwht_triton(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if block_size < 1 or block_size & (block_size - 1):
        raise ValueError("Hadamard block size must be a positive power of two")
    current = x.contiguous()
    scratch = torch.empty_like(current)
    numel = current.numel()
    dimension = current.shape[-1]
    grid = (triton.cdiv(numel, 256),)
    step = 1
    while step < block_size:
        scale = block_size**-0.5 if step * 2 == block_size else 1.0
        _fwht_stage_kernel[grid](
            current,
            scratch,
            numel=numel,
            dimension=dimension,
            block_size=block_size,
            step=step,
            scale=scale,
            BLOCK=256,
        )
        current, scratch = scratch, current
        step *= 2
    return current


def _launch_rpbh_forward(
    x: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    _require_triton(x)
    if x.shape[-1] != permutation.numel() or signs.numel() != permutation.numel():
        raise ValueError("Orbit rotation metadata does not match the activation")
    prepared = torch.empty_like(x)
    numel = x.numel()
    grid = (triton.cdiv(numel, 256),)
    _rpbh_prepare_forward_kernel[grid](
        x.contiguous(),
        permutation,
        signs,
        prepared,
        numel=numel,
        dimension=x.shape[-1],
        BLOCK=256,
    )
    return _fwht_triton(prepared, block_size)


def _launch_rpbh_inverse(
    y: torch.Tensor,
    inverse_permutation: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    _require_triton(y)
    transformed = _fwht_triton(y, block_size)
    output = torch.empty_like(y)
    numel = y.numel()
    grid = (triton.cdiv(numel, 256),)
    _rpbh_finish_inverse_kernel[grid](
        transformed,
        inverse_permutation,
        signs,
        output,
        numel=numel,
        dimension=y.shape[-1],
        BLOCK=256,
    )
    return output


def _launch_linear_forward(
    rotated: torch.Tensor,
    packed: torch.Tensor,
    row_norms: torch.Tensor,
    codebook: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    _require_triton(rotated)
    if rotated.shape[-1] != in_features:
        raise ValueError("Orbit input feature count does not match the packed weight")
    leading_shape = rotated.shape[:-1]
    flat = rotated.contiguous().view(-1, in_features)
    output = torch.empty(
        (flat.shape[0], out_features),
        dtype=rotated.dtype,
        device=rotated.device,
    )
    grid = (
        triton.cdiv(flat.shape[0], 32),
        triton.cdiv(out_features, 32),
    )
    bias_pointer = row_norms if bias is None else bias
    _orbit4_forward_kernel[grid](
        flat,
        packed,
        row_norms,
        codebook,
        bias_pointer,
        output,
        flat.shape[0],
        out_features=out_features,
        in_features=in_features,
        HAS_BIAS=bias is not None,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    return output.view(*leading_shape, out_features)


def _launch_linear_backward_input(
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    row_norms: torch.Tensor,
    codebook: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    _require_triton(grad_output)
    if grad_output.shape[-1] != out_features:
        raise ValueError("Orbit output feature count does not match the packed weight")
    leading_shape = grad_output.shape[:-1]
    flat = grad_output.contiguous().view(-1, out_features)
    grad_rotated = torch.empty(
        (flat.shape[0], in_features),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    grid = (
        triton.cdiv(flat.shape[0], 32),
        triton.cdiv(in_features, 32),
    )
    _orbit4_backward_input_kernel[grid](
        flat,
        packed,
        row_norms,
        codebook,
        grad_rotated,
        flat.shape[0],
        out_features=out_features,
        in_features=in_features,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    return grad_rotated.view(*leading_shape, in_features)


# torch.library makes the CUDA operations visible to torch.compile rather than
# hiding the Triton launches behind arbitrary Python calls.  Registration is
# deliberately conditional so CPU-only installations have no custom-op burden.
_library = None
if _TRITON_AVAILABLE:  # pragma: no cover - exercised only in CUDA/Triton CI
    try:
        _library = torch.library.Library("aitk_orbit", "DEF")
        _library.define(
            "rpbh_forward(Tensor x, Tensor permutation, Tensor signs, int block_size) -> Tensor"
        )
        _library.define(
            "rpbh_inverse(Tensor y, Tensor inverse_permutation, Tensor signs, int block_size) -> Tensor"
        )
        _library.define(
            "linear_forward(Tensor x, Tensor packed, Tensor row_norms, Tensor codebook, "
            "Tensor? bias, int out_features, int in_features) -> Tensor"
        )
        _library.define(
            "linear_backward_input(Tensor grad_output, Tensor packed, Tensor row_norms, "
            "Tensor codebook, int out_features, int in_features) -> Tensor"
        )
        _library.impl("rpbh_forward", _launch_rpbh_forward, "CUDA")
        _library.impl("rpbh_inverse", _launch_rpbh_inverse, "CUDA")
        _library.impl("linear_forward", _launch_linear_forward, "CUDA")
        _library.impl(
            "linear_backward_input",
            _launch_linear_backward_input,
            "CUDA",
        )
        _library.impl(
            "rpbh_forward",
            lambda x, permutation, signs, block_size: torch.empty_like(x),
            "Meta",
        )
        _library.impl(
            "rpbh_inverse",
            lambda y, inverse_permutation, signs, block_size: torch.empty_like(y),
            "Meta",
        )
        _library.impl(
            "linear_forward",
            lambda x, packed, row_norms, codebook, bias, out_features, in_features: x.new_empty(
                (*x.shape[:-1], out_features)
            ),
            "Meta",
        )
        _library.impl(
            "linear_backward_input",
            lambda grad_output, packed, row_norms, codebook, out_features, in_features: grad_output.new_empty(
                (*grad_output.shape[:-1], in_features)
            ),
            "Meta",
        )
    except RuntimeError:
        # A notebook/module reload may already own the namespace. Direct launch
        # wrappers remain valid, so re-registration is unnecessary.
        _library = None


def rpbh_forward(
    x: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if _library is not None:
        return torch.ops.aitk_orbit.rpbh_forward(x, permutation, signs, block_size)
    return _launch_rpbh_forward(x, permutation, signs, block_size)


def rpbh_inverse(
    y: torch.Tensor,
    inverse_permutation: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if _library is not None:
        return torch.ops.aitk_orbit.rpbh_inverse(
            y,
            inverse_permutation,
            signs,
            block_size,
        )
    return _launch_rpbh_inverse(y, inverse_permutation, signs, block_size)


def linear_forward(
    rotated: torch.Tensor,
    packed: torch.Tensor,
    row_norms: torch.Tensor,
    codebook: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    if _library is not None:
        return torch.ops.aitk_orbit.linear_forward(
            rotated,
            packed,
            row_norms,
            codebook,
            bias,
            out_features,
            in_features,
        )
    return _launch_linear_forward(
        rotated,
        packed,
        row_norms,
        codebook,
        bias,
        out_features,
        in_features,
    )


def linear_backward_input(
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    row_norms: torch.Tensor,
    codebook: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    if _library is not None:
        return torch.ops.aitk_orbit.linear_backward_input(
            grad_output,
            packed,
            row_norms,
            codebook,
            out_features,
            in_features,
        )
    return _launch_linear_backward_input(
        grad_output,
        packed,
        row_norms,
        codebook,
        out_features,
        in_features,
    )
