"""
ConvRot quantization backends (convrot4 / convrot8 qtypes).

convrot4 is the paper's W4A4 NVFP4 method described below. convrot8 pairs the same
rotation with per-token / per-output-channel symmetric int8 (W8A8) and
torch._int_mm: near-lossless (~1% weight error), and the fast path runs on any int8
tensor-core gpu (Ampere+), not just Blackwell. The rotation is what makes the coarse
per-row scales safe — it spreads outliers so a whole row shares one scale without
clipping damage (the classic SmoothQuant failure mode).

Implements "ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion
Transformers" (arXiv:2512.03673) as an OstrisQuantizer backend, self-contained on
top of torch (no torchao version requirements).

Method: weights and activations are rotated with a block *regular* Hadamard
transform (R4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]/2 Kronecker-powered
to rot_size, a power of 4, default 256). Unlike the standard Hadamard whose all-ones
row concentrates the block mean into one coordinate, the regular Hadamard has
constant row sums, smoothing row-wise and column-wise outliers symmetrically. The
rotation is folded into the weight offline and applied to the activation at runtime,
so it cancels in the matmul. Both sides are then quantized to NVFP4 (fp4 e2m1 values,
fp8 e4m3 scale per 16 elements, plus one fp32 per-tensor scale) and multiplied with
the Blackwell fp4 tensor cores via torch._scaled_mm — a real ~5-6x gemm speedup, ~2x
at the layer level after rotation + activation-quant overhead.

Paths:
  - inference (no grad): rotate -> fused triton nvfp4 activation quant ->
    hardware fp4 gemm. Requires sm_100+ (Blackwell); otherwise falls back to the
    dequantized matmul below.
  - training (grad enabled): rotate -> straight-through fake-quant of the
    activation (so adapters train against the same W4A4 numerics that deployment
    uses) -> bf16 matmul with the dequantized rotated weight. Fully differentiable
    w.r.t. the input.

Everything is deterministic: the rotation is a fixed matrix (no randomness at all)
and quantization is pure rounding.

Quantized state attached to each module:
  cr_qdata    packed e2m1 codes (uint8, out x in/2; low nibble = even element)
  cr_scales   e4m3 block scales (out x in/16)
  cr_scales_blocked  the same scales pre-swizzled for torch._scaled_mm
  cr_pts      fp32 per-tensor scale (scalar)
  cr_rot / module.cr_rot_size  rotation block size
"""

from typing import Optional
import warnings

import torch
import torch.nn.functional as F

from toolkit.print import print_acc
from toolkit.util.ostris_quant import OstrisQuantizer

CONVROT_QTYPES = ("convrot4", "convrot8")


def get_convrot_quantizer(
    qtype: str,
    *,
    kernel: str = "auto",
    max_workspace_mb: int = 64,
):
    if qtype == "convrot4":
        return ConvRotQuantizer(
            rot_size=256,
            kernel=kernel,
            max_workspace_mb=max_workspace_mb,
        )
    if qtype == "convrot8":
        return ConvRotInt8Quantizer(
            rot_size=256,
            kernel=kernel,
            max_workspace_mb=max_workspace_mb,
        )
    return None


F4_MAX = 6.0
F8_E4M3_MAX = 448.0
BLOCK = 16  # nvfp4 scale block

_E2M1_VALS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_EDGES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]

_hadamard_cache = {}
_vals_cache = {}
_edges_cache = {}
_skip_warned = set()
_triton_fallback_warned = set()
_runtime_fast_warned = set()


def _cached(cache, key, build):
    if key not in cache:
        cache[key] = build()
    return cache[key]


def _workspace_rows(total_rows: int, columns: int, dtype: torch.dtype, max_workspace_mb: int) -> int:
    element_bytes = max(4, torch.empty((), dtype=dtype).element_size())
    # Rotation plus quantization/dequantization can hold roughly three row tiles.
    row_bytes = max(1, columns * element_bytes * 3)
    budget = max_workspace_mb * 1024 * 1024
    return max(1, min(total_rows, budget // row_bytes))


def _row_chunks(total_rows: int, rows_per_chunk: int):
    for start in range(0, total_rows, rows_per_chunk):
        yield start, min(total_rows, start + rows_per_chunk)


def _use_triton(kernel: str) -> bool:
    if kernel == "torch":
        return False
    available = _triton_available()
    if not available and kernel == "triton" and kernel not in _triton_fallback_warned:
        _triton_fallback_warned.add(kernel)
        warnings.warn(
            "ConvRot Triton kernels were requested but are unavailable; using Torch fallbacks",
            RuntimeWarning,
            stacklevel=3,
        )
    return available


def regular_hadamard(rot_size: int, device, dtype=torch.bfloat16) -> torch.Tensor:
    """The ConvRot rotation: Kronecker powers of the 4x4 regular Hadamard matrix,
    orthonormal. Symmetric and orthogonal, so it is its own inverse."""
    key = (rot_size, str(device), dtype)

    def build():
        r4 = torch.tensor(
            [[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=torch.float64,
        )
        h = r4.clone()
        while h.shape[0] < rot_size:
            h = torch.kron(h, r4)
        if h.shape[0] != rot_size:
            raise ValueError(f"rot_size {rot_size} is not a power of 4")
        return (h / rot_size**0.5).to(device=device, dtype=dtype)

    return _cached(_hadamard_cache, key, build)


def largest_pow4_divisor(d: int) -> int:
    h = 1
    while d % (h * 4) == 0:
        h *= 4
    return h


def rotate(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Apply the block regular-Hadamard rotation along the last dim (self-inverse)."""
    if rot_size == 1:
        return x
    h = regular_hadamard(rot_size, x.device, x.dtype)
    shape = x.shape
    xb = x.reshape(-1, shape[-1] // rot_size, rot_size)
    return torch.matmul(xb, h).reshape(shape)


def to_blocked(m: torch.Tensor) -> torch.Tensor:
    """Rearrange an (R, C) scale matrix into the swizzled layout torch._scaled_mm
    expects for block-scaled fp4 (cublas 128x4-tile layout)."""
    rows, cols = m.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    if (rows, cols) != (rb * 128, cb * 4):
        padded = torch.zeros(rb * 128, cb * 4, device=m.device, dtype=m.dtype)
        padded[:rows, :cols] = m
        m = padded
    blocks = m.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def quantize_nvfp4(x: torch.Tensor, pts: Optional[torch.Tensor] = None):
    """Quantize (rows, K) to nvfp4. Returns (packed uint8 (rows, K/2),
    e4m3 scales (rows, K/16), fp32 per-tensor scale)."""
    rows, K = x.shape
    xf = x.float()
    if pts is None:
        pts = xf.abs().amax() / (F4_MAX * F8_E4M3_MAX)
        pts = torch.where(pts > 0, pts, torch.ones_like(pts))
    xb = xf.view(rows, K // BLOCK, BLOCK)
    scales = (xb.abs().amax(dim=-1) / (F4_MAX * pts)).to(torch.float8_e4m3fn)
    denom = (scales.float() * pts).unsqueeze(-1)
    z = (xb / torch.where(denom > 0, denom, torch.ones_like(denom))).clamp(
        -F4_MAX, F4_MAX
    )
    edges = _cached(
        _edges_cache, str(x.device), lambda: torch.tensor(_E2M1_EDGES, device=x.device)
    )
    mag = torch.bucketize(z.abs(), edges).to(torch.uint8)
    codes = (mag | ((z < 0).to(torch.uint8) << 3)).view(rows, K)
    packed = ((codes[:, 1::2] << 4) | codes[:, ::2]).contiguous()
    return packed, scales, pts


def dequantize_nvfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    pts: torch.Tensor,
    rows: int,
    K: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).view(rows, K)
    vals = _cached(
        _vals_cache,
        str(packed.device),
        lambda: torch.tensor(_E2M1_VALS, device=packed.device),
    )
    mag = torch.index_select(vals, 0, (codes & 7).flatten().to(torch.int32)).view(
        rows, K
    )
    v = mag * torch.where((codes & 8) > 0, -1.0, 1.0)
    v = v.view(rows, K // BLOCK, BLOCK) * (scales.float() * pts).unsqueeze(-1)
    return v.view(rows, K).to(dtype)


# ---------------- fused triton activation quant ----------------

_triton_ok = None


def _triton_available() -> bool:
    global _triton_ok
    if _triton_ok is None:
        try:
            import triton  # noqa: F401
            import triton.language as tl  # noqa: F401

            _triton_ok = True
        except Exception:
            _triton_ok = False
            print_acc(
                "ConvRot: triton is not available. The fused activation-quant kernel is "
                "disabled and activations will be quantized with plain torch ops instead "
                "— inference gets slower (most of the fp4 speedup is lost), but quality "
                "and training are unaffected."
            )
    return _triton_ok


_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    import triton
    import triton.language as tl

    @triton.jit
    def nvfp4_act_quant_kernel(
        x_ptr,
        out_ptr,
        scale_ptr,
        pts_ptr,
        K,
        n_col_tiles,
        BLOCK_K: tl.constexpr,
        BLOCKED_SCALES: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        pts = tl.load(pts_ptr)
        offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + pid_m * K + offs, mask=mask, other=0.0).to(tl.float32)
        xb = tl.reshape(x, (BLOCK_K // 16, 16))
        amax = tl.max(tl.abs(xb), axis=1)
        scale8 = (amax / (6.0 * pts)).to(tl.float8e4nv)
        denom = scale8.to(tl.float32) * pts
        denom = tl.where(denom > 0, denom, 1.0)
        # note: triton fp32 division on this backend is ~1ulp off ieee (even with
        # tl.fdiv ieee_rounding=True), so values landing exactly on a code boundary
        # can round to the adjacent code vs the torch path. ties are equidistant, so
        # this changes nothing quantitatively; activation codes are transient (never
        # stored), and the kernel itself is deterministic.
        z = xb / denom[:, None]
        z = tl.minimum(tl.maximum(z, -6.0), 6.0)
        az = tl.abs(z)
        # strict > so exact midpoints go to the lower code, matching torch.bucketize
        code = (
            (az > 0.25).to(tl.uint8)
            + (az > 0.75).to(tl.uint8)
            + (az > 1.25).to(tl.uint8)
            + (az > 1.75).to(tl.uint8)
            + (az > 2.5).to(tl.uint8)
            + (az > 3.5).to(tl.uint8)
            + (az > 5.0).to(tl.uint8)
        )
        code = code | ((z < 0).to(tl.uint8) << 3)
        lo, hi = tl.split(tl.reshape(code, (BLOCK_K // 2, 2)))
        byte = lo | (hi << 4)
        offs_b = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
        tl.store(out_ptr + pid_m * (K // 2) + offs_b, byte, mask=offs_b < K // 2)
        s_idx = pid_k * (BLOCK_K // 16) + tl.arange(0, BLOCK_K // 16)
        if BLOCKED_SCALES:
            # store straight into the cublas 128x4-tile swizzle (see to_blocked)
            r_t = pid_m // 128
            r_in = pid_m % 128
            c_t = s_idx // 4
            c = s_idx % 4
            offs_s = (
                ((r_t * n_col_tiles + c_t) * 32 + (r_in % 32)) * 16
                + (r_in // 32) * 4
                + c
            )
        else:
            offs_s = pid_m * (K // 16) + s_idx
        tl.store(scale_ptr + offs_s, scale8, mask=s_idx < K // 16)

    _kernel = nvfp4_act_quant_kernel
    return _kernel


def quantize_nvfp4_fused(
    x: torch.Tensor,
    blocked_scales: bool = False,
    *,
    use_triton: bool = True,
):
    """Triton path of quantize_nvfp4 for the inference hot loop: one read of x,
    writes packed codes + e4m3 scales (row-major, or directly in the swizzled
    layout torch._scaled_mm wants when blocked_scales=True). Falls back to the
    torch ops (row-major only)."""
    rows, K = x.shape
    if not (use_triton and _triton_available() and x.is_cuda and K % 16 == 0):
        packed, scales, pts = quantize_nvfp4(x)
        return (
            (packed, to_blocked(scales), pts)
            if blocked_scales
            else (packed, scales, pts)
        )
    pts = x.float().abs().amax() / (F4_MAX * F8_E4M3_MAX)
    pts = torch.where(pts > 0, pts, torch.ones_like(pts))
    x = x.contiguous()
    packed = torch.empty(rows, K // 2, device=x.device, dtype=torch.uint8)
    n_col_tiles = -(-(K // BLOCK) // 4)
    if blocked_scales:
        # zero-init: rows are padded to 128-tiles and the pad region must be zero
        scales = torch.zeros(
            (-(-rows // 128)) * 128 * n_col_tiles * 4,
            device=x.device,
            dtype=torch.float8_e4m3fn,
        )
    else:
        scales = torch.empty(
            rows, K // BLOCK, device=x.device, dtype=torch.float8_e4m3fn
        )
    BLOCK_K = 2048 if K >= 2048 else K
    grid = (rows, -(-K // BLOCK_K))
    _get_kernel()[grid](
        x,
        packed,
        scales,
        pts,
        K,
        n_col_tiles,
        BLOCK_K=BLOCK_K,
        BLOCKED_SCALES=blocked_scales,
        num_warps=4,
    )
    return packed, scales, pts


# ---------------- backend ----------------


_warned_no_fp4 = False


def _fp4_gemm_supported(device) -> bool:
    global _warned_no_fp4
    device = torch.device(device)
    supported = (
        device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(device)[0] >= 10  # Blackwell
        and hasattr(torch, "_scaled_mm")
        and hasattr(torch, "float4_e2m1fn_x2")
    )
    if not supported and not _warned_no_fp4:
        _warned_no_fp4 = True
        print_acc(
            f"ConvRot: no fp4 tensor-core support on this device ({device}; needs an "
            "NVIDIA Blackwell GPU, sm_100+). Inference falls back to dequantized bf16 "
            "matmuls: correct output but NO speedup, and inference activations stay "
            "unquantized (W4A16 numerics instead of W4A4). The training path is "
            "unaffected (it always simulates W4A4 via fake-quant)."
        )
    return supported


class ConvRotQuantizer(OstrisQuantizer):
    """ConvRot W4A4 backend with bounded conversion and fallback workspaces."""

    def __init__(
        self,
        rot_size: int = 256,
        *,
        kernel: str = "auto",
        max_workspace_mb: int = 64,
    ):
        if kernel not in ("auto", "triton", "torch"):
            raise ValueError(f"Unsupported ConvRot kernel {kernel!r}")
        if isinstance(max_workspace_mb, bool) or not isinstance(max_workspace_mb, int) or max_workspace_mb <= 0:
            raise ValueError("max_workspace_mb must be a positive integer")
        self.rot_size = rot_size
        self.kernel = kernel
        self.max_workspace_mb = max_workspace_mb

    def _rot_for(self, dimension: int) -> int:
        return min(self.rot_size, largest_pow4_divisor(dimension))

    def _runtime_rows(self, rows: int, columns: int, dtype: torch.dtype) -> int:
        return _workspace_rows(rows, columns, dtype, self.max_workspace_mb)

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        dimension = module.in_features
        rot = self._rot_for(dimension)
        if dimension % BLOCK != 0 or module.out_features % BLOCK != 0 or rot < 16:
            if dimension not in _skip_warned:
                _skip_warned.add(dimension)
                print_acc(
                    f"ConvRot: skipping linears with in_features={dimension} "
                    f"(needs in/out divisible by 16 and a power-of-4 block >= 16)"
                )
            return False
        return True

    def _quantize_weight(self, weight: torch.Tensor, rot: int):
        rows, columns = weight.shape
        rows_per_chunk = self._runtime_rows(rows, columns, torch.float32)
        max_value = torch.zeros((), device=weight.device, dtype=torch.float32)
        for start, end in _row_chunks(rows, rows_per_chunk):
            rotated = rotate(weight[start:end], rot)
            max_value = torch.maximum(max_value, rotated.float().abs().amax())
        pts = max_value / (F4_MAX * F8_E4M3_MAX)
        pts = torch.where(pts > 0, pts, torch.ones_like(pts))

        packed = torch.empty((rows, columns // 2), device=weight.device, dtype=torch.uint8)
        scales = torch.empty(
            (rows, columns // BLOCK),
            device=weight.device,
            dtype=torch.float8_e4m3fn,
        )
        for start, end in _row_chunks(rows, rows_per_chunk):
            rotated = rotate(weight[start:end], rot)
            packed_chunk, scales_chunk, _ = quantize_nvfp4(rotated, pts)
            packed[start:end].copy_(packed_chunk)
            scales[start:end].copy_(scales_chunk)
        return packed, scales, pts

    def quantize_(self, module: torch.nn.Linear, weight: torch.Tensor) -> None:
        rot = self._rot_for(module.in_features)
        packed, scales, pts = self._quantize_weight(weight, rot)
        module.register_buffer("cr_qdata", packed, persistent=False)
        module.register_buffer("cr_scales", scales.view(torch.uint8), persistent=False)
        module.register_buffer(
            "cr_scales_blocked", to_blocked(scales).view(torch.uint8), persistent=False
        )
        module.register_buffer(
            "cr_pts",
            pts.detach().clone().reshape(1).view(torch.uint8),
            persistent=False,
        )
        module.cr_rot_size = rot
        module.convrot_kernel = self.kernel
        module.convrot_max_workspace_mb = self.max_workspace_mb
        module.convrot_packed_layout = "nvfp4_e2m1_e4m3_v1"

    @staticmethod
    def _pts(module) -> torch.Tensor:
        return module.cr_pts.view(torch.float32).reshape(())

    def _decode_rotated_rows(
        self,
        module,
        start: int,
        end: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return dequantize_nvfp4(
            module.cr_qdata[start:end],
            module.cr_scales.view(torch.float8_e4m3fn)[start:end],
            self._pts(module),
            end - start,
            module.in_features,
            dtype,
        )

    def dequantize_to(
        self,
        module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        target_device = torch.device(device)
        weight = torch.empty(
            (module.out_features, module.in_features),
            device=target_device,
            dtype=dtype,
        )
        rows_per_chunk = self._runtime_rows(
            module.out_features, module.in_features, torch.float32
        )
        for start, end in _row_chunks(module.out_features, rows_per_chunk):
            logical = rotate(
                self._decode_rotated_rows(module, start, end, torch.float32),
                module.cr_rot_size,
            )
            weight[start:end].copy_(logical.to(device=target_device, dtype=dtype))
        return weight

    def dequantize(self, module) -> torch.Tensor:
        return self.dequantize_to(
            module,
            module.cr_qdata.device,
            torch.float32,
        )

    def requantize_(self, module, fp_weight: torch.Tensor) -> None:
        weight = fp_weight.to(device=module.cr_qdata.device)
        packed, scales, pts = self._quantize_weight(weight, module.cr_rot_size)
        module.cr_qdata = packed
        module.cr_scales = scales.view(torch.uint8)
        module.cr_scales_blocked = to_blocked(scales).view(torch.uint8)
        module.cr_pts = pts.detach().clone().reshape(1).view(torch.uint8)

    def _fake_quantize_activation(self, value: torch.Tensor) -> torch.Tensor:
        rows, columns = value.shape
        output = torch.empty_like(value)
        rows_per_chunk = self._runtime_rows(rows, columns, torch.float32)
        for start, end in _row_chunks(rows, rows_per_chunk):
            packed, scales, pts = quantize_nvfp4(value[start:end])
            output[start:end].copy_(
                dequantize_nvfp4(
                    packed,
                    scales,
                    pts,
                    end - start,
                    columns,
                    value.dtype,
                )
            )
        return output

    def _fallback_forward(self, module, x2d: torch.Tensor) -> torch.Tensor:
        output = x2d.new_empty((x2d.shape[0], module.out_features))
        rows_per_chunk = self._runtime_rows(
            module.out_features, module.in_features, x2d.dtype
        )
        for start, end in _row_chunks(module.out_features, rows_per_chunk):
            with torch.no_grad():
                weight = self._decode_rotated_rows(module, start, end, x2d.dtype)
            bias = None if module.bias is None else module.bias[start:end].to(x2d.dtype)
            output[:, start:end] = F.linear(x2d, weight, bias)
        return output

    def forward(self, module, x: torch.Tensor) -> torch.Tensor:
        in_features = module.in_features
        x_rotated = rotate(x, module.cr_rot_size)
        x2d = x_rotated.reshape(-1, in_features)
        rows = x2d.shape[0]

        if x.requires_grad:
            with torch.no_grad():
                quantized_activation = self._fake_quantize_activation(x2d.detach())
            x2d = x2d + (quantized_activation - x2d).detach()
            output = self._fallback_forward(module, x2d)
            return output.reshape(*x.shape[:-1], module.out_features)

        if _fp4_gemm_supported(x.device):
            try:
                pad = (-rows) % BLOCK
                fast_input = F.pad(x2d, (0, 0, 0, pad)) if pad else x2d
                activation, scales, pts = quantize_nvfp4_fused(
                    fast_input,
                    blocked_scales=True,
                    use_triton=_use_triton(self.kernel),
                )
                output = torch._scaled_mm(
                    activation.view(torch.float4_e2m1fn_x2),
                    module.cr_qdata.view(torch.float4_e2m1fn_x2).t(),
                    scales.view(torch.float8_e4m3fn),
                    module.cr_scales_blocked.view(torch.float8_e4m3fn),
                    out_dtype=x.dtype,
                )
                if pad:
                    output = output[:rows]
                multiplier = (pts * self._pts(module)).to(x.dtype)
                output = (
                    torch.addcmul(module.bias, output, multiplier)
                    if module.bias is not None
                    else output * multiplier
                )
                return output.reshape(*x.shape[:-1], module.out_features)
            except (AttributeError, RuntimeError, TypeError) as error:
                key = ("convrot4", type(error).__name__)
                if key not in _runtime_fast_warned:
                    _runtime_fast_warned.add(key)
                    warnings.warn(
                        f"ConvRot4 hardware path failed ({type(error).__name__}); using bounded fallback",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        output = self._fallback_forward(module, x2d)
        return output.reshape(*x.shape[:-1], module.out_features)

    def backward_input(self, module, grad_output: torch.Tensor) -> torch.Tensor:
        flat = grad_output.reshape(-1, module.out_features)
        rotated_gradient = flat.new_zeros((flat.shape[0], module.in_features))
        rows_per_chunk = self._runtime_rows(
            module.out_features, module.in_features, grad_output.dtype
        )
        for start, end in _row_chunks(module.out_features, rows_per_chunk):
            with torch.no_grad():
                weight = self._decode_rotated_rows(
                    module, start, end, grad_output.dtype
                )
            rotated_gradient.add_(
                F.linear(flat[:, start:end], weight.transpose(0, 1))
            )
        gradient = rotate(rotated_gradient, module.cr_rot_size)
        return gradient.reshape(*grad_output.shape[:-1], module.in_features)

# ---------------- convrot8: W8A8 int8 backend ----------------


def quantize_int8_rows(x: torch.Tensor):
    """Symmetric per-row int8 quantization. Returns (int8 (rows, K), fp32 scales (rows,))."""
    xf = x.float()
    scales = xf.abs().amax(dim=1) / 127.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    q = torch.round(xf / scales.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
    return q, scales


_int8_kernels = None


def _get_int8_kernels():
    global _int8_kernels
    if _int8_kernels is not None:
        return _int8_kernels
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def int8_act_quant_kernel(x_ptr, q_ptr, s_ptr, K, BLOCK_K: tl.constexpr):
        row = tl.program_id(0)
        base = row * K
        acc = tl.zeros((BLOCK_K,), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            v = tl.load(x_ptr + base + offs, mask=offs < K, other=0.0).to(tl.float32)
            acc = tl.maximum(acc, tl.abs(v))
        amax = tl.max(acc, axis=0)
        scale = tl.where(amax > 0, amax / 127.0, 1.0)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            # rint = round-half-to-even, matching torch.round in the reference path
            q = libdevice.rint(v / scale)
            q = tl.minimum(tl.maximum(q, -127.0), 127.0)
            tl.store(q_ptr + base + offs, q.to(tl.int8), mask=mask)
        tl.store(s_ptr + row, scale)

    @triton.jit
    def int8_epilogue_kernel(
        i_ptr,
        as_ptr,
        ws_ptr,
        b_ptr,
        o_ptr,
        N,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cb = tl.program_id(1)
        offs = cb * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        acc = tl.load(i_ptr + row * N + offs, mask=mask, other=0).to(tl.float32)
        a_s = tl.load(as_ptr + row)
        w_s = tl.load(ws_ptr + offs, mask=mask, other=0.0)
        out = acc * (a_s * w_s)
        if HAS_BIAS:
            out += tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(o_ptr + row * N + offs, out.to(o_ptr.dtype.element_ty), mask=mask)

    _int8_kernels = (int8_act_quant_kernel, int8_epilogue_kernel)
    return _int8_kernels


def quantize_int8_rows_fused(x: torch.Tensor, *, use_triton: bool = True):
    """Triton path of quantize_int8_rows: one extra read of x instead of the
    multi-kernel torch chain. Falls back to the torch ops."""
    rows, K = x.shape
    if not (use_triton and _triton_available() and x.is_cuda):
        return quantize_int8_rows(x)
    x = x.contiguous()
    q = torch.empty(rows, K, device=x.device, dtype=torch.int8)
    scales = torch.empty(rows, device=x.device, dtype=torch.float32)
    kernel, _ = _get_int8_kernels()
    kernel[(rows,)](x, q, scales, K, BLOCK_K=2048 if K >= 2048 else K, num_warps=8)
    return q, scales


def _int8_epilogue(
    i32: torch.Tensor,
    a_scales: torch.Tensor,
    w_scales: torch.Tensor,
    bias,
    out_dtype: torch.dtype,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    """out = i32 * a_scales[:, None] * w_scales[None, :] (+ bias), in out_dtype."""
    m, n = i32.shape
    if use_triton and _triton_available() and i32.is_cuda:
        out = torch.empty(m, n, device=i32.device, dtype=out_dtype)
        _, kernel = _get_int8_kernels()
        grid = (m, -(-n // 1024))
        kernel[grid](
            i32,
            a_scales,
            w_scales,
            bias if bias is not None else i32,
            out,
            n,
            HAS_BIAS=bias is not None,
            BLOCK_N=1024,
            num_warps=4,
        )
        return out
    out = i32.float() * w_scales
    out = out * a_scales.unsqueeze(1)
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


_int8_mm_ok = None


def _int8_gemm_supported(device) -> bool:
    global _int8_mm_ok
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        supported = False
    else:
        if _int8_mm_ok is None:
            try:
                a = torch.zeros(32, 64, dtype=torch.int8, device=device)
                b = torch.zeros(64, 32, dtype=torch.int8, device=device)
                torch._int_mm(a, b)
                _int8_mm_ok = True
            except Exception:
                _int8_mm_ok = False
        supported = _int8_mm_ok
    global _warned_no_int8
    if not supported and not _warned_no_int8:
        _warned_no_int8 = True
        print_acc(
            f"ConvRot: int8 matmul (torch._int_mm) is not usable on this device "
            f"({device}). Inference falls back to dequantized bf16 matmuls: correct "
            "output but NO speedup, and inference activations stay unquantized "
            "(W8A16 numerics instead of W8A8). The training path is unaffected "
            "(it always simulates W8A8 via fake-quant)."
        )
    return supported


_warned_no_int8 = False


class ConvRotInt8Quantizer(OstrisQuantizer):
    """ConvRot W8A8 backend with bounded conversion and fallback workspaces."""

    def __init__(
        self,
        rot_size: int = 256,
        *,
        kernel: str = "auto",
        max_workspace_mb: int = 64,
    ):
        if kernel not in ("auto", "triton", "torch"):
            raise ValueError(f"Unsupported ConvRot kernel {kernel!r}")
        if isinstance(max_workspace_mb, bool) or not isinstance(max_workspace_mb, int) or max_workspace_mb <= 0:
            raise ValueError("max_workspace_mb must be a positive integer")
        self.rot_size = rot_size
        self.kernel = kernel
        self.max_workspace_mb = max_workspace_mb

    def _rot_for(self, dimension: int) -> int:
        return min(self.rot_size, largest_pow4_divisor(dimension))

    def _runtime_rows(self, rows: int, columns: int, dtype: torch.dtype) -> int:
        return _workspace_rows(rows, columns, dtype, self.max_workspace_mb)

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        dimension = module.in_features
        if dimension % BLOCK != 0 or module.out_features % 8 != 0 or self._rot_for(dimension) < 16:
            if dimension not in _skip_warned:
                _skip_warned.add(dimension)
                print_acc(
                    f"ConvRot: skipping linears with in_features={dimension} "
                    f"(needs in divisible by 16, out by 8, and a power-of-4 block >= 16)"
                )
            return False
        return True

    def _quantize_weight(self, weight: torch.Tensor, rot: int):
        rows, columns = weight.shape
        quantized = torch.empty((rows, columns), device=weight.device, dtype=torch.int8)
        scales = torch.empty(rows, device=weight.device, dtype=torch.float32)
        rows_per_chunk = self._runtime_rows(rows, columns, torch.float32)
        for start, end in _row_chunks(rows, rows_per_chunk):
            chunk, chunk_scales = quantize_int8_rows(rotate(weight[start:end], rot))
            quantized[start:end].copy_(chunk)
            scales[start:end].copy_(chunk_scales)
        return quantized, scales

    def quantize_(self, module: torch.nn.Linear, weight: torch.Tensor) -> None:
        rot = self._rot_for(module.in_features)
        quantized, scales = self._quantize_weight(weight, rot)
        module.register_buffer("cr8_qdata", quantized, persistent=False)
        module.register_buffer("cr8_scales", scales.view(torch.uint8), persistent=False)
        module.cr8_rot_size = rot
        module.convrot_kernel = self.kernel
        module.convrot_max_workspace_mb = self.max_workspace_mb
        module.convrot_packed_layout = "int8_per_row_v1"

    @staticmethod
    def _scales(module) -> torch.Tensor:
        return module.cr8_scales.view(torch.float32)

    def _decode_rotated_rows(
        self,
        module,
        start: int,
        end: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        weight = (
            module.cr8_qdata[start:end].float()
            * self._scales(module)[start:end].unsqueeze(1)
        )
        return weight.to(dtype)

    def dequantize_to(
        self,
        module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        target_device = torch.device(device)
        weight = torch.empty(
            (module.out_features, module.in_features),
            device=target_device,
            dtype=dtype,
        )
        rows_per_chunk = self._runtime_rows(
            module.out_features, module.in_features, torch.float32
        )
        for start, end in _row_chunks(module.out_features, rows_per_chunk):
            logical = rotate(
                self._decode_rotated_rows(module, start, end, torch.float32),
                module.cr8_rot_size,
            )
            weight[start:end].copy_(logical.to(device=target_device, dtype=dtype))
        return weight

    def dequantize(self, module) -> torch.Tensor:
        return self.dequantize_to(
            module,
            module.cr8_qdata.device,
            torch.float32,
        )

    def requantize_(self, module, fp_weight: torch.Tensor) -> None:
        weight = fp_weight.to(device=module.cr8_qdata.device)
        quantized, scales = self._quantize_weight(weight, module.cr8_rot_size)
        module.cr8_qdata = quantized
        module.cr8_scales = scales.view(torch.uint8)

    def _fake_quantize_activation(self, value: torch.Tensor) -> torch.Tensor:
        rows, columns = value.shape
        output = torch.empty_like(value)
        rows_per_chunk = self._runtime_rows(rows, columns, torch.float32)
        for start, end in _row_chunks(rows, rows_per_chunk):
            quantized, scales = quantize_int8_rows(value[start:end])
            output[start:end].copy_(
                (quantized.float() * scales.unsqueeze(1)).to(value.dtype)
            )
        return output

    def _fallback_forward(self, module, x2d: torch.Tensor) -> torch.Tensor:
        output = x2d.new_empty((x2d.shape[0], module.out_features))
        rows_per_chunk = self._runtime_rows(
            module.out_features, module.in_features, x2d.dtype
        )
        for start, end in _row_chunks(module.out_features, rows_per_chunk):
            with torch.no_grad():
                weight = self._decode_rotated_rows(module, start, end, x2d.dtype)
            bias = None if module.bias is None else module.bias[start:end].to(x2d.dtype)
            output[:, start:end] = F.linear(x2d, weight, bias)
        return output

    def forward(self, module, x: torch.Tensor) -> torch.Tensor:
        x_rotated = rotate(x, module.cr8_rot_size)
        x2d = x_rotated.reshape(-1, module.in_features)
        rows = x2d.shape[0]

        if x.requires_grad:
            with torch.no_grad():
                quantized_activation = self._fake_quantize_activation(x2d.detach())
            x2d = x2d + (quantized_activation - x2d).detach()
            output = self._fallback_forward(module, x2d)
            return output.reshape(*x.shape[:-1], module.out_features)

        if _int8_gemm_supported(x.device):
            try:
                pad = (-rows) % 32
                fast_input = F.pad(x2d, (0, 0, 0, pad)) if pad else x2d
                activation, activation_scales = quantize_int8_rows_fused(
                    fast_input,
                    use_triton=_use_triton(self.kernel),
                )
                integer_output = torch._int_mm(activation, module.cr8_qdata.t())
                output = _int8_epilogue(
                    integer_output,
                    activation_scales,
                    self._scales(module),
                    module.bias,
                    x.dtype,
                    use_triton=_use_triton(self.kernel),
                )
                if pad:
                    output = output[:rows]
                return output.reshape(*x.shape[:-1], module.out_features)
            except (AttributeError, RuntimeError, TypeError) as error:
                key = ("convrot8", type(error).__name__)
                if key not in _runtime_fast_warned:
                    _runtime_fast_warned.add(key)
                    warnings.warn(
                        f"ConvRot8 hardware path failed ({type(error).__name__}); using bounded fallback",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        output = self._fallback_forward(module, x2d)
        return output.reshape(*x.shape[:-1], module.out_features)

    def backward_input(self, module, grad_output: torch.Tensor) -> torch.Tensor:
        flat = grad_output.reshape(-1, module.out_features)
        rotated_gradient = flat.new_zeros((flat.shape[0], module.in_features))
        rows_per_chunk = self._runtime_rows(
            module.out_features, module.in_features, grad_output.dtype
        )
        for start, end in _row_chunks(module.out_features, rows_per_chunk):
            with torch.no_grad():
                weight = self._decode_rotated_rows(
                    module, start, end, grad_output.dtype
                )
            rotated_gradient.add_(
                F.linear(flat[:, start:end], weight.transpose(0, 1))
            )
        gradient = rotate(rotated_gradient, module.cr8_rot_size)
        return gradient.reshape(*grad_output.shape[:-1], module.in_features)
