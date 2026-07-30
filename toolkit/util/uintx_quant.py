"""
Bit-exact reimplementation of torchao 0.10.0's UIntXWeightOnlyConfig weight
quantization as an OstrisQuantizer backend, so the uint2..uint7 qtypes keep
producing byte-identical weights after torchao drops uintx support. Existing
accuracy recovery adapters were trained against these exact quantized bases,
so every op below mirrors torchao's sequence (same order, same dtypes):

  choose_qparams_affine (ASYMMETRIC, block_size (1, 64), preserve_zero=True,
  eps=float32 eps, INT zero-point domain, scale in the weight dtype):
    min/max per 64-wide group along in_features, extended to include 0
    scale      = (max_val_pos - min_val_neg) / (qmax - qmin), clamped to eps
    zero_point = clamp(qmin - round(min_val_neg / scale), qmin, qmax) as int
  quantize_affine:
    q = clamp(round(w * (1.0 / scale)) + zero_point, qmin, qmax)
  dequantize_affine:
    w = ((q as int) - zero_point) cast to the weight dtype, times scale

The arithmetic is done in the original weight dtype (usually bfloat16) because
that is what torchao did; doing it in float32 would round differently.

uint8 resolves to a backend too, but can_quantize always refuses it: torchao
0.10.0's uint8 path raised inside UintxTensor.from_uint8 (packing only supports
1..7 bits) before the module was touched, so "uint8" layers have always been
silently left unquantized. Refusing keeps that exact behavior — flip the check
in can_quantize if real uint8 quantization is ever wanted for new models.

Buffers on the module (registered by quantize_):
  uintx_packed      quantized codes packed into power-of-2 bit shards (like
                    torchao's UintxTensor: e.g. uint3 = a 2-bit + a 1-bit
                    shard), concatenated into one flat uint8 buffer. Shards
                    unpack with a couple of elementwise shift/mask kernels,
                    which is much cheaper per forward than a generic bitstream.
  uintx_scale       per-group scale, stored as a uint8 byte view of the weight
                    dtype so module.to(dtype=...) can't cast it
  uintx_zero_point  per-group zero point, uint8 (values are in [0, qmax])
"""

import torch

from toolkit.util.ostris_quant import OstrisLinear, OstrisQuantizer

UINTX_QTYPES = {f"uint{bits}": bits for bits in range(2, 9)}

_EPS = torch.finfo(torch.float32).eps


def _pack_shard(vals: torch.Tensor, k: int) -> torch.Tensor:
    """Pack flat uint8 values (< 2**k) into bytes, 8 // k values per byte.
    Values are laid out in 8//k contiguous chunks (chunk j holds bits
    [j*k, j*k+k) of every byte) so pack and unpack touch memory coalesced."""
    vpb = 8 // k
    if vpb == 1:
        return vals.clone()
    pad = (-vals.numel()) % vpb
    if pad:
        vals = torch.cat([vals, vals.new_zeros(pad)])
    chunks = vals.view(vpb, -1)
    out = chunks[0].clone()
    for j in range(1, vpb):
        out |= chunks[j] << (j * k)
    return out


def _unpack_shard(packed: torch.Tensor, k: int, numel: int) -> torch.Tensor:
    vpb = 8 // k
    if vpb == 1:
        return packed[:numel]
    out = torch.empty(vpb, packed.numel(), dtype=torch.uint8, device=packed.device)
    for j in range(vpb):
        torch.bitwise_right_shift(packed, j * k, out=out[j])
    out.bitwise_and_((1 << k) - 1)
    return out.view(-1)[:numel]


def pack_uintx(codes: torch.Tensor, nbits: int) -> torch.Tensor:
    """Pack integer codes (values < 2**nbits) into concatenated power-of-2 bit
    shards; bits [offset, offset+k) of each code land in the k-bit shard."""
    flat = codes.flatten().to(torch.uint8)
    shards = []
    offset = 0
    for k in (8, 4, 2, 1):
        if nbits & k:
            if k == nbits:  # single shard, values are already < 2**k
                shards.append(_pack_shard(flat, k))
            else:
                shards.append(_pack_shard((flat >> offset) & ((1 << k) - 1), k))
            offset += k
    return torch.cat(shards) if len(shards) > 1 else shards[0]


def unpack_uintx(packed: torch.Tensor, nbits: int, numel: int) -> torch.Tensor:
    """Inverse of pack_uintx. Returns a flat uint8 tensor of length numel."""
    return unpack_uintx_range(packed, nbits, numel, 0, numel)


def unpack_uintx_range(
    packed: torch.Tensor,
    nbits: int,
    numel: int,
    start: int,
    end: int,
) -> torch.Tensor:
    """Unpack a flat code range without materializing every logical code."""
    if not 0 <= start <= end <= numel:
        raise ValueError(f"invalid UIntX code range [{start}, {end}) for {numel} values")
    indices = torch.arange(start, end, dtype=torch.int64, device=packed.device)
    out = torch.zeros(end - start, dtype=torch.uint8, device=packed.device)
    offset = 0
    pos = 0
    for k in (8, 4, 2, 1):
        if nbits & k:
            vpb = 8 // k
            nbytes = -(-numel // vpb)
            byte_indices = torch.remainder(indices, nbytes)
            slots = torch.div(indices, nbytes, rounding_mode="floor")
            vals = torch.bitwise_right_shift(
                packed[pos + byte_indices],
                slots * k,
            )
            vals.bitwise_and_((1 << k) - 1)
            out |= vals.bitwise_left_shift(offset)
            offset += k
            pos += nbytes
    return out


class UIntXQuantizer(OstrisQuantizer):
    # quantization runs in the weight's own dtype (that is what torchao did);
    # skipping the float32 copy keeps peak memory at torchao levels
    wants_fp32_weight = False

    def __init__(
        self,
        nbits: int,
        group_size: int = 64,
        max_workspace_mb: int = 64,
    ):
        if nbits not in range(2, 9):
            raise ValueError(f"UIntX nbits must be between 2 and 8, got {nbits}")
        if group_size <= 0:
            raise ValueError("UIntX group_size must be greater than zero")
        if max_workspace_mb <= 0:
            raise ValueError("UIntX max_workspace_mb must be greater than zero")
        self.nbits = nbits
        self.group_size = group_size
        self.max_workspace_mb = max_workspace_mb
        self.qmin = 0
        self.qmax = (1 << nbits) - 1

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        if self.nbits == 8:
            # see module docstring: uint8 has always been a silent no-op
            return False
        weight = getattr(module, "weight", None)
        if weight is None or weight.dim() != 2:
            return False
        # torchao asserted on non-divisible in_features, which the quantize loop
        # caught, leaving the layer unquantized; refuse for the same end state
        return weight.shape[1] % self.group_size == 0

    @torch.no_grad()
    def quantize_(self, module: torch.nn.Linear, weight: torch.Tensor) -> None:
        # weight arrives in its original dtype (wants_fp32_weight is False)
        self._quantize_impl(module, weight)

    @torch.no_grad()
    def requantize_(self, module: "OstrisLinear", fp_weight: torch.Tensor) -> None:
        # the torchao path cast the merged weight to the original dtype and
        # re-quantized in that dtype
        self._quantize_impl(module, fp_weight.to(module.ostris_orig_dtype))

    def _quantize_impl(self, module: torch.nn.Module, w: torch.Tensor) -> None:
        out_f, in_f = w.shape
        groups = in_f // self.group_size
        wv = w.contiguous().view(out_f, groups, self.group_size)

        min_val = torch.amin(wv, dim=2)
        max_val = torch.amax(wv, dim=2)
        # preserve_zero: the qparams must be able to represent 0.0 exactly
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = (max_val_pos - min_val_neg) / float(self.qmax - self.qmin)
        scale = torch.clamp(scale, min=_EPS)
        zero_point = self.qmin - torch.round(min_val_neg / scale)
        zero_point = torch.clamp(zero_point, self.qmin, self.qmax).to(torch.int32)

        q = torch.clamp(
            torch.round(wv * (1.0 / scale.view(out_f, groups, 1)))
            + zero_point.view(out_f, groups, 1),
            self.qmin,
            self.qmax,
        )
        q = q.view(out_f, in_f).to(torch.uint8)

        module.register_buffer(
            "uintx_packed", pack_uintx(q, self.nbits), persistent=False
        )
        module.register_buffer(
            "uintx_scale", scale.contiguous().view(torch.uint8), persistent=False
        )
        module.register_buffer(
            "uintx_zero_point", zero_point.to(torch.uint8), persistent=False
        )
        module.uintx_bits = self.nbits
        module.uintx_group_size = self.group_size
        module.uintx_max_workspace_mb = self.max_workspace_mb
        module.uintx_packed_layout = "power2_shards_v1"

    def _rows_per_chunk(self, module: "OstrisLinear") -> int:
        element_size = torch.empty((), dtype=module.ostris_orig_dtype).element_size()
        bytes_per_row = max(1, module.in_features * (element_size + 1))
        workspace_bytes = self.max_workspace_mb * 1024 * 1024
        return max(1, min(module.out_features, workspace_bytes // bytes_per_row))

    def _dequantize_rows(
        self,
        module: "OstrisLinear",
        start_row: int,
        end_row: int,
        device: torch.device,
    ) -> torch.Tensor:
        out_f, in_f = module.out_features, module.in_features
        groups = in_f // self.group_size
        packed = module.uintx_packed.to(device=device)
        scale = module.uintx_scale.view(module.ostris_orig_dtype).to(device=device)
        zero_point = module.uintx_zero_point.to(device=device)
        q = unpack_uintx_range(
            packed,
            self.nbits,
            out_f * in_f,
            start_row * in_f,
            end_row * in_f,
        )
        dq = q.view(end_row - start_row, groups, self.group_size).to(scale.dtype)
        dq -= zero_point[start_row:end_row].to(scale.dtype).view(
            end_row - start_row,
            groups,
            1,
        )
        dq *= scale[start_row:end_row].view(end_row - start_row, groups, 1)
        return dq.view(end_row - start_row, in_f)

    def dequantize_to(
        self,
        module: "OstrisLinear",
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reconstruct in bounded row chunks, preserving native-dtype rounding."""
        device = torch.device(device)
        result = torch.empty(
            module.out_features,
            module.in_features,
            device=device,
            dtype=dtype,
        )
        rows_per_chunk = self._rows_per_chunk(module)
        for start in range(0, module.out_features, rows_per_chunk):
            end = min(module.out_features, start + rows_per_chunk)
            result[start:end].copy_(
                self._dequantize_rows(module, start, end, device).to(dtype=dtype)
            )
        return result

    def dequantize(self, module: "OstrisLinear") -> torch.Tensor:
        return self.dequantize_to(
            module,
            module.quantized_device,
            torch.float32,
        )

    def forward(self, module: "OstrisLinear", x: torch.Tensor) -> torch.Tensor:
        outputs = []
        rows_per_chunk = self._rows_per_chunk(module)
        with torch.no_grad():
            for start in range(0, module.out_features, rows_per_chunk):
                end = min(module.out_features, start + rows_per_chunk)
                weight = self._dequantize_rows(module, start, end, x.device)
                if weight.dtype != x.dtype:
                    weight = weight.to(x.dtype)
                bias = module.bias[start:end] if module.bias is not None else None
                outputs.append(torch.nn.functional.linear(x, weight, bias))
        return torch.cat(outputs, dim=-1)

    def backward_input(
        self,
        module: "OstrisLinear",
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        grad_input = torch.zeros(
            *grad_output.shape[:-1],
            module.in_features,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        rows_per_chunk = self._rows_per_chunk(module)
        with torch.no_grad():
            for start in range(0, module.out_features, rows_per_chunk):
                end = min(module.out_features, start + rows_per_chunk)
                weight = self._dequantize_rows(
                    module,
                    start,
                    end,
                    grad_output.device,
                )
                if weight.dtype != grad_output.dtype:
                    weight = weight.to(grad_output.dtype)
                grad_input.add_(
                    torch.nn.functional.linear(
                        grad_output[..., start:end],
                        weight.transpose(0, 1),
                    )
                )
        return grad_input
