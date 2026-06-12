from __future__ import annotations

import json
import warnings

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F


_BNB_SIBLING_SUFFIXES = (
  ".absmax",
  ".quant_map",
  ".nested_absmax",
  ".nested_quant_map",
)

# Largest magnitude representable by the e4m3 float8 format. Per-row weight
# scales map each row's max abs value onto this so we use the full range.
FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"
# Marker written into the text encoder's config.json so the loader knows to take
# the custom weight-only FP8 path instead of transformers' from_pretrained.
FP8_TEXT_ENCODER_CONFIG_FLAG = "ideogram_fp8_weight_only"

COMFY_QUANT_SUFFIX = ".comfy_quant"
NVFP4_GLOBAL_SCALE_SUFFIX = ".weight_scale_2"
NVFP4_GROUP_SIZE = 16
NVFP4_SCALE_ROW_BLOCK = 128
NVFP4_SCALE_COL_BLOCK = 4
NVFP4_WEIGHT_DTYPE = torch.uint8
COMFY_FLOAT8_FORMAT = "float8_e4m3fn"
COMFY_NVFP4_FORMAT = "nvfp4"


def is_bnb4bit_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  """True if any key looks like a bnb 4-bit quant_state sibling."""
  return any(".quant_state.bitsandbytes__" in k for k in state_dict)


def swap_linears_to_bnb4bit(
  module: nn.Module,
  compute_dtype: torch.dtype,
  *,
  quant_type: str = "nf4",
  compress_statistics: bool = False,
) -> None:
  for name, child in list(module.named_children()):
    if isinstance(child, nn.Linear):
      new_linear = bnb.nn.Linear4bit(
        child.in_features,
        child.out_features,
        bias=child.bias is not None,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
      )
      setattr(module, name, new_linear)
    else:
      swap_linears_to_bnb4bit(
        child,
        compute_dtype,
        quant_type=quant_type,
        compress_statistics=compress_statistics,
      )


def load_bnb4bit_state_dict(
  model: nn.Module,
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
) -> None:
  consumed: set[str] = set()
  for full_name, tensor in state_dict.items():
    if ".quant_state." in full_name or full_name.endswith(_BNB_SIBLING_SUFFIXES):
      continue
    parent_path, _, param_name = full_name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    current = parent._parameters.get(param_name)
    if not isinstance(current, bnb.nn.Params4bit):
      continue
    prefix = full_name + "."
    quantized_stats = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
    # bnb's from_prequantized pops keys it consumes from the dict, so snapshot
    # the names first.
    consumed.add(full_name)
    consumed.update(quantized_stats.keys())
    parent._parameters[param_name] = bnb.nn.Params4bit.from_prequantized(
      data=tensor,
      quantized_stats=quantized_stats,
      requires_grad=False,
      device=device,
    )

  remaining = {k: v for k, v in state_dict.items() if k not in consumed}
  for k in list(remaining):
    if remaining[k].is_floating_point():
      remaining[k] = remaining[k].to(device=device, dtype=dtype)
    else:
      remaining[k] = remaining[k].to(device=device)

  missing, unexpected = model.load_state_dict(remaining, strict=False)
  # Quantized weights are loaded via from_prequantized above, so they appear in
  # `missing` from load_state_dict's perspective — filter those out.
  real_missing = [m for m in missing if m not in consumed]
  if real_missing:
    raise RuntimeError(f"missing keys after quantized load: {real_missing[:10]}")
  if unexpected:
    raise RuntimeError(f"unexpected keys after quantized load: {unexpected[:10]}")

  for p in model.parameters():
    if isinstance(p, bnb.nn.Params4bit):
      continue
    if p.is_floating_point() and p.dtype != dtype:
      p.data = p.data.to(dtype=dtype)
    if p.device != device:
      p.data = p.data.to(device=device)
  for name, b in list(model.named_buffers()):
    if b.is_floating_point() and b.dtype != dtype:
      parent_path, _, leaf = name.rpartition(".")
      parent = model.get_submodule(parent_path) if parent_path else model
      parent.register_buffer(
        leaf,
        b.to(device=device, dtype=dtype),
        persistent=leaf not in parent._non_persistent_buffers_set,
      )
    elif b.device != device:
      parent_path, _, leaf = name.rpartition(".")
      parent = model.get_submodule(parent_path) if parent_path else model
      parent.register_buffer(
        leaf,
        b.to(device=device),
        persistent=leaf not in parent._non_persistent_buffers_set,
      )


# ---------------------------------------------------------------------------
# Weight-only FP8 (e4m3)
#
# Activations stay in the compute dtype (e.g. bfloat16); only Linear weights are
# stored as float8 with a per-output-channel (per-row) float32 scale. At forward
# time the weight is dequantized back to the compute dtype and a normal bf16
# matmul runs, so this needs no FP8 tensor-core hardware and works on any device
# that can store float8 (CPU included). The win is ~2x smaller Linear weights.
# ---------------------------------------------------------------------------


def quantize_weight_to_fp8(
  weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Quantize a 2-D Linear weight to e4m3 float8 with per-row scales.

  Returns ``(weight_fp8, scale)`` where ``weight_fp8`` has shape ``(out, in)``
  in ``float8_e4m3fn`` and ``scale`` has shape ``(out,)`` in float32 such that
  ``weight ≈ weight_fp8.to(dtype) * scale[:, None]``.
  """
  w = weight.detach().to(torch.float32)
  amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
  scale = amax / FP8_E4M3_MAX
  q = (w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
  return q, scale.squeeze(1).to(torch.float32)


def is_fp8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  """True if the checkpoint carries weight-only FP8 Linear weights."""
  return any(k.endswith(FP8_SCALE_SUFFIX) for k in state_dict) or any(
    v.dtype == FP8_WEIGHT_DTYPE for v in state_dict.values()
  )


def decode_comfy_quant_marker(marker: torch.Tensor) -> dict[str, str]:
  """Decode a Comfy ``.comfy_quant`` metadata tensor into its JSON object."""
  if marker.dtype != torch.uint8:
    raise ValueError(
      f"Comfy quant marker tensors must be uint8 bytes, got {marker.dtype}"
    )
  raw = bytes(marker.detach().cpu().reshape(-1).tolist())
  return json.loads(raw.decode("utf-8").rstrip("\x00"))


def _comfy_quant_format(
  state_dict: dict[str, torch.Tensor],
  prefix: str,
) -> str | None:
  marker = state_dict.get(f"{prefix}{COMFY_QUANT_SUFFIX}")
  if marker is None:
    return None
  return str(decode_comfy_quant_marker(marker).get("format", "")).lower()


def is_comfy_quant_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  """True if the checkpoint has Comfy's per-module ``comfy_quant`` markers."""
  return any(k.endswith(COMFY_QUANT_SUFFIX) for k in state_dict)


def is_nvfp4_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  """True if the checkpoint carries Comfy NVFP4 packed weights."""
  if any(k.endswith(NVFP4_GLOBAL_SCALE_SUFFIX) for k in state_dict):
    return True
  return any(
    k.endswith(COMFY_QUANT_SUFFIX)
    and _comfy_quant_format(state_dict, k.removesuffix(COMFY_QUANT_SUFFIX))
    == COMFY_NVFP4_FORMAT
    for k in state_dict
  )


def _fp8_scale_view(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  while scale.dim() < weight.dim():
    scale = scale.unsqueeze(-1)
  return scale


def _round_up(value: int, multiple: int) -> int:
  return ((value + multiple - 1) // multiple) * multiple


def _ceil_div(value: int, divisor: int) -> int:
  return (value + divisor - 1) // divisor


def dequantize_fp8_weight(
  weight: torch.Tensor,
  scale: torch.Tensor,
  *,
  device: torch.device,
  dtype: torch.dtype,
) -> torch.Tensor:
  """Dequantize an FP8 weight tensor with scalar or per-row scale."""
  w = weight.to(device=device, dtype=dtype)
  s = scale.to(device=device, dtype=dtype)
  return w * _fp8_scale_view(w, s)


def _unswizzle_comfy_block_scales(
  blocked_scales: torch.Tensor,
  *,
  num_rows: int,
  num_cols: int,
) -> torch.Tensor:
  """Undo Comfy Kitchen/cuBLAS SWIZZLE_32_4_4 scale layout."""
  padded_rows = _round_up(num_rows, NVFP4_SCALE_ROW_BLOCK)
  padded_cols = _round_up(num_cols, NVFP4_SCALE_COL_BLOCK)
  if blocked_scales.dim() != 2:
    raise RuntimeError(
      f"NVFP4 block scales must be rank 2, got {tuple(blocked_scales.shape)}"
    )
  if blocked_scales.shape[0] < padded_rows or blocked_scales.shape[1] < padded_cols:
    raise RuntimeError(
      "NVFP4 block scales are too small for Comfy's swizzled layout: "
      f"got {tuple(blocked_scales.shape)}, need at least "
      f"({padded_rows}, {padded_cols})"
    )

  n_row_blocks = _ceil_div(num_rows, NVFP4_SCALE_ROW_BLOCK)
  n_col_blocks = _ceil_div(num_cols, NVFP4_SCALE_COL_BLOCK)
  blocked = blocked_scales[:padded_rows, :padded_cols].contiguous()
  unblocked = (
    blocked.reshape(-1, 32, 16)
    .reshape(-1, 32, 4, 4)
    .transpose(1, 2)
    .reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    .reshape(n_row_blocks, n_col_blocks, 128, 4)
    .permute(0, 2, 1, 3)
    .reshape(padded_rows, padded_cols)
  )
  return unblocked[:num_rows, :num_cols]


def _get_submodule_or_none(module: nn.Module, path: str) -> nn.Module | None:
  try:
    return module.get_submodule(path) if path else module
  except AttributeError:
    return None


def _move_non_meta_tensors_to_device(module: nn.Module, device: torch.device) -> None:
  """Move materialized tensors while leaving unresolved meta tensors untouched."""
  for full_name, param in list(module.named_parameters(recurse=True)):
    if param is None or param.is_meta or param.device == device:
      continue
    parent_path, _, leaf = full_name.rpartition(".")
    parent = module.get_submodule(parent_path) if parent_path else module
    parent._parameters[leaf] = nn.Parameter(
      param.to(device=device),
      requires_grad=param.requires_grad,
    )

  for full_name, buf in list(module.named_buffers(recurse=True)):
    if buf is None or buf.is_meta or buf.device == device:
      continue
    parent_path, _, leaf = full_name.rpartition(".")
    parent = module.get_submodule(parent_path) if parent_path else module
    parent.register_buffer(
      leaf,
      buf.to(device=device),
      persistent=leaf not in parent._non_persistent_buffers_set,
    )


def _finalize_quantized_load(
  model: nn.Module,
  device: torch.device,
  *,
  assign: bool,
  strict: bool,
) -> None:
  meta_names = [
    name
    for name, tensor in (
      list(model.named_parameters(recurse=True))
      + list(model.named_buffers(recurse=True))
    )
    if tensor is not None and tensor.is_meta
  ]
  if not meta_names:
    model.to(device)
    return

  if strict or not assign:
    raise RuntimeError(
      "quantized load left meta tensors unresolved: "
      f"{meta_names[:10]}. This usually means checkpoint keys are missing."
    )

  _move_non_meta_tensors_to_device(model, device)


class Fp8Linear(nn.Module):
  """Linear layer holding an e4m3 float8 weight + per-row float32 scale.

  The weight and scale are registered as buffers (not parameters) so they load
  via ``load_state_dict`` and are excluded from optimizer/grad machinery. The
  dequantized matmul runs in ``compute_dtype``.
  """

  weight: torch.Tensor
  weight_scale: torch.Tensor
  bias: torch.Tensor | None

  def __init__(
    self,
    in_features: int,
    out_features: int,
    bias: bool,
    compute_dtype: torch.dtype,
    scale_shape: torch.Size | tuple[int, ...] | None = None,
  ) -> None:
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.compute_dtype = compute_dtype
    self.register_buffer(
      "weight",
      torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE),
    )
    scale_shape = tuple(scale_shape) if scale_shape is not None else (out_features,)
    self.register_buffer("weight_scale", torch.empty(scale_shape, dtype=torch.float32))
    if bias:
      self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
    else:
      self.bias = None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    w = dequantize_fp8_weight(
      self.weight,
      self.weight_scale,
      device=x.device,
      dtype=x.dtype,
    )
    bias = self.bias.to(x.dtype) if self.bias is not None else None
    return F.linear(x, w, bias)


class Nvfp4Linear(nn.Module):
  """Linear layer holding Comfy packed NVFP4 weights.

  The packed uint8 weight, e4m3 block scales, and scalar global scale are frozen
  buffers. Forward expands nibbles through the E2M1 value table, applies the
  scales, trims any padded input columns, and runs a normal dense matmul.
  """

  weight: torch.Tensor
  weight_scale: torch.Tensor
  weight_scale_2: torch.Tensor
  bias: torch.Tensor | None

  def __init__(
    self,
    in_features: int,
    out_features: int,
    bias: bool,
    compute_dtype: torch.dtype,
    *,
    packed_weight_shape: torch.Size | tuple[int, ...] | None = None,
    block_scale_shape: torch.Size | tuple[int, ...] | None = None,
  ) -> None:
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.compute_dtype = compute_dtype

    group_count = (in_features + NVFP4_GROUP_SIZE - 1) // NVFP4_GROUP_SIZE
    padded_in = group_count * NVFP4_GROUP_SIZE
    default_weight_shape = (out_features, padded_in // 2)
    default_scale_shape = (
      _round_up(out_features, NVFP4_SCALE_ROW_BLOCK),
      _round_up(group_count, NVFP4_SCALE_COL_BLOCK),
    )

    weight_shape = (
      tuple(packed_weight_shape)
      if packed_weight_shape is not None
      else default_weight_shape
    )
    scale_shape = (
      tuple(block_scale_shape)
      if block_scale_shape is not None
      else default_scale_shape
    )
    self.register_buffer("weight", torch.empty(weight_shape, dtype=NVFP4_WEIGHT_DTYPE))
    self.register_buffer(
      "weight_scale", torch.empty(scale_shape, dtype=FP8_WEIGHT_DTYPE)
    )
    self.register_buffer("weight_scale_2", torch.empty((), dtype=torch.float32))
    if bias:
      self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
    else:
      self.bias = None

  @staticmethod
  def e2m1_table(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
      [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
      ],
      device=device,
      dtype=dtype,
    )

  def dequantize_weight(
    self,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ) -> torch.Tensor:
    device = device or self.weight.device
    dtype = dtype or self.compute_dtype

    packed = self.weight.to(device=device)
    high = torch.bitwise_right_shift(packed, 4).to(torch.long)
    low = torch.bitwise_and(packed, 0x0F).to(torch.long)
    codes = torch.stack((high, low), dim=-1).reshape(packed.shape[0], -1)

    group_count = (packed.shape[1] * 2) // NVFP4_GROUP_SIZE
    padded_in = group_count * NVFP4_GROUP_SIZE
    if codes.shape[1] < padded_in:
      raise RuntimeError(
        "NVFP4 packed weight has too few columns for its block scales: "
        f"packed expands to {codes.shape[1]}, scales imply {padded_in}"
      )

    block_scales = _unswizzle_comfy_block_scales(
      self.weight_scale.to(device=device, dtype=torch.float32),
      num_rows=packed.shape[0],
      num_cols=group_count,
    )
    table = self.e2m1_table(device=device, dtype=torch.float32)
    values = table[codes[:, :padded_in]]
    values = values.view(packed.shape[0], group_count, NVFP4_GROUP_SIZE)
    values = values * block_scales.unsqueeze(-1)
    values = values[: self.out_features].flatten(1)[:, : self.in_features]
    values = values * self.weight_scale_2.to(device=device, dtype=torch.float32)
    return values.to(dtype=dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    w = self.dequantize_weight(device=x.device, dtype=x.dtype)
    bias = self.bias.to(x.dtype) if self.bias is not None else None
    return F.linear(x, w, bias)


def swap_linears_to_fp8(
  module: nn.Module,
  state_dict: dict[str, torch.Tensor],
  compute_dtype: torch.dtype,
  *,
  prefix: str = "",
) -> None:
  """Replace each ``nn.Linear`` that has a saved FP8 scale with an ``Fp8Linear``.

  Gating on the presence of ``<name>.weight_scale`` means only layers that were
  actually quantized at save time are swapped; everything else loads normally in
  the compute dtype.
  """
  for name, child in list(module.named_children()):
    child_prefix = f"{prefix}{name}"
    if (
      isinstance(child, nn.Linear) and f"{child_prefix}{FP8_SCALE_SUFFIX}" in state_dict
    ):
      scale = state_dict[f"{child_prefix}{FP8_SCALE_SUFFIX}"]
      setattr(
        module,
        name,
        Fp8Linear(
          child.in_features,
          child.out_features,
          bias=child.bias is not None,
          compute_dtype=compute_dtype,
          scale_shape=scale.shape,
        ),
      )
    else:
      swap_linears_to_fp8(child, state_dict, compute_dtype, prefix=f"{child_prefix}.")


def _is_nvfp4_linear_state(
  state_dict: dict[str, torch.Tensor],
  prefix: str,
) -> bool:
  fmt = _comfy_quant_format(state_dict, prefix)
  if fmt == COMFY_NVFP4_FORMAT:
    return True
  return (
    f"{prefix}{NVFP4_GLOBAL_SCALE_SUFFIX}" in state_dict
    and state_dict.get(f"{prefix}.weight") is not None
    and state_dict[f"{prefix}.weight"].dtype == NVFP4_WEIGHT_DTYPE
  )


def _is_fp8_linear_state(
  state_dict: dict[str, torch.Tensor],
  prefix: str,
) -> bool:
  fmt = _comfy_quant_format(state_dict, prefix)
  if fmt == COMFY_FLOAT8_FORMAT:
    return True
  return (
    f"{prefix}{FP8_SCALE_SUFFIX}" in state_dict
    and state_dict.get(f"{prefix}.weight") is not None
    and state_dict[f"{prefix}.weight"].dtype == FP8_WEIGHT_DTYPE
  )


def swap_linears_to_comfy_quant(
  module: nn.Module,
  state_dict: dict[str, torch.Tensor],
  compute_dtype: torch.dtype,
  *,
  prefix: str = "",
) -> None:
  """Replace ``nn.Linear`` modules described by Comfy quant markers."""
  for name, child in list(module.named_children()):
    child_prefix = f"{prefix}{name}"
    if isinstance(child, nn.Linear) and _is_nvfp4_linear_state(
      state_dict, child_prefix
    ):
      weight = state_dict[f"{child_prefix}.weight"]
      block_scale = state_dict[f"{child_prefix}{FP8_SCALE_SUFFIX}"]
      setattr(
        module,
        name,
        Nvfp4Linear(
          child.in_features,
          child.out_features,
          bias=child.bias is not None,
          compute_dtype=compute_dtype,
          packed_weight_shape=weight.shape,
          block_scale_shape=block_scale.shape,
        ),
      )
    elif isinstance(child, nn.Linear) and _is_fp8_linear_state(
      state_dict, child_prefix
    ):
      scale = state_dict[f"{child_prefix}{FP8_SCALE_SUFFIX}"]
      setattr(
        module,
        name,
        Fp8Linear(
          child.in_features,
          child.out_features,
          bias=child.bias is not None,
          compute_dtype=compute_dtype,
          scale_shape=scale.shape,
        ),
      )
    else:
      swap_linears_to_comfy_quant(
        child, state_dict, compute_dtype, prefix=f"{child_prefix}."
      )


def load_fp8_state_dict(
  model: nn.Module,
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
  *,
  assign: bool = False,
  strict: bool = True,
) -> None:
  """Load a weight-only FP8 checkpoint into ``model``.

  ``model`` must already have its FP8 Linear layers swapped in (see
  ``swap_linears_to_fp8``). FP8 weights are kept as float8, scales stay float32,
  and every other floating tensor is cast to ``dtype``.

  ``assign=True`` replaces the module's tensors with the prepared ones rather than
  copying into them. Use it when the model was built with ``from_config`` so the
  non-quantized params take the loaded dtype directly and computed non-persistent
  buffers (e.g. rotary caches) are left untouched. With ``assign=False`` (default),
  the caller must have already put the unquantized params in ``dtype``.

  ``strict=False`` downgrades missing keys to a warning (e.g. tied weights that a
  ``transformers`` model resolves itself); unexpected keys always raise.
  """
  prepared: dict[str, torch.Tensor] = {}
  for k, v in state_dict.items():
    if v.dtype == FP8_WEIGHT_DTYPE:
      prepared[k] = v.to(device=device)
    elif k.endswith(FP8_SCALE_SUFFIX):
      prepared[k] = v.to(device=device, dtype=torch.float32)
    elif v.is_floating_point():
      prepared[k] = v.to(device=device, dtype=dtype)
    else:
      prepared[k] = v.to(device=device)

  missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
  if unexpected:
    raise RuntimeError(f"unexpected keys after fp8 load: {unexpected[:10]}")
  if missing:
    if strict:
      raise RuntimeError(f"missing keys after fp8 load: {missing[:10]}")
    warnings.warn(f"missing keys after fp8 load: {missing[:10]}", stacklevel=2)

  _finalize_quantized_load(model, device, assign=assign, strict=strict)


def _prepare_comfy_tensor(
  model: nn.Module,
  state_dict: dict[str, torch.Tensor],
  key: str,
  tensor: torch.Tensor,
  device: torch.device,
  dtype: torch.dtype,
) -> torch.Tensor | None:
  parent_path, _, leaf = key.rpartition(".")
  parent = _get_submodule_or_none(model, parent_path)
  fmt = _comfy_quant_format(state_dict, parent_path)

  if key.endswith(COMFY_QUANT_SUFFIX):
    return None

  if leaf == "weight_scale_2":
    return tensor.to(device=device, dtype=torch.float32) if isinstance(
      parent, Nvfp4Linear
    ) else None

  if leaf == "weight_scale":
    if isinstance(parent, (Fp8Linear, Nvfp4Linear)):
      scale_dtype = FP8_WEIGHT_DTYPE if isinstance(parent, Nvfp4Linear) else torch.float32
      return tensor.to(device=device, dtype=scale_dtype)
    return None

  if leaf == "weight" and fmt == COMFY_FLOAT8_FORMAT and not isinstance(
    parent, Fp8Linear
  ):
    scale = state_dict.get(f"{parent_path}{FP8_SCALE_SUFFIX}")
    if scale is None:
      raise RuntimeError(f"Comfy FP8 weight {key!r} is missing its weight_scale")
    return dequantize_fp8_weight(tensor, scale, device=device, dtype=dtype)

  if leaf == "weight" and fmt == COMFY_NVFP4_FORMAT and not isinstance(
    parent, Nvfp4Linear
  ):
    raise RuntimeError(
      f"Comfy NVFP4 weight {key!r} did not map to a Linear module. Packed "
      "NVFP4 is only supported for Linear weights."
    )

  if isinstance(parent, Fp8Linear) and (leaf == "weight" or tensor.dtype == FP8_WEIGHT_DTYPE):
    return tensor.to(device=device)

  if isinstance(parent, Nvfp4Linear) and leaf == "weight":
    return tensor.to(device=device)

  if tensor.is_floating_point():
    return tensor.to(device=device, dtype=dtype)
  return tensor.to(device=device)


def load_comfy_quant_state_dict(
  model: nn.Module,
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
  *,
  assign: bool = False,
  strict: bool = True,
) -> None:
  """Load a Comfy quantized checkpoint into a model with swapped modules."""
  expected_keys = set(model.state_dict().keys())
  prepared: dict[str, torch.Tensor] = {}
  for key, tensor in state_dict.items():
    if not strict and key not in expected_keys and not key.endswith(
      (COMFY_QUANT_SUFFIX, FP8_SCALE_SUFFIX, NVFP4_GLOBAL_SCALE_SUFFIX)
    ):
      continue
    prepared_tensor = _prepare_comfy_tensor(
      model,
      state_dict,
      key,
      tensor,
      device=device,
      dtype=dtype,
    )
    if prepared_tensor is not None:
      prepared[key] = prepared_tensor

  missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
  if unexpected:
    raise RuntimeError(f"unexpected keys after comfy quant load: {unexpected[:10]}")
  if missing:
    if strict:
      raise RuntimeError(f"missing keys after comfy quant load: {missing[:10]}")
    warnings.warn(
      f"missing keys after comfy quant load: {missing[:10]}",
      stacklevel=2,
    )

  _finalize_quantized_load(model, device, assign=assign, strict=strict)
