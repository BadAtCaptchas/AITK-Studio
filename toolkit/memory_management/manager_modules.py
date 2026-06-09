"""
This code was heavily inspired by the work of Lodestone-Rock, pretty much all credit goes
to them. The original code can be found here:
https://github.com/lodestone-rock/RamTorch/blob/main/ramtorch/modules/linear.py

I simply modified it to work with a memory management model and with AI Toolkit's models
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional, Tuple
from torch.overrides import has_torch_function_unary  # (ADD) torchao detection

if TYPE_CHECKING:
    from .manager import MemoryManager

# --- Per-device global state registry ---
_DEVICE_STATE = {}

# How many layers deep to prefetch weights. The old ping-pong used 2 slots, which
# only lets one transfer overlap one compute. Override with AI_TOOLKIT_OFFLOAD_DEPTH.
PIPELINE_DEPTH = int(os.environ.get("AI_TOOLKIT_OFFLOAD_DEPTH", "4"))


def _get_device_state(device: torch.device):
    """Get or initialize per-device state."""
    if isinstance(device, str):
        device = torch.device(device)

    # CPU path needs no CUDA state
    if device.type != "cuda":
        if device not in _DEVICE_STATE:
            _DEVICE_STATE[device] = {}
        return _DEVICE_STATE[device]

    if device not in _DEVICE_STATE:
        d = max(2, PIPELINE_DEPTH)
        with torch.cuda.device(device):
            _DEVICE_STATE[device] = {
                "depth": d,
                # streams
                "transfer_stream": torch.cuda.Stream(device=device),
                "transfer_grad_stream": torch.cuda.Stream(device=device),
                # forward weight ring: slot_ready = H2D done, slot_free = compute
                # that consumed the slot done, so it can be overwritten.
                "w_buffers": [None] * d,
                "b_buffers": [None] * d,
                "fwd_slot_ready": [torch.cuda.Event() for _ in range(d)],
                "fwd_slot_free": [torch.cuda.Event() for _ in range(d)],
                "forward_clk": 0,
                # backward weight ring: re-fetch for grad-input.
                "w_bwd_buffers": [None] * d,
                "bwd_slot_ready": [torch.cuda.Event() for _ in range(d)],
                "bwd_slot_free": [torch.cuda.Event() for _ in range(d)],
                "backward_clk": 0,
                # backward grad-staging ring: device-side grads -> CPU.
                "w_grad_buffers": [None] * d,
                "b_grad_buffers": [None] * d,
                "grad_compute_done": [torch.cuda.Event() for _ in range(d)],
                "grad_xfer_done": [torch.cuda.Event() for _ in range(d)],
            }
    return _DEVICE_STATE[device]


def _stage_forward_weight(
    state, device, materialize, weight_cpu, bias_cpu, materialize_bias=None
):
    d = state["depth"]
    idx = state["forward_clk"]
    state["forward_clk"] = (idx + 1) % d
    ts = state["transfer_stream"]
    with torch.cuda.stream(ts):
        ts.wait_event(state["fwd_slot_free"][idx])
        state["w_buffers"][idx] = materialize(weight_cpu, device)
        if bias_cpu is not None:
            if materialize_bias is None:
                state["b_buffers"][idx] = bias_cpu.to(device, non_blocking=True)
            else:
                state["b_buffers"][idx] = materialize_bias(bias_cpu, device)
        else:
            state["b_buffers"][idx] = None
        state["fwd_slot_ready"][idx].record()
    torch.cuda.current_stream().wait_event(state["fwd_slot_ready"][idx])
    return idx, state["w_buffers"][idx], state["b_buffers"][idx]


def _release_forward_slot(state, idx):
    state["fwd_slot_free"][idx].record()


def _stage_backward_weight(state, device, materialize, weight_cpu):
    d = state["depth"]
    idx = state["backward_clk"]
    state["backward_clk"] = (idx + 1) % d
    ts = state["transfer_stream"]
    with torch.cuda.stream(ts):
        ts.wait_event(state["bwd_slot_free"][idx])
        state["w_bwd_buffers"][idx] = materialize(weight_cpu)
        state["bwd_slot_ready"][idx].record()
    torch.cuda.current_stream().wait_event(state["bwd_slot_ready"][idx])
    return idx, state["w_bwd_buffers"][idx]


def _release_backward_weight_slot(state, idx):
    state["bwd_slot_free"][idx].record()


def _stage_grads_to_cpu(state, idx, grad_w_gpu, grad_b_gpu):
    gs = state["transfer_grad_stream"]
    state["grad_compute_done"][idx].record()
    grad_w_cpu = grad_b_cpu = None
    with torch.cuda.stream(gs):
        gs.wait_event(state["grad_compute_done"][idx])
        if grad_w_gpu is not None:
            grad_w_cpu = grad_w_gpu.to("cpu", non_blocking=True)
        if grad_b_gpu is not None:
            grad_b_cpu = grad_b_gpu.to("cpu", non_blocking=True)
        state["grad_xfer_done"][idx].record()
    return grad_w_cpu, grad_b_cpu


# (ADD) detect torchao wrapper tensors
def _is_ao_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
    if t is None:
        return False
    try:
        if has_torch_function_unary(t):
            return t.__class__.__module__.startswith("torchao.")
    except Exception:
        pass
    for attr in (
        "_scale",
        "_scales",
        "_zero_point",
        "_zp",
        "_block_size",
        "_group_size",
        "_pack_dim",
    ):
        if hasattr(t, attr):
            return True
    return False


def _is_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
    if t is None:
        return False
    # torch quantized tensors
    try:
        if torch.is_quantized(t):  # type: ignore[attr-defined]
            return True
    except Exception:
        pass
    # (ADD) torchao quantized wrappers
    if _is_ao_quantized_tensor(t):
        return True
    # bitsandbytes quantized wrappers, including Params4bit used by Linear4bit.
    try:
        cls = t.__class__
        if cls.__name__ in {"Params4bit", "Int8Params"}:
            return True
        if cls.__module__.startswith("bitsandbytes."):
            return True
    except Exception:
        pass
    try:
        if getattr(t, "quant_state", None) is not None:
            return True
    except Exception:
        pass
    # packed/int formats (weight-only)
    try:
        return not t.dtype.is_floating_point
    except Exception:
        return False


def _prod(shape: Tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _safe_to(t: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    try:
        return t.to(*args, **kwargs)
    except TypeError:
        kwargs.pop("non_blocking", None)
        return t.to(*args, **kwargs)


def _safe_to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    try:
        if t.device == device:
            return t
    except Exception:
        pass
    return _safe_to(t, device, non_blocking=True)


def _is_float8_dtype(dtype: Optional[torch.dtype]) -> bool:
    return dtype is not None and str(dtype).startswith("torch.float8")


def _linear_logical_shape(module: nn.Module) -> Optional[Tuple[int, int]]:
    try:
        return (int(module.out_features), int(module.in_features))
    except Exception:
        return None


def _reshape_to_logical_linear_shape(
    weight: torch.Tensor, logical_shape: Optional[Tuple[int, int]]
) -> torch.Tensor:
    if logical_shape is None:
        return weight
    try:
        if (
            tuple(weight.shape) != logical_shape
            and weight.numel() == _prod(logical_shape)
        ):
            return weight.reshape(logical_shape)
    except Exception:
        pass
    return weight


def _materialize_linear_weight(
    weight_cpu: torch.Tensor,
    device: torch.device,
    target_dtype: torch.dtype,
    logical_shape: Optional[Tuple[int, int]] = None,
    weight_scale_cpu: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    is_quantized = _is_quantized_tensor(weight_cpu)
    weight = _safe_to_device(weight_cpu, device)

    if is_quantized or _is_quantized_tensor(weight):
        dequantize = getattr(weight, "dequantize", None)
        if callable(dequantize):
            try:
                weight = dequantize()
            except Exception:
                pass
        if _is_quantized_tensor(weight) or not getattr(
            getattr(weight, "dtype", None), "is_floating_point", False
        ):
            weight = _safe_to(weight, dtype=torch.float32, non_blocking=True)

    weight = _reshape_to_logical_linear_shape(weight, logical_shape)

    should_cast = (
        is_quantized
        or weight_scale_cpu is not None
        or _is_float8_dtype(getattr(weight, "dtype", None))
        or not getattr(getattr(weight, "dtype", None), "is_floating_point", False)
    )
    if should_cast and weight.dtype != target_dtype:
        weight = _safe_to(weight, dtype=target_dtype, non_blocking=True)

    if weight_scale_cpu is not None:
        scale = _safe_to_device(weight_scale_cpu, device)
        if getattr(scale, "dtype", None) != weight.dtype:
            scale = _safe_to(scale, dtype=weight.dtype, non_blocking=True)
        scale = scale.view(-1, *[1] * (weight.dim() - 1))
        weight = weight * scale

    return weight


def _materialize_linear_bias(
    bias_cpu: torch.Tensor,
    device: torch.device,
    target_dtype: torch.dtype,
    cast_bias: bool,
) -> torch.Tensor:
    bias = _safe_to_device(bias_cpu, device)
    if cast_bias and bias.dtype != target_dtype:
        bias = _safe_to(bias, dtype=target_dtype, non_blocking=True)
    return bias


def _pin_inner_tensors(t: torch.Tensor) -> None:
    """Pin tensor-subclass leaf storage, such as torchao float8 internals."""
    try:
        names, _ = t.__tensor_flatten__()
    except Exception:
        return
    for name in names:
        inner = getattr(t, name, None)
        if inner is None:
            continue
        if hasattr(inner, "__tensor_flatten__"):
            _pin_inner_tensors(inner)
        elif (
            isinstance(inner, torch.Tensor)
            and inner.device.type == "cpu"
            and not inner.is_pinned()
        ):
            try:
                setattr(t, name, inner.pin_memory())
            except Exception:
                pass


def _ensure_cpu_pinned(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if t.device.type != "cpu":
        try:
            t = t.to("cpu", copy=True)
        except Exception:
            t = t.to("cpu")
    # Quantized wrappers cannot always be pin_memory()'d directly, but pinning
    # their inner storage still allows async H2D transfer for supported wrappers.
    if _is_quantized_tensor(t):
        if torch.cuda.is_available():
            _pin_inner_tensors(t)
        return t
    if torch.cuda.is_available():
        try:
            t = t.pin_memory()
        except RuntimeError:
            pass
    return t


def _move_params_to_cpu_and_pin(module: nn.Module):
    """Force parameters/buffers to CPU (+pinned) so we can bounce them per call."""
    with torch.no_grad():
        for name in ("weight", "bias"):
            param = getattr(module, name, None)
            if not isinstance(param, nn.Parameter):
                continue
            if _is_quantized_tensor(param):
                try:
                    cpu_param = _ensure_cpu_pinned(param)
                    if isinstance(cpu_param, nn.Parameter):
                        module._parameters[name] = cpu_param
                        continue
                except Exception:
                    pass
            cpu_data = _ensure_cpu_pinned(param.data).detach()
            try:
                param.data = cpu_data
            except Exception:
                if not _is_quantized_tensor(param):
                    setattr(
                        module,
                        name,
                        nn.Parameter(cpu_data, requires_grad=param.requires_grad),
                    )

        for name in ("weight", "bias", "weight_scale"):
            if name not in module._buffers:
                continue
            buffer = module._buffers[name]
            if not isinstance(buffer, torch.Tensor):
                continue
            cpu_buffer = _ensure_cpu_pinned(buffer)
            if _is_quantized_tensor(cpu_buffer):
                module._buffers[name] = cpu_buffer
            else:
                module._buffers[name] = cpu_buffer.detach()


# ==========================
# Autograd functions (CUDA)
# ==========================


class _BouncingLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight_cpu,
        bias_cpu,
        weight_scale_cpu,
        device: torch.device,
        logical_shape: Optional[Tuple[int, int]],
    ):
        # choose compute dtype to match activations
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )
        cast_linear_bias = (
            _is_quantized_tensor(weight_cpu)
            or weight_scale_cpu is not None
            or _is_float8_dtype(getattr(weight_cpu, "dtype", None))
        )

        def _materialize_weight(cpu_w, dev):
            return _materialize_linear_weight(
                cpu_w,
                dev,
                target_dtype,
                logical_shape=logical_shape,
                weight_scale_cpu=weight_scale_cpu,
            )

        def _materialize_bias(cpu_b, dev):
            return _materialize_linear_bias(
                cpu_b, dev, target_dtype, cast_linear_bias
            )

        if device.type != "cuda":
            x_cpu = x.to("cpu")
            if cast_linear_bias and x_cpu.dtype != target_dtype:
                x_cpu = x_cpu.to(target_dtype)
            out = F.linear(
                x_cpu,
                _materialize_weight(weight_cpu, torch.device("cpu")),
                _materialize_bias(bias_cpu, torch.device("cpu"))
                if bias_cpu is not None
                else None,
            )
            ctx.save_for_backward(x_cpu, weight_cpu, bias_cpu, weight_scale_cpu)
            ctx.device = torch.device("cpu")
            ctx.logical_shape = logical_shape
            ctx.target_dtype = target_dtype
            ctx.cast_linear_bias = cast_linear_bias
            return out.to(x.device)

        state = _get_device_state(device)
        idx, w_gpu, b_gpu = _stage_forward_weight(
            state,
            device,
            _materialize_weight,
            weight_cpu,
            bias_cpu,
            _materialize_bias,
        )
        out = F.linear(x, w_gpu, b_gpu)
        _release_forward_slot(state, idx)

        ctx.save_for_backward(x, weight_cpu, bias_cpu, weight_scale_cpu)
        ctx.device = device
        ctx.logical_shape = logical_shape
        ctx.target_dtype = target_dtype
        ctx.cast_linear_bias = cast_linear_bias
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu, weight_scale_cpu = ctx.saved_tensors
        device = ctx.device
        target_dtype = getattr(ctx, "target_dtype", grad_out.dtype)
        logical_shape = getattr(ctx, "logical_shape", None)

        if device.type != "cuda":
            go_cpu = grad_out.to("cpu")
            x_cpu = x.to("cpu")
            w_mat = _materialize_linear_weight(
                weight_cpu,
                torch.device("cpu"),
                target_dtype,
                logical_shape=logical_shape,
                weight_scale_cpu=weight_scale_cpu,
            )
            if go_cpu.dtype != w_mat.dtype:
                go_cpu = go_cpu.to(w_mat.dtype)
            grad_input = go_cpu @ w_mat
            grad_weight = (
                go_cpu.flatten(0, -2).T @ x_cpu.flatten(0, -2)
                if getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
                else None
            )
            grad_bias = (
                go_cpu.sum(dim=tuple(range(go_cpu.ndim - 1)))
                if (bias_cpu is not None and getattr(bias_cpu, "requires_grad", False))
                else None
            )
            return (
                grad_input.to(grad_out.device),
                grad_weight,
                grad_bias,
                None,
                None,
                None,
            )

        state = _get_device_state(device)

        def _materialize_for_bwd(cpu_w):
            return _materialize_linear_weight(
                cpu_w,
                device,
                target_dtype,
                logical_shape=logical_shape,
                weight_scale_cpu=weight_scale_cpu,
            )

        idx, w_bwd = _stage_backward_weight(
            state, device, _materialize_for_bwd, weight_cpu
        )

        # grad wrt input (GPU)
        grad_input = grad_out.to(dtype=target_dtype) @ w_bwd
        _release_backward_weight_slot(state, idx)

        # compute grads if float masters exist (frozen/quantized bases skip this)
        grad_weight = None
        grad_bias = None
        need_w = (
            getattr(weight_cpu, "requires_grad", False)
            and weight_cpu.dtype.is_floating_point
        )
        need_b = bias_cpu is not None and getattr(bias_cpu, "requires_grad", False)
        if need_w or need_b:
            torch.cuda.current_stream().wait_event(state["grad_xfer_done"][idx])
            w_grad_gpu = b_grad_gpu = None
            if need_w:
                w_grad_gpu = grad_out.flatten(0, -2).T @ x.flatten(0, -2)
                state["w_grad_buffers"][idx] = w_grad_gpu
            if need_b:
                b_grad_gpu = grad_out.sum(dim=tuple(range(grad_out.ndim - 1)))
                state["b_grad_buffers"][idx] = b_grad_gpu
            grad_weight, grad_bias = _stage_grads_to_cpu(
                state, idx, w_grad_gpu, b_grad_gpu
            )

        return (
            grad_input.to(dtype=grad_out.dtype),
            grad_weight,
            grad_bias,
            None,
            None,
            None,
        )


class _BouncingConv2dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight_cpu,
        bias_cpu,
        device: torch.device,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
    ):
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        # GPU-side dequant/cast for quantized; float path unchanged
        def _materialize_conv_weight(cpu_w, dev):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = cpu_w.to(dev, non_blocking=True)
                try:
                    w_fp_gpu = w_q_gpu.dequantize()
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            # float path (preserve original behavior: NO dtype cast)
            w_gpu = cpu_w.to(dev, non_blocking=True)
            return w_gpu

        if device.type != "cuda":
            out = F.conv2d(
                x.to("cpu"),
                _materialize_conv_weight(weight_cpu, torch.device("cpu")),
                bias_cpu,
                stride,
                padding,
                dilation,
                groups,
            )
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.meta = ("cpu", stride, padding, dilation, groups, target_dtype)
            return out.to(x.device)

        state = _get_device_state(device)
        idx, w_gpu, b_gpu = _stage_forward_weight(
            state, device, _materialize_conv_weight, weight_cpu, bias_cpu
        )
        out = F.conv2d(x, w_gpu, b_gpu, stride, padding, dilation, groups)
        _release_forward_slot(state, idx)

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.meta = (device, stride, padding, dilation, groups, target_dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device, stride, padding, dilation, groups, target_dtype = ctx.meta

        if (
            isinstance(device, torch.device) and device.type != "cuda"
        ) or device == "cpu":
            go = grad_out.to("cpu")
            x_cpu = x.to("cpu")
            w_cpu = (
                weight_cpu.dequantize()
                if _is_quantized_tensor(weight_cpu)
                else weight_cpu
            )
            if w_cpu.dtype != target_dtype and target_dtype in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
            ):
                w_cpu = w_cpu.to(target_dtype)
            from torch.nn.grad import conv2d_input, conv2d_weight  # type: ignore

            grad_input = conv2d_input(
                x_cpu.shape,
                w_cpu,
                go,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            grad_weight = (
                conv2d_weight(
                    x_cpu,
                    w_cpu.shape,
                    go,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                if getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
                else None
            )
            grad_bias = (
                go.sum(dim=(0, 2, 3))
                if (bias_cpu is not None and getattr(bias_cpu, "requires_grad", False))
                else None
            )
            return (
                grad_input.to(grad_out.device),
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
            )

        state = _get_device_state(device)

        # GPU-side dequant/cast for quantized; float path unchanged
        def _materialize_for_bwd(cpu_w):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = cpu_w.to(device, non_blocking=True)
                try:
                    w_fp_gpu = w_q_gpu.dequantize()
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            # float path (preserve original behavior: NO dtype cast)
            w = cpu_w.to(device, non_blocking=True)
            return w

        idx, w_bwd = _stage_backward_weight(
            state, device, _materialize_for_bwd, weight_cpu
        )

        from torch.nn.grad import conv2d_input, conv2d_weight  # type: ignore

        grad_input = conv2d_input(
            x.shape,
            w_bwd,
            grad_out.to(dtype=target_dtype),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        _release_backward_weight_slot(state, idx)

        # Compute heavy grads on GPU into staging buffers (frozen bases skip this)
        grad_weight = None
        grad_bias = None
        need_w = (
            getattr(weight_cpu, "requires_grad", False)
            and weight_cpu.dtype.is_floating_point
        )
        need_b = bias_cpu is not None and getattr(bias_cpu, "requires_grad", False)
        if need_w or need_b:
            torch.cuda.current_stream().wait_event(state["grad_xfer_done"][idx])
            w_grad_gpu = b_grad_gpu = None
            if need_w:
                w_grad_gpu = conv2d_weight(
                    x,
                    weight_cpu.shape,
                    grad_out,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                state["w_grad_buffers"][idx] = w_grad_gpu
            if need_b:
                b_grad_gpu = grad_out.sum(dim=(0, 2, 3))
                state["b_grad_buffers"][idx] = b_grad_gpu
            grad_weight, grad_bias = _stage_grads_to_cpu(
                state, idx, w_grad_gpu, b_grad_gpu
            )

        return (
            grad_input.to(dtype=grad_out.dtype),
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )


class BaseLayerMemoryManager:
    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        self.module: nn.Module = module
        self.manager: "MemoryManager" = manager

    @classmethod
    def attach(cls, module: nn.Module, manager: "MemoryManager"):
        if hasattr(module, "_layer_memory_manager"):
            return
        module._layer_memory_manager = cls(module, manager)

        # mark parameters as memory managed
        for param in module.parameters(recurse=False):
            param._is_memory_managed = True


class LinearLayerMemoryManager(BaseLayerMemoryManager):
    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        super().__init__(module, manager)

        # 1) Move params to CPU + pin memory for fast H2D
        _move_params_to_cpu_and_pin(self.module)

        # 2) Hijack forward
        if hasattr(self.module, "ara_lora_ref"):
            # ARA, we need to replace the lora forward
            self._original_forward = getattr(self.module.ara_lora_ref(), "org_forward")
        else:
            self._original_forward = getattr(self.module, "forward")

        def _mm_forward(x, *args, **kwargs):
            # ensure we only use expected signature (Linear: x)
            if args or kwargs:
                # fall back to original if a custom signature is used
                return self._original_forward(x, *args, **kwargs)

            weight_cpu = self.module.weight
            bias_cpu = getattr(self.module, "bias", None)
            weight_scale_cpu = getattr(self.module, "weight_scale", None)
            device = self.manager.process_device
            logical_shape = _linear_logical_shape(self.module)

            # NOTE: do NOT move params to device here; autograd fn streams & bounces them
            return _BouncingLinearFn.apply(
                x, weight_cpu, bias_cpu, weight_scale_cpu, device, logical_shape
            )

        if hasattr(self.module, "ara_lora_ref"):
            self.module.ara_lora_ref().org_forward = _mm_forward
        else:
            self.module.forward = _mm_forward
        
        self.module._memory_management_device = self.manager.process_device


class ConvLayerMemoryManager(BaseLayerMemoryManager):
    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        super().__init__(module, manager)

        # 1) Move params to CPU + pin memory for fast H2D
        _move_params_to_cpu_and_pin(self.module)

        # Cache static conv attributes from the module
        stride = (
            self.module.stride
            if isinstance(self.module.stride, tuple)
            else (self.module.stride, self.module.stride)
        )
        padding = (
            self.module.padding
            if isinstance(self.module.padding, tuple)
            else (self.module.padding, self.module.padding)
        )
        dilation = (
            self.module.dilation
            if isinstance(self.module.dilation, tuple)
            else (self.module.dilation, self.module.dilation)
        )
        groups = self.module.groups

        # 2) Hijack forward
        if hasattr(self.module, "ara_lora_ref"):
            # ARA, we need to replace the lora forward
            self._original_forward = getattr(self.module.ara_lora_ref(), "org_forward")
        else:
            self._original_forward = getattr(self.module, "forward")

        def _mm_forward(x, *args, **kwargs):
            # Support the typical Conv2d(x) call; if user passes uncommon extras, fallback.
            if args or kwargs:
                return self._original_forward(x, *args, **kwargs)

            weight_cpu = self.module.weight
            bias_cpu = getattr(self.module, "bias", None)
            device = self.manager.process_device

            return _BouncingConv2dFn.apply(
                x, weight_cpu, bias_cpu, device, stride, padding, dilation, groups
            )

        if hasattr(self.module, "ara_lora_ref"):
            self.module.ara_lora_ref().org_forward = _mm_forward
        else:
            self.module.forward = _mm_forward
        
        self.module._memory_management_device = self.manager.process_device
