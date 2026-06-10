# Based on current LyCORIS LoKr, adapted for ai-toolkit's network lifecycle.

import math
from typing import TYPE_CHECKING, Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from optimum.quanto import QBytesTensor, QTensor
from torchao.dtypes import AffineQuantizedTensor

from toolkit.network_mixins import ToolkitModuleMixin, is_mergeable_lora_target

if TYPE_CHECKING:
    from toolkit.lora_special import LoRASpecialNetwork


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_kernel_tuple(kernel_size):
    if isinstance(kernel_size, int):
        return (kernel_size,)
    return tuple(kernel_size)


def _prod(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _is_floating_dtype(dtype):
    return isinstance(dtype, torch.dtype) and torch.empty((), dtype=dtype).is_floating_point()


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    Return upstream AI Toolkit LoKr factors.

    Positive factors keep the requested factor on the left side, even when it
    is larger than the paired factor. LoKr uses Kronecker products, so that
    order is part of the adapter layout.
    """
    if factor > 0 and (dimension % factor) == 0:
        return factor, dimension // factor
    return balanced_factorization(dimension, factor)


def balanced_factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    Return Revamped's balanced LoKr factors, keeping the smaller value first.
    """
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def legacy_factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    return factorization(dimension, factor)


def rebuild_tucker(t, wa, wb):
    return torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)


def make_weight_cp(t, wa, wb):
    return rebuild_tucker(t, wa, wb)


def make_kron(w1, w2, scale):
    for _ in range(w2.dim() - w1.dim()):
        w1 = w1.unsqueeze(-1)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    if scale != 1:
        rebuild = rebuild * scale
    return rebuild


class LokrModule(ToolkitModuleMixin, nn.Module):
    _warned_normal_dropout = False

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        use_cp=False,
        use_tucker=None,
        use_scalar=False,
        decompose_both=False,
        network: "LoRASpecialNetwork" = None,
        factor: int = -1,
        rank_dropout_scale=False,
        weight_decompose=False,
        wd_on_out=True,
        full_matrix=False,
        bypass_mode=False,
        rs_lora=False,
        unbalanced_factorization=False,
        legacy_factorization=True,
        **kwargs,
    ):
        ToolkitModuleMixin.__init__(self, network=network)
        torch.nn.Module.__init__(self)

        if use_tucker is None:
            use_tucker = use_cp

        factor = int(factor)
        self.lora_name = lora_name
        self.lora_dim = int(lora_dim)
        self.tucker = False
        self.cp = False
        self.use_w1 = False
        self.use_w2 = False
        self.full_matrix = _as_bool(full_matrix)
        self.rs_lora = _as_bool(rs_lora)
        self.rank_dropout_scale = _as_bool(rank_dropout_scale)
        self.bypass_mode = _as_bool(bypass_mode)
        self.legacy_factorization = _as_bool(legacy_factorization)
        self.unbalanced_factorization = _as_bool(unbalanced_factorization)
        self.org_module = [org_module]
        self.can_merge_in = is_mergeable_lora_target(org_module)

        self.module_type, self.op, self.extra_args = self._get_module_ops(org_module)
        self.shape = self._get_weight_shape(org_module, self.module_type)
        if self.module_type == "linear":
            out_dim, in_dim = self.shape
            kernel_size = ()
        elif self.module_type.startswith("conv"):
            out_dim, in_dim = self.shape[:2]
            kernel_size = _as_kernel_tuple(self.shape[2:])
        else:
            raise ValueError(f"{org_module.__class__.__name__} is not supported in LoKr.")

        factorize = factorization if self.legacy_factorization else balanced_factorization
        in_m, in_n = factorize(in_dim, factor)
        out_l, out_k = factorize(out_dim, factor)
        if self.unbalanced_factorization:
            out_l, out_k = out_k, out_l
        self.lokr_shape = ((out_l, out_k), (in_m, in_n), *kernel_size)

        self.tucker = (
            _as_bool(use_tucker)
            and self.module_type.startswith("conv")
            and any(i != 1 for i in kernel_size)
        )
        self.cp = self.tucker
        self._make_weights(_as_bool(decompose_both))

        self.dropout = dropout
        if dropout:
            if not LokrModule._warned_normal_dropout:
                print("[WARN]LoKr does not support normal dropout yet; ignoring dropout.")
                LokrModule._warned_normal_dropout = True
            self.dropout = None
        self.rank_dropout = float(rank_dropout or 0.0)
        self.module_dropout = float(module_dropout or 0.0)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().item()
        alpha = self.lora_dim if alpha is None or alpha == 0 else float(alpha)
        if self.use_w2 and self.use_w1:
            alpha = self.lora_dim

        r_factor = math.sqrt(self.lora_dim) if self.rs_lora else self.lora_dim
        self.scale = alpha / r_factor
        self.register_buffer("alpha", torch.tensor(alpha * (self.lora_dim / r_factor)))

        if _as_bool(use_scalar):
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        self.wd = _as_bool(weight_decompose)
        self.wd_on_out = _as_bool(wd_on_out)
        if self.wd:
            self._init_weight_decompose(org_module)

        self._init_weights(_as_bool(use_scalar))

        self.multiplier: Union[float, List[float]] = multiplier
        self.register_load_state_dict_post_hook(self.load_weight_hook)

        weight = self.get_weight(self.shape)
        assert torch.sum(torch.isnan(weight)) == 0, "weight is nan"

    @staticmethod
    def _get_module_ops(org_module):
        module_name = org_module.__class__.__name__
        if hasattr(org_module, "in_features") and hasattr(org_module, "out_features"):
            return "linear", F.linear, {}
        if all(hasattr(org_module, attr) for attr in ("in_channels", "out_channels", "kernel_size")):
            dim = len(tuple(org_module.weight.shape)) - 2
            op = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}.get(dim)
            if op is None:
                raise ValueError(f"{module_name} has unsupported convolution dimension {dim}.")
            return f"conv{dim}d", op, {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        return "unknown", None, {}

    @staticmethod
    def _get_weight_shape(org_module, module_type):
        if module_type == "linear":
            logical_shape = (int(org_module.out_features), int(org_module.in_features))
            storage_shape = tuple(org_module.weight.shape)
            quantized_linear = org_module.__class__.__name__ in {"Fp8Linear", "Linear4bit", "Linear8bitLt", "QLinear"}
            if quantized_linear or _prod(logical_shape) == _prod(storage_shape):
                return logical_shape
            return storage_shape
        return tuple(org_module.weight.shape)

    def _make_weights(self, decompose_both: bool):
        shape = self.lokr_shape
        if decompose_both and self.lora_dim < max(shape[0][0], shape[1][0]) / 2 and not self.full_matrix:
            self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], self.lora_dim))
            self.lokr_w1_b = nn.Parameter(torch.empty(self.lora_dim, shape[1][0]))
        else:
            self.use_w1 = True
            self.lokr_w1 = nn.Parameter(torch.empty(shape[0][0], shape[1][0]))

        if self.module_type.startswith("conv"):
            if self.lora_dim >= max(shape[0][1], shape[1][1]) / 2 or self.full_matrix:
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(torch.empty(shape[0][1], shape[1][1], *shape[2:]))
            elif self.tucker:
                self.lokr_t2 = nn.Parameter(torch.empty(self.lora_dim, self.lora_dim, *shape[2:]))
                self.lokr_w2_a = nn.Parameter(torch.empty(self.lora_dim, shape[0][1]))
                self.lokr_w2_b = nn.Parameter(torch.empty(self.lora_dim, shape[1][1]))
            else:
                self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], self.lora_dim))
                self.lokr_w2_b = nn.Parameter(torch.empty(self.lora_dim, shape[1][1] * _prod(shape[2:])))
        else:
            if self.lora_dim < max(shape[0][1], shape[1][1]) / 2 and not self.full_matrix:
                self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], self.lora_dim))
                self.lokr_w2_b = nn.Parameter(torch.empty(self.lora_dim, shape[1][1]))
            else:
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(torch.empty(shape[0][1], shape[1][1]))

    def _init_weights(self, use_scalar: bool):
        if self.use_w2:
            if use_scalar:
                torch.nn.init.kaiming_uniform_(self.lokr_w2, a=math.sqrt(5))
            else:
                torch.nn.init.constant_(self.lokr_w2, 0)
        else:
            if self.tucker:
                torch.nn.init.kaiming_uniform_(self.lokr_t2, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            if use_scalar:
                torch.nn.init.kaiming_uniform_(self.lokr_w2_b, a=math.sqrt(5))
            else:
                torch.nn.init.constant_(self.lokr_w2_b, 0)

        if self.use_w1:
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))

    def _init_weight_decompose(self, org_module):
        org_weight = self.get_orig_weight(org_module.weight.device).detach().cpu().float()
        self.dora_norm_dims = org_weight.dim() - 1
        if self.wd_on_out:
            dora_scale = torch.norm(
                org_weight.reshape(org_weight.shape[0], -1),
                dim=1,
                keepdim=True,
            ).reshape(org_weight.shape[0], *[1] * self.dora_norm_dims)
        else:
            dora_scale = (
                torch.norm(
                    org_weight.transpose(1, 0).reshape(org_weight.shape[1], -1),
                    dim=1,
                    keepdim=True,
                )
                .reshape(org_weight.shape[1], *[1] * self.dora_norm_dims)
                .transpose(1, 0)
            )
        self.dora_scale = nn.Parameter(dora_scale.float())

    @staticmethod
    def _dequantize_tensor(tensor):
        if isinstance(tensor, (QTensor, QBytesTensor, AffineQuantizedTensor)):
            return tensor.dequantize().data
        dequantize = getattr(tensor, "dequantize", None)
        is_bnb_quantized = tensor.__class__.__name__ == "Params4bit" or getattr(tensor, "quant_state", None) is not None
        is_torch_quantized = isinstance(tensor, torch.Tensor) and getattr(tensor, "is_quantized", False)
        if callable(dequantize) and (is_bnb_quantized or is_torch_quantized):
            return dequantize().data
        return tensor.data if hasattr(tensor, "data") else tensor

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def _w1(self):
        return self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b

    def _w2(self):
        if self.use_w2:
            return self.lokr_w2
        if self.tucker:
            return rebuild_tucker(self.lokr_t2, self.lokr_w2_a, self.lokr_w2_b)
        w2 = self.lokr_w2_a @ self.lokr_w2_b
        if self.module_type.startswith("conv"):
            shape = self.lokr_shape
            return w2.reshape(shape[0][1], shape[1][1], *shape[2:])
        return w2

    def get_weight(self, orig_weight=None):
        weight = make_kron(self._w1(), self._w2(), self.scale)
        if orig_weight is not None:
            shape = tuple(orig_weight.shape) if hasattr(orig_weight, "shape") else tuple(orig_weight)
            weight = weight.reshape(shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0), device=weight.device) > self.rank_dropout).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                keep = drop.mean()
                if keep > 0:
                    drop = drop / keep
            weight = weight * drop
        return weight

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        diff = self.get_weight(shape)
        diff = diff * self.scalar.to(diff.device, dtype=diff.dtype)
        if multiplier != 1:
            diff = diff * multiplier
        if device is not None:
            diff = diff.to(device)
        return diff, None

    def apply_weight_decompose(self, weight, multiplier=1):
        weight = weight.to(self.dora_scale.dtype)
        if self.wd_on_out:
            weight_norm = (
                weight.reshape(weight.shape[0], -1)
                .norm(dim=1)
                .reshape(weight.shape[0], *[1] * self.dora_norm_dims)
            ) + torch.finfo(weight.dtype).eps
        else:
            weight_norm = (
                weight.transpose(0, 1)
                .reshape(weight.shape[1], -1)
                .norm(dim=1, keepdim=True)
                .reshape(weight.shape[1], *[1] * self.dora_norm_dims)
                .transpose(0, 1)
            ) + torch.finfo(weight.dtype).eps

        scale = self.dora_scale.to(weight.device) / weight_norm
        if multiplier != 1:
            scale = multiplier * (scale - 1) + 1
        return weight * scale

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        base_weight = self.get_orig_weight(device or self.org_module[0].weight.device)
        diff_weight = self.get_weight(shape or base_weight.shape).to(base_weight.device, dtype=base_weight.dtype)
        diff_weight = diff_weight * self.scalar.to(diff_weight.device, dtype=diff_weight.dtype)
        if self.wd:
            merged = self.apply_weight_decompose(base_weight + diff_weight, multiplier)
        else:
            merged = base_weight + diff_weight * multiplier
        return merged, None

    @torch.no_grad()
    def merge_in(self, merge_weight=1.0):
        if not self.can_merge_in:
            return
        if not is_mergeable_lora_target(self.org_module[0]):
            self.can_merge_in = False
            return

        org_sd = self.org_module[0].state_dict()
        weight_key = "weight"
        if weight_key not in org_sd:
            self.can_merge_in = False
            return

        orig_dtype = org_sd[weight_key].dtype
        weight = org_sd[weight_key].float()
        lokr_weight = self.get_weight(weight).to(weight.device, dtype=weight.dtype)
        lokr_weight = lokr_weight * self.scalar.to(weight.device, dtype=weight.dtype)

        if self.wd:
            merged_weight = self.apply_weight_decompose(weight + lokr_weight, merge_weight)
        else:
            merged_weight = weight + lokr_weight * merge_weight
        if merged_weight.shape != weight.shape:
            self.can_merge_in = False
            return

        org_sd[weight_key] = merged_weight.to(orig_dtype)
        self.org_module[0].load_state_dict(org_sd)

    def get_orig_weight(self, device):
        module = self.org_module[0]
        weight = module.weight
        if weight.device != device:
            weight = weight.to(device)
        weight = self._dequantize_tensor(weight).detach()
        if tuple(weight.shape) != self.shape and weight.numel() == _prod(self.shape):
            weight = weight.reshape(self.shape)
        weight_scale = getattr(module, "weight_scale", None)
        if self.module_type == "linear" and weight_scale is not None and weight.shape == self.shape:
            compute_dtype = self._compute_dtype(weight)
            weight = weight.to(dtype=compute_dtype)
            scale = weight_scale.to(weight.device, dtype=compute_dtype).view(-1, *[1] * (weight.dim() - 1))
            weight = weight * scale
        return weight

    def get_orig_bias(self, device):
        if hasattr(self.org_module[0], "bias") and self.org_module[0].bias is not None:
            bias = self.org_module[0].bias
            if bias.device != device:
                bias = bias.to(device)
            return self._dequantize_tensor(bias).detach()
        return None

    def _compute_dtype(self, *preferred_tensors):
        module = self.org_module[0]
        compute_dtype = getattr(module, "compute_dtype", None)
        if _is_floating_dtype(compute_dtype):
            return compute_dtype
        for tensor in preferred_tensors:
            dtype = getattr(tensor, "dtype", None)
            if _is_floating_dtype(dtype):
                return dtype
        for tensor in (self._w1(), self._w2()):
            dtype = getattr(tensor, "dtype", None)
            if _is_floating_dtype(dtype):
                return dtype
        return torch.float32

    def _call_forward(self, x, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1, device=x.device) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)

        multiplier = self.network_ref().torch_multiplier
        multiplier = torch.mean(multiplier)

        if self.bypass_mode:
            return self.bypass_forward(x, multiplier, *args, **kwargs)

        base = self.org_forward(x, *args, **kwargs)
        if isinstance(base, (QTensor, QBytesTensor)):
            base = base.dequantize()

        lora_x = x.dequantize() if isinstance(x, (QTensor, QBytesTensor)) else x
        compute_dtype = self._compute_dtype(base, lora_x)
        lokr_weight = self.get_weight(self.shape).to(lora_x.device, dtype=compute_dtype)
        lokr_weight = lokr_weight * self.scalar.to(lokr_weight.device, dtype=lokr_weight.dtype)

        if lora_x.dtype != compute_dtype:
            lora_x = lora_x.to(dtype=compute_dtype)

        if self.wd:
            orig_weight = self.get_orig_weight(lora_x.device).to(dtype=compute_dtype)
            weight = self.apply_weight_decompose(orig_weight + lokr_weight, multiplier)
            bias = self.get_orig_bias(lora_x.device)
            if bias is not None:
                bias = bias.to(weight.device, dtype=weight.dtype)
            output = self.op(lora_x, weight.view(self.shape), bias, **self.extra_args)
            if _is_floating_dtype(base.dtype):
                output = output.to(base.dtype)
            return output
        else:
            if isinstance(multiplier, torch.Tensor):
                multiplier = multiplier.to(lokr_weight.device, dtype=lokr_weight.dtype)
            delta = lokr_weight * multiplier
            output = self.op(lora_x, delta.view(self.shape), None, **self.extra_args)

        if _is_floating_dtype(base.dtype):
            output = output.to(base.dtype)
        return base + output

    def _bypass_w2(self):
        if self.use_w2:
            return self.lokr_w2

        a = self.lokr_w2_b
        b = self.lokr_w2_a
        kernel_dims = len(self.shape) - 2
        if self.tucker:
            t = self.lokr_t2
            a = a.view(*a.shape, *[1] * (len(t.shape) - 2))
            b = b.view(*b.shape, *[1] * (len(t.shape) - 2))
            return a, b, t
        if self.module_type.startswith("conv"):
            shape = self.lokr_shape
            a = a.view(a.shape[0], shape[1][1], *shape[2:])
            b = b.view(*b.shape, *[1] * kernel_dims)
        return a, b, None

    def bypass_forward_diff(self, h, scale=1):
        is_conv = self.module_type.startswith("conv")
        if self.use_w2:
            ba = self.lokr_w2
            a = b = t = None
        else:
            a, b, t = self._bypass_w2()
            ba = None

        if ba is not None:
            ba = ba.to(h.device, dtype=h.dtype)
        if a is not None:
            a = a.to(h.device, dtype=h.dtype)
        if b is not None:
            b = b.to(h.device, dtype=h.dtype)
        if t is not None:
            t = t.to(h.device, dtype=h.dtype)

        c = self._w1().to(h.device, dtype=h.dtype)
        uq = c.size(1)

        if is_conv:
            batch, _, *rest = h.shape
            h_in_group = h.reshape(batch * uq, -1, *rest)
        else:
            h_in_group = h.reshape(*h.shape[:-1], uq, -1)

        if self.use_w2:
            hb = self.op(h_in_group, ba, **self.extra_args)
        elif is_conv:
            if t is not None:
                ha = self.op(h_in_group, a)
                ht = self.op(ha, t, **self.extra_args)
                hb = self.op(ht, b)
            else:
                ha = self.op(h_in_group, a, **self.extra_args)
                hb = self.op(ha, b)
        else:
            ha = self.op(h_in_group, a, **self.extra_args)
            hb = self.op(ha, b)

        if is_conv:
            hb = hb.view(batch, -1, *hb.shape[1:])
            h_cross_group = hb.transpose(1, -1)
        else:
            h_cross_group = hb.transpose(-1, -2)

        hc = F.linear(h_cross_group, c)
        if is_conv:
            hc = hc.transpose(1, -1)
            h = hc.reshape(batch, -1, *hc.shape[3:])
        else:
            hc = hc.transpose(-1, -2)
            h = hc.reshape(*hc.shape[:-2], -1)

        scalar = self.scalar.to(h.device, dtype=h.dtype)
        return h * self.scale * scale * scalar

    def bypass_forward(self, x, scale=1, *args, **kwargs):
        base = self.org_forward(x, *args, **kwargs)
        if isinstance(base, (QTensor, QBytesTensor)):
            base = base.dequantize()
        lora_x = x.dequantize() if isinstance(x, (QTensor, QBytesTensor)) else x
        compute_dtype = self._compute_dtype(base, lora_x)
        if lora_x.dtype != compute_dtype:
            lora_x = lora_x.to(dtype=compute_dtype)
        diff = self.bypass_forward_diff(lora_x, scale=scale)
        if _is_floating_dtype(base.dtype):
            diff = diff.to(base.dtype)
        return base + diff

    def load_weight_hook(self, module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        for key in list(missing_keys):
            if key.endswith("scalar"):
                missing_keys.remove(key)
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.copy_(torch.ones_like(self.scalar))
        elif getattr(self, "scalar", None) is not None:
            self.scalar.copy_(torch.ones_like(self.scalar))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        def save_tensor(name, tensor):
            destination[prefix + name] = tensor if keep_vars else tensor.detach()

        save_tensor("alpha", self.alpha)
        if self.wd:
            save_tensor("dora_scale", self.dora_scale)

        scalar = self.scalar
        if self.use_w1:
            save_tensor("lokr_w1", self.lokr_w1 * scalar)
        else:
            save_tensor("lokr_w1_a", self.lokr_w1_a * scalar)
            save_tensor("lokr_w1_b", self.lokr_w1_b)

        if self.use_w2:
            save_tensor("lokr_w2", self.lokr_w2)
        else:
            save_tensor("lokr_w2_a", self.lokr_w2_a)
            save_tensor("lokr_w2_b", self.lokr_w2_b)
            if self.tucker:
                save_tensor("lokr_t2", self.lokr_t2)

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.get_weight(self.shape).norm()
        norm = torch.clamp(orig_norm, min=max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            modules = 4 - int(self.use_w1) - int(self.use_w2) + int(not self.use_w2 and self.tucker)
            if self.use_w1:
                self.lokr_w1 *= ratio ** (1 / modules)
            else:
                self.lokr_w1_a *= ratio ** (1 / modules)
                self.lokr_w1_b *= ratio ** (1 / modules)

            if self.use_w2:
                self.lokr_w2 *= ratio ** (1 / modules)
            else:
                if self.tucker:
                    self.lokr_t2 *= ratio ** (1 / modules)
                self.lokr_w2_a *= ratio ** (1 / modules)
                self.lokr_w2_b *= ratio ** (1 / modules)

        return scaled, orig_norm * ratio
