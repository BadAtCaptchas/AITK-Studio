#based off https://github.com/catid/dora/blob/main/dora.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Union, List

from optimum.quanto import QBytesTensor, QTensor

from toolkit.network_mixins import ToolkitModuleMixin, ExtractableModuleMixin
from toolkit.util.orbit_quant import rpbh_forward

if TYPE_CHECKING:
    from toolkit.lora_special import LoRASpecialNetwork

# diffusers specific stuff
LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear',
    'Linear4bit',
    'Linear8bitLt',
    'Fp8Linear',
    'Nvfp4Linear',
    'OstrisLinear',
    # 'GroupNorm',
]
CONV_MODULES = [
    'Conv2d',
    'LoRACompatibleConv'
]

def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T

class DoRAModule(ToolkitModuleMixin, ExtractableModuleMixin, torch.nn.Module):
    # def __init__(self, d_in, d_out, rank=4, weight=None, bias=None):
    def __init__(
            self,
            lora_name,
            org_module: torch.nn.Module,
            multiplier=1.0,
            lora_dim=4,
            alpha=1,
            dropout=None,
            rank_dropout=None,
            module_dropout=None,
            network: 'LoRASpecialNetwork' = None,
            use_bias: bool = False,
            **kwargs
    ):
        self.can_merge_in = False
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        ToolkitModuleMixin.__init__(self, network=network)
        torch.nn.Module.__init__(self)
        self.lora_name = lora_name

        self.lora_dim = lora_dim

        if org_module.__class__.__name__ in CONV_MODULES:
            raise NotImplementedError("Convolutional layers are not supported yet")

        if type(alpha) == torch.Tensor:
            alpha = float(alpha.detach().float().item())
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        scale = float(alpha) / self.lora_dim
        # self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える eng: treat as constant

        self.multiplier: Union[float, List[float]] = multiplier
        # wrap the original module so it doesn't get weights updated
        self.org_module = [org_module]
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.is_checkpointing = False

        d_out = org_module.out_features
        d_in = org_module.in_features

        std_dev = 1 / torch.sqrt(torch.tensor(self.lora_dim).float())
        # self.lora_up = nn.Parameter(torch.randn(d_out, self.lora_dim) * std_dev)  # lora_A
        # self.lora_down = nn.Parameter(torch.zeros(self.lora_dim, d_in))  # lora_B
        self.lora_up = nn.Linear(self.lora_dim, d_out, bias=False)  # lora_B
        # self.lora_up.weight.data = torch.randn_like(self.lora_up.weight.data) * std_dev
        self.lora_up.weight.data = torch.zeros_like(self.lora_up.weight.data)
        # self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        # self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        self.lora_down = nn.Linear(d_in, self.lora_dim, bias=False)  # lora_A
        # self.lora_down.weight.data = torch.zeros_like(self.lora_down.weight.data)
        self.lora_down.weight.data = torch.randn_like(self.lora_down.weight.data) * std_dev

        self._set_runtime_scale(scale)

        # Compute magnitude through a factorized path. Accessing
        # ``OstrisLinear.weight`` would reconstruct the complete logical weight.
        weight_norm = self._get_weight_norm_from_factors(1.0)
        self.magnitude = nn.Parameter(weight_norm.detach().clone(), requires_grad=True)

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward
        # del self.org_module

    def get_orig_weight(self):
        weight = self.org_module[0].weight
        if isinstance(weight, QTensor) or isinstance(weight, QBytesTensor):
            return weight.dequantize().data.detach()
        else:
            return weight.data.detach()

    def get_orig_bias(self):
        if hasattr(self.org_module[0], 'bias') and self.org_module[0].bias is not None:
            return self.org_module[0].bias.data.detach()
        return None

    # def dora_forward(self, x, *args, **kwargs):
    #     lora = torch.matmul(self.lora_A, self.lora_B)
    #     adapted = self.get_orig_weight() + lora
    #     column_norm = adapted.norm(p=2, dim=0, keepdim=True)
    #     norm_adapted = adapted / column_norm
    #     calc_weights = self.magnitude * norm_adapted
    #     return F.linear(x, calc_weights, self.get_orig_bias())

    def _get_weight_norm(self, weight, scaled_lora_weight) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = weight + scaled_lora_weight.to(weight.device)
        weight_norm = torch.linalg.norm(weight, dim=1)
        return weight_norm

    def apply_dora(self, x, scaled_lora_weight):
        # ref https://github.com/huggingface/peft/blob/1e6d1d73a0850223b0916052fd8d2382a90eae5a/src/peft/tuners/lora/layer.py#L192
        # lora weight is already scaled

        # magnitude = self.lora_magnitude_vector[active_adapter]
        weight = self.get_orig_weight()
        weight = weight.to(scaled_lora_weight.device, dtype=scaled_lora_weight.dtype)
        weight_norm = self._get_weight_norm(weight, scaled_lora_weight)
        # see section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        weight_norm = weight_norm.detach()
        dora_weight = transpose(weight + scaled_lora_weight, False)
        return (self.magnitude / weight_norm - 1).view(1, -1) * F.linear(x.to(dora_weight.dtype), dora_weight)

    @staticmethod
    def _row_norms_from_chunks(weight_chunks, down, up):
        """Calculate ``||W + U D||`` without constructing ``U D``.

        ``weight_chunks`` yields base rows in the same orthonormal basis as
        ``down``. The factorized identity keeps every temporary proportional to
        the adapter rank or one bounded base-weight row tile.
        """
        down = down.float()
        gram = down @ down.transpose(0, 1)
        norms = up.new_empty(up.shape[0], dtype=torch.float32)
        for start, end, weight_rows in weight_chunks:
            weight_rows = weight_rows.float()
            up_rows = up[start:end].float()
            cross_projection = weight_rows @ down.transpose(0, 1)
            base_sq = weight_rows.square().sum(dim=1)
            cross = (up_rows * cross_projection).sum(dim=1)
            delta_sq = torch.einsum("ir,rs,is->i", up_rows, gram, up_rows)
            norms[start:end] = (
                base_sq + (2.0 * cross) + delta_sq
            ).clamp_min_(0.0).sqrt_()
        return norms

    def _dense_weight_chunks(self, device, rows_per_chunk=256):
        module = self.org_module[0]
        weight = module.weight
        if isinstance(weight, (QTensor, QBytesTensor)):
            weight = weight.dequantize()
        for start in range(0, module.out_features, rows_per_chunk):
            end = min(start + rows_per_chunk, module.out_features)
            yield start, end, weight[start:end].detach().to(device=device)

    def _orbit_weight_chunks(self, device):
        module = self.org_module[0]
        quantizer = module.ostris_quantizer
        decode_rows = getattr(quantizer, "_decode_rotated_rows", None)
        runtime_rows = getattr(quantizer, "_runtime_rows", None)
        if not callable(decode_rows) or not callable(runtime_rows):
            backend = getattr(module, "ostris_backend_name", type(quantizer).__name__)
            raise ValueError(
                f"DoRA does not support packed backend {backend!r} without a "
                "bounded row decoder. Use orbit4 or a standard LoRA adapter."
            )
        rows_per_chunk = runtime_rows(
            module.out_features,
            module.in_features,
            torch.float32,
        )
        for start in range(0, module.out_features, rows_per_chunk):
            end = min(start + rows_per_chunk, module.out_features)
            yield start, end, decode_rows(module, start, end, torch.float32).to(device)

    @torch.no_grad()
    def _get_weight_norm_from_factors(self, multiplier) -> torch.Tensor:
        module = self.org_module[0]
        device = self.lora_down.weight.device
        down = self.lora_down.weight.detach().float()
        up = self.lora_up.weight.detach().float()
        # Layer offloading can move the adapter factors independently of the
        # parent module buffer. Keep the compiled-forward buffer in place, but
        # perform this out-of-graph norm calculation beside the factors.
        adapter_scale = self._runtime_scale.detach().to(
            device=up.device,
            dtype=torch.float32,
        )
        if isinstance(multiplier, torch.Tensor):
            adapter_scale = adapter_scale * multiplier.detach().to(
                device=adapter_scale.device,
                dtype=torch.float32,
            ).mean()
        else:
            adapter_scale = adapter_scale * float(multiplier)
        up = up * adapter_scale

        if getattr(module, "is_ostris_quantized", False):
            quantizer = module.ostris_quantizer
            if not callable(getattr(quantizer, "_decode_rotated_rows", None)):
                backend = getattr(module, "ostris_backend_name", type(quantizer).__name__)
                raise ValueError(
                    f"DoRA with {backend!r} would require reconstructing a full "
                    "base weight. Use orbit4 or a standard LoRA adapter."
                )
            # Orbit stores W R. Rotate D by the same orthonormal R so all
            # inner products and row norms remain unchanged.
            down = rpbh_forward(
                down,
                module.orbit_perm,
                module.orbit_signs,
                module.orbit_block,
            )
            chunks = self._orbit_weight_chunks(device)
        else:
            chunks = self._dense_weight_chunks(device)
        return self._row_norms_from_chunks(chunks, down, up)

    def compose_dora_output(self, base_output, lora_output, multiplier):
        """Compose DoRA without materializing a dense base or delta weight."""
        # DoRA treats the dynamically recomputed norm as a constant during
        # backward, while the magnitude vector remains trainable.
        weight_norm = self._get_weight_norm_from_factors(multiplier).detach()
        ratio = self.magnitude.to(weight_norm.device, weight_norm.dtype)
        ratio = ratio / weight_norm.clamp_min(torch.finfo(weight_norm.dtype).eps)
        ratio = ratio.to(base_output.device, base_output.dtype)
        ratio = ratio.view(*([1] * (base_output.dim() - 1)), -1)

        bias = self.get_orig_bias()
        if bias is None:
            return (base_output + lora_output) * ratio
        bias = bias.to(base_output.device, base_output.dtype)
        bias = bias.view(*([1] * (base_output.dim() - 1)), -1)
        return (base_output - bias + lora_output) * ratio + bias
