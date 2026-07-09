"""Pluggable weight-only quantized linear layers.

``OstrisLinear`` is created in place from an existing ``nn.Linear`` so references
held by LoRA wrappers and parent modules remain valid. Quantizer backends own their
registered buffers and reconstruct ordinary floating-point weights for saving.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class OstrisQuantizer:
    """Base class for weight quantization backends used by ``OstrisLinear``."""

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        return True

    def quantize_(self, module: torch.nn.Linear, weight_fp32: torch.Tensor) -> None:
        raise NotImplementedError

    def dequantize(self, module: "OstrisLinear") -> torch.Tensor:
        """Reconstruct the full weight in the original basis as float32."""
        raise NotImplementedError

    def requantize_(self, module: "OstrisLinear", fp_weight: torch.Tensor) -> None:
        raise NotImplementedError

    def forward(self, module: "OstrisLinear", x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight = self.dequantize(module).to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, module.bias)

    def backward_input(
        self,
        module: "OstrisLinear",
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute the frozen weight for the activation gradient.

        Keeping this separate from ``forward`` prevents autograd from retaining a
        full dequantized weight for every layer until backward.
        """
        with torch.no_grad():
            weight = self.dequantize(module).to(
                device=grad_output.device,
                dtype=grad_output.dtype,
            )
        return F.linear(grad_output, weight.transpose(0, 1))


class _FrozenOstrisLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, module: "OstrisLinear") -> torch.Tensor:
        ctx.module = module
        return module.ostris_quantizer.forward(module, x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        module = ctx.module
        grad_input = module.ostris_quantizer.backward_input(module, grad_output)
        return grad_input, None


class OstrisLinear(torch.nn.Linear):
    """A frozen linear layer backed by a pluggable quantized representation."""

    is_ostris_quantized = True

    @property
    def logical_weight_numel(self) -> int:
        return int(self.out_features) * int(self.in_features)

    @property
    def quantized_device(self) -> torch.device:
        for buffer in self._buffers.values():
            if isinstance(buffer, torch.Tensor):
                return buffer.device
        if self.bias is not None:
            return self.bias.device
        return torch.device("cpu")

    @torch.no_grad()
    def dequantize_weight(self) -> torch.Tensor:
        return self.ostris_quantizer.dequantize(self).to(self.ostris_orig_dtype)

    @property
    def weight(self) -> torch.Tensor:
        # Compatibility view for code that inspects or merges a Linear weight.
        weight = self.dequantize_weight()
        weight._is_ostris_weight = True
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and x.requires_grad:
            return _FrozenOstrisLinearFunction.apply(x, self)
        return self.ostris_quantizer.forward(self, x)

    @torch.no_grad()
    def requantize_(self, fp_weight: torch.Tensor) -> None:
        self.ostris_quantizer.requantize_(self, fp_weight)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        weight = self.dequantize_weight()
        destination[prefix + "weight"] = weight if keep_vars else weight.detach()
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()


def get_ostris_quantizer(qtype: str) -> Optional[OstrisQuantizer]:
    """Resolve a custom qtype, returning ``None`` for other quantizers."""
    from toolkit.util.orbit_quant import ORBIT_QTYPES, OrbitQuantizer
    from toolkit.util.orbit_vq_quant import ORBIT_VQ_QTYPES, OrbitVQQuantizer

    if qtype in ORBIT_QTYPES:
        return OrbitQuantizer(ORBIT_QTYPES[qtype])
    if qtype in ORBIT_VQ_QTYPES:
        return OrbitVQQuantizer(**ORBIT_VQ_QTYPES[qtype])
    return None


@torch.no_grad()
def convert_linear_to_ostris(
    module: torch.nn.Linear,
    quantizer: OstrisQuantizer,
) -> bool:
    """Quantize an ``nn.Linear`` in place while preserving object identity."""
    if isinstance(module, OstrisLinear):
        return True

    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or not weight.dtype.is_floating_point:
        return False
    if type(weight.data) is not torch.Tensor:
        # A tensor-subclass quantizer such as torchao already owns this weight.
        return False
    if not quantizer.can_quantize(module):
        return False

    quantizer.quantize_(module, weight.detach().to(torch.float32))
    module.ostris_quantizer = quantizer
    module.ostris_orig_dtype = weight.dtype
    del module._parameters["weight"]
    if module.bias is not None:
        module.bias.requires_grad_(False)
    module.__class__ = OstrisLinear
    return True
