"""Portable FP8 storage for Ming's frozen, block-offloaded transformer.

The public toolkit FP8 importer supplies the same per-tensor quantization
math. This scoped backend stores FP8 payloads as bytes so generic dtype casts
cannot silently widen every frozen weight. OstrisLinear recomputes weights
for input gradients instead of retaining dense weights across backward.
"""

import torch

from toolkit.util.ostris_quant import (
    OstrisBackendMetadata, OstrisLinear, OstrisQuantizer,
    get_ostris_backend_metadata, register_ostris_backend,
)


class MingFP8Quantizer(OstrisQuantizer):
    maximum = torch.finfo(torch.float8_e4m3fn).max

    def quantize_(self, module: torch.nn.Linear, weight_fp32: torch.Tensor) -> None:
        scale = (weight_fp32.abs().max() / self.maximum).clamp(min=1e-12)
        data = (weight_fp32 / scale).clamp(-self.maximum, self.maximum).to(torch.float8_e4m3fn)
        module.register_buffer("ming_fp8_data", data.contiguous().view(torch.uint8), persistent=False)
        module.register_buffer("ming_fp8_scale", scale.detach().float().reshape(1).view(torch.uint8).clone(), persistent=False)

    def dequantize(self, module: OstrisLinear) -> torch.Tensor:
        data = module.ming_fp8_data.view(torch.float8_e4m3fn)
        scale = module.ming_fp8_scale.view(torch.float32)[0]
        return data.float() * scale

    def dequantize_to(self, module: OstrisLinear, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Move compressed data first. Dense compatibility exports requested on
        # CPU must not transiently materialize a full GPU matrix.
        data = module.ming_fp8_data.to(device).view(torch.float8_e4m3fn)
        scale = module.ming_fp8_scale.to(device).view(torch.float32)[0]
        return (data.float() * scale).to(dtype)

    @torch.no_grad()
    def requantize_(self, module: OstrisLinear, fp_weight: torch.Tensor) -> None:
        weight = fp_weight.float()
        scale = (weight.abs().max() / self.maximum).clamp(min=1e-12)
        data = (weight / scale).clamp(-self.maximum, self.maximum).to(torch.float8_e4m3fn)
        module.ming_fp8_data.copy_(data.contiguous().view(torch.uint8))
        module.ming_fp8_scale.copy_(scale.reshape(1).view(torch.uint8))


def register_ming_fp8() -> None:
    if get_ostris_backend_metadata("ming_fp8") is not None:
        return
    register_ostris_backend(
        OstrisBackendMetadata(name="ming_fp8", format_version=1, bits=8, status="stable",
            capabilities=("frozen_weight", "activation_backward", "packed_cache",
                          "compressed_offload", "portable_state_dict", "torch_fallback"),
            supported_devices=("cpu", "cuda"), shape_notes="nn.Linear with per-tensor E4M3FN scaling",
            device_notes="Byte-backed frozen FP8 weights with BF16/FP32 dequantized matmul; no custom kernels"),
        lambda options: MingFP8Quantizer(),
    )


register_ming_fp8()
