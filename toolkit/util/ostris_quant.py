"""Pluggable weight-only quantized linear layers.

``OstrisLinear`` is created in place from an existing ``nn.Linear`` so references
held by LoRA wrappers and parent modules remain valid. Quantizer backends own their
registered buffers and reconstruct ordinary floating-point weights for saving.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Callable, Dict, Literal, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


OstrisKernel = Literal["auto", "triton", "torch"]


@dataclass(frozen=True)
class OstrisBackendOptions:
    """Runtime options shared by first-party quantization backends."""

    kernel: OstrisKernel = "auto"
    max_workspace_mb: int = 64

    def __post_init__(self) -> None:
        if self.kernel not in ("auto", "triton", "torch"):
            raise ValueError(
                "kernel must be one of 'auto', 'triton', or 'torch', "
                f"got {self.kernel!r}"
            )
        if isinstance(self.max_workspace_mb, bool) or not isinstance(
            self.max_workspace_mb, int
        ):
            raise TypeError("max_workspace_mb must be an integer")
        if self.max_workspace_mb <= 0:
            raise ValueError("max_workspace_mb must be greater than zero")

    def to_dict(self) -> Dict[str, object]:
        return {
            "kernel": self.kernel,
            "max_workspace_mb": self.max_workspace_mb,
        }


@dataclass(frozen=True)
class OstrisBackendMetadata:
    """Serializable description of a first-party quantization backend."""

    name: str
    format_version: int
    bits: int
    status: Literal["stable", "experimental"]
    capabilities: Tuple[str, ...]
    supported_devices: Tuple[str, ...]
    shape_notes: str
    device_notes: str

    @property
    def experimental(self) -> bool:
        return self.status == "experimental"

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "format_version": self.format_version,
            "bits": self.bits,
            "status": self.status,
            "experimental": self.experimental,
            "capabilities": list(self.capabilities),
            "supported_devices": list(self.supported_devices),
            "shape_notes": self.shape_notes,
            "device_notes": self.device_notes,
        }


OstrisBackendFactory = Callable[[OstrisBackendOptions], "OstrisQuantizer"]
_backend_metadata: Dict[str, OstrisBackendMetadata] = {}
_backend_factories: Dict[str, OstrisBackendFactory] = {}
_builtin_backends_registered = False
_backend_lock = RLock()


def register_ostris_backend(
    metadata: OstrisBackendMetadata,
    factory: OstrisBackendFactory,
) -> None:
    """Register a first-party backend without coupling qtype resolution to it."""
    with _backend_lock:
        if metadata.name in _backend_metadata:
            raise ValueError(f"quantization backend {metadata.name!r} is already registered")
        _backend_metadata[metadata.name] = metadata
        _backend_factories[metadata.name] = factory


def _register_builtin_backends() -> None:
    global _builtin_backends_registered
    if _builtin_backends_registered:
        return
    with _backend_lock:
        if _builtin_backends_registered:
            return
        # These imports are intentionally lazy: both backend modules derive from
        # OstrisQuantizer and therefore cannot be imported while this module itself
        # is still being initialized.
        from toolkit.util.orbit_quant import ORBIT_QTYPES, OrbitQuantizer
        from toolkit.util.orbit_vq_quant import ORBIT_VQ_QTYPES, OrbitVQQuantizer

        common_capabilities = (
            "frozen_weight",
            "activation_backward",
            "packed_cache",
            "compressed_offload",
            "portable_state_dict",
        )
        for name, bits in ORBIT_QTYPES.items():
            register_ostris_backend(
                OstrisBackendMetadata(
                    name=name,
                    format_version=1,
                    bits=bits,
                    status="stable" if name == "orbit4" else "experimental",
                    capabilities=common_capabilities
                    + ("torch_fallback", "triton_optional"),
                    supported_devices=("cpu", "cuda"),
                    shape_notes=(
                        "nn.Linear only; the largest power-of-two divisor of "
                        "in_features must be at least 32"
                    ),
                    device_notes=(
                        "Triton is optional on CUDA; the bounded torch kernel is "
                        "used on CPU and as the CUDA fallback"
                    ),
                ),
                lambda options, bits=bits: OrbitQuantizer(
                    bits,
                    kernel=options.kernel,
                    max_workspace_mb=options.max_workspace_mb,
                ),
            )

        for name, config in ORBIT_VQ_QTYPES.items():
            register_ostris_backend(
                OstrisBackendMetadata(
                    name=name,
                    format_version=1,
                    bits=int(config["bits"]),
                    status="experimental",
                    capabilities=common_capabilities + ("torch_fallback",),
                    supported_devices=("cpu", "cuda"),
                    shape_notes=(
                        "nn.Linear only; the largest power-of-two divisor of "
                        "in_features must be at least 32"
                    ),
                    device_notes=(
                        "Experimental torch implementation; no fused Triton kernel"
                    ),
                ),
                lambda options, config=config: OrbitVQQuantizer(**config),
            )

        _builtin_backends_registered = True


def get_ostris_backend_registry() -> Mapping[str, OstrisBackendMetadata]:
    _register_builtin_backends()
    return MappingProxyType(_backend_metadata)


def get_ostris_backend_metadata(qtype: str) -> Optional[OstrisBackendMetadata]:
    _register_builtin_backends()
    return _backend_metadata.get(qtype)


def is_ostris_qtype(qtype: object) -> bool:
    return isinstance(qtype, str) and get_ostris_backend_metadata(qtype) is not None


class OstrisQuantizer:
    """Base class for weight quantization backends used by ``OstrisLinear``."""

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        return True

    def quantize_(self, module: torch.nn.Linear, weight_fp32: torch.Tensor) -> None:
        raise NotImplementedError

    def dequantize(self, module: "OstrisLinear") -> torch.Tensor:
        """Reconstruct the full weight in the original basis as float32."""
        raise NotImplementedError

    def dequantize_to(
        self,
        module: "OstrisLinear",
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reconstruct a compatibility weight on the requested device.

        Backends with a bounded decoder should override this method so exporting
        a CUDA-resident packed model does not first allocate a logical-sized
        dense weight on CUDA. The default keeps third-party/experimental backend
        compatibility.
        """
        return self.dequantize(module).to(device=device, dtype=dtype)

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
        ctx.input_dtype = x.dtype
        return module.ostris_quantizer.forward(module, x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        module = ctx.module
        grad_input = module.ostris_quantizer.backward_input(module, grad_output)
        if grad_input.dtype != ctx.input_dtype:
            grad_input = grad_input.to(ctx.input_dtype)
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
    def dequantize_weight(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        target_device = self.quantized_device if device is None else torch.device(device)
        target_dtype = self.ostris_orig_dtype if dtype is None else dtype
        return self.ostris_quantizer.dequantize_to(
            self,
            target_device,
            target_dtype,
        )

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
        # Portable checkpoints intentionally contain an ordinary floating-point
        # weight. Reconstruct it directly onto CPU so saving a CUDA model never
        # retains dense compatibility weights in scarce device memory.
        weight = self.dequantize_weight(device=torch.device("cpu"))
        destination[prefix + "weight"] = weight if keep_vars else weight.detach()
        if self.bias is not None:
            bias = self.bias.to("cpu")
            destination[prefix + "bias"] = bias if keep_vars else bias.detach()


def get_ostris_quantizer(
    qtype: str,
    *,
    kernel: OstrisKernel = "auto",
    max_workspace_mb: int = 64,
) -> Optional[OstrisQuantizer]:
    """Resolve a registered custom qtype, returning ``None`` for other backends."""
    _register_builtin_backends()
    factory = _backend_factories.get(qtype)
    if factory is None:
        return None
    options = OstrisBackendOptions(
        kernel=kernel,
        max_workspace_mb=max_workspace_mb,
    )
    quantizer = factory(options)
    quantizer.backend_name = qtype
    quantizer.backend_options = options
    return quantizer


def prepare_linear_for_ostris_cache(
    module: torch.nn.Linear,
    quantizer: OstrisQuantizer,
    original_dtype: torch.dtype,
) -> None:
    """Convert an unmaterialized Linear before packed cache buffers are restored."""
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"expected nn.Linear, got {type(module).__name__}")
    if not isinstance(module, OstrisLinear):
        module._parameters.pop("weight", None)
        module.__class__ = OstrisLinear
    module.ostris_quantizer = quantizer
    module.ostris_orig_dtype = original_dtype
    module.ostris_backend_name = getattr(quantizer, "backend_name", None)


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

    original_dtype = weight.dtype
    # Backends own any bounded casting they require. An unconditional full
    # float32 copy here defeats low-VRAM quantization for very large linears.
    quantizer.quantize_(module, weight.detach())
    module.ostris_quantizer = quantizer
    module.ostris_orig_dtype = original_dtype
    module.ostris_backend_name = getattr(quantizer, "backend_name", None)
    del module._parameters["weight"]
    if module.bias is not None:
        module.bias.requires_grad_(False)
    module.__class__ = OstrisLinear
    adapter_ref = getattr(module, "ara_lora_ref", None)
    adapter = adapter_ref() if callable(adapter_ref) else adapter_ref
    if adapter is not None and hasattr(adapter, "can_merge_in"):
        adapter.can_merge_in = False
    return True
