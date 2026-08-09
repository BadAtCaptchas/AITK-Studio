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
        from toolkit.util.convrot_quant import CONVROT_QTYPES, get_convrot_quantizer
        from toolkit.util.nvfp4_quant import NVFP4_QTYPES, Nvfp4Quantizer
        from toolkit.util.uintx_quant import UINTX_QTYPES, UIntXQuantizer

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
                    format_version=2,
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

        for name in CONVROT_QTYPES:
            if name in {"convrot4", "convrotcomfyw4a4"}:
                bits = 4
            elif name == "convrotbitnet":
                bits = 2
            elif name.startswith("convrotint"):
                bits = int(name.removeprefix("convrotint"))
            else:
                bits = 8
            extra_capabilities = (
                "quantization_aware_training",
                "packed_save_load",
            )
            if name == "convrotcomfyw4a4":
                extra_capabilities += ("comfyui_export",)
            register_ostris_backend(
                OstrisBackendMetadata(
                    name=name,
                    format_version=1,
                    bits=bits,
                    status="experimental",
                    capabilities=common_capabilities
                    + ("torch_fallback", "triton_optional")
                    + extra_capabilities,
                    supported_devices=("cpu", "cuda"),
                    shape_notes=(
                        "nn.Linear only; FP4 requires in/out divisible by 16; "
                        "integer backends require in divisible by 16 and out by 8; "
                        "Comfy W4A4 additionally requires in divisible by 256"
                    ),
                    device_notes=(
                        "convrot4 accelerates with FP4 on Blackwell; integer backends "
                        "use runtime-probed INT8 matmul; unsupported devices use "
                        "bounded dequantized fallbacks"
                    ),
                ),
                lambda options, name=name: get_convrot_quantizer(
                    name,
                    kernel=options.kernel,
                    max_workspace_mb=options.max_workspace_mb,
                ),
            )

        for name in NVFP4_QTYPES:
            register_ostris_backend(
                OstrisBackendMetadata(
                    name=name,
                    format_version=1,
                    bits=4,
                    status="experimental",
                    capabilities=common_capabilities
                    + (
                        "torch_fallback",
                        "packed_save_load",
                        "comfyui_import",
                        "awq_pre_scale",
                    ),
                    supported_devices=("cpu", "cuda"),
                    shape_notes="nn.Linear only; in_features must be divisible by 16",
                    device_notes=(
                        "Full-precision activations with bounded NVFP4 weight "
                        "dequantization; packed-weight merge and requantization "
                        "are intentionally unsupported"
                    ),
                ),
                lambda options: Nvfp4Quantizer(),
            )

        for name, bits in UINTX_QTYPES.items():
            register_ostris_backend(
                OstrisBackendMetadata(
                    name=name,
                    format_version=1,
                    bits=bits,
                    status="stable",
                    capabilities=common_capabilities
                    + (
                        "torch_fallback",
                        "packed_save_load",
                        "torchao_0_10_bit_exact",
                    ),
                    supported_devices=("cpu", "cuda"),
                    shape_notes=(
                        "nn.Linear only; in_features must be divisible by 64; "
                        "uint8 remains intentionally unsupported"
                    ),
                    device_notes=(
                        "Bounded row-wise torch implementation on CPU and CUDA"
                    ),
                ),
                lambda options, bits=bits: UIntXQuantizer(
                    bits,
                    max_workspace_mb=options.max_workspace_mb,
                ),
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
        if (
            torch.is_grad_enabled()
            and x.requires_grad
            and not getattr(self.ostris_quantizer, "handles_autograd", False)
        ):
            return _FrozenOstrisLinearFunction.apply(x, self)
        return self.ostris_quantizer.forward(self, x)

    @torch.no_grad()
    def requantize_(self, fp_weight: torch.Tensor) -> None:
        self.ostris_quantizer.requantize_(self, fp_weight)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Avoid retaining every dequantized layer at once when a large model's
        # state dict is collected. Save paths materialize this view one key at a time.
        destination[prefix + "weight"] = OstrisLazyWeight(self)
        if self.bias is not None:
            destination[prefix + "bias"] = (
                self.bias if keep_vars else self.bias.detach()
            )


class OstrisLazyWeight(torch.Tensor):
    """A no-storage state-dict view that materializes its packed layer on demand."""

    @staticmethod
    def __new__(cls, module: "OstrisLinear"):
        buffer = next(
            value for value in module._buffers.values() if value is not None
        )
        result = torch.Tensor._make_wrapper_subclass(
            cls,
            (module.out_features, module.in_features),
            dtype=module.ostris_orig_dtype,
            device=buffer.device,
            requires_grad=False,
        )
        result._ostris_module = module
        result._is_ostris_weight = True
        return result

    def dequantize(self) -> torch.Tensor:
        return self._ostris_module.dequantize_weight()

    def __repr__(self):
        return (
            f"OstrisLazyWeight(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device})"
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        from torch.utils._pytree import tree_map

        def unwrap(value):
            if isinstance(value, cls):
                return value._ostris_module.dequantize_weight()
            return value

        return func(
            *tree_map(unwrap, args),
            **tree_map(unwrap, kwargs or {}),
        )


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
    # Upstream's packed-layer helpers historically called this field qtype.
    # Keep both names so old packed saves and the local registry interoperate.
    quantizer.qtype = qtype
    quantizer.backend_options = options
    return quantizer


QUANT_LAYERS_METADATA_KEY = "aitk_quantization"
_PACKED_LAYER_ATTRIBUTES = (
    "orbit_bits",
    "orbit_block",
    "orbit_kernel",
    "orbit_max_workspace_mb",
    "orbit_packed_layout",
    "ovq_block",
    "ovq_group",
    "cr_rot_size",
    "cr8_rot_size",
    "crn_bits",
    "crn_rot_size",
    "convrot_kernel",
    "convrot_max_workspace_mb",
    "convrot_packed_layout",
    "uintx_bits",
    "uintx_group_size",
    "uintx_max_workspace_mb",
    "uintx_packed_layout",
)


@torch.no_grad()
def save_quantized_layers(
    modules: Dict[str, "OstrisLinear"],
    file_path: str,
    metadata: Optional[Dict[str, str]] = None,
    extra_tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Save packed backend buffers for a set of named linear layers."""
    import json

    from safetensors.torch import save_file

    quant_map: Dict[str, object] = {}
    state_dict: Dict[str, torch.Tensor] = {}
    for name, module in modules.items():
        if not isinstance(module, OstrisLinear):
            raise TypeError(f"{name!r} is not an OstrisLinear")
        qtype = getattr(module, "ostris_backend_name", None) or getattr(
            module.ostris_quantizer, "backend_name", None
        )
        if not isinstance(qtype, str) or not is_ostris_qtype(qtype):
            raise ValueError(f"Cannot identify the packed backend for module {name!r}")
        entry = {
            "qtype": qtype,
            "dtype": str(module.ostris_orig_dtype).removeprefix("torch."),
            "buffers": [],
            "attrs": {},
        }
        if module.bias is not None:
            state_dict[f"{name}.bias"] = module.bias
        for buffer_name, buffer in module._buffers.items():
            if buffer is None:
                continue
            state_dict[f"{name}.{buffer_name}"] = buffer
            entry["buffers"].append(buffer_name)
        for attribute in _PACKED_LAYER_ATTRIBUTES:
            value = getattr(module, attribute, None)
            if isinstance(value, (bool, int, float, str)):
                entry["attrs"][attribute] = value
        quant_map[name] = entry

    if extra_tensors:
        state_dict.update(extra_tensors)
    cpu_state = {
        key: value.detach().to("cpu", copy=True).contiguous()
        for key, value in state_dict.items()
    }
    file_metadata = dict(metadata or {})
    file_metadata[QUANT_LAYERS_METADATA_KEY] = json.dumps(
        {"modules": quant_map, "layers_only": True}, separators=(",", ":")
    )
    save_file(cpu_state, file_path, metadata=file_metadata)


@torch.no_grad()
def load_quantized_layers(root: torch.nn.Module, file_path: str) -> int:
    """Restore a packed-layer file without materializing dense weights."""
    import json

    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(file_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    payload = metadata.get(QUANT_LAYERS_METADATA_KEY)
    if payload is None:
        raise ValueError(f"{file_path} has no quantized-layer metadata")
    manifest = json.loads(payload)
    quant_map = manifest.get("modules")
    if not isinstance(quant_map, dict):
        raise ValueError(f"{file_path} has invalid quantized-layer metadata")
    state_dict = load_file(file_path, device="cpu")

    restored = 0
    for name, raw_entry in quant_map.items():
        if not isinstance(name, str) or not isinstance(raw_entry, dict):
            raise ValueError(f"{file_path} has an invalid module entry")
        module = root.get_submodule(name)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"target module {name!r} is not an nn.Linear")
        qtype = raw_entry.get("qtype")
        attrs = raw_entry.get("attrs", {})
        if not isinstance(qtype, str) or not isinstance(attrs, dict):
            raise ValueError(f"{file_path} has invalid metadata for {name!r}")
        kernel = attrs.get("convrot_kernel", attrs.get("orbit_kernel", "auto"))
        workspace = attrs.get(
            "convrot_max_workspace_mb",
            attrs.get(
                "orbit_max_workspace_mb",
                attrs.get("uintx_max_workspace_mb", 64),
            ),
        )
        quantizer = get_ostris_quantizer(
            qtype, kernel=kernel, max_workspace_mb=workspace
        )
        if quantizer is None:
            raise ValueError(f"Unknown qtype {qtype!r} in {file_path}")

        if isinstance(module, OstrisLinear):
            device = module.quantized_device
            module._buffers.clear()
        else:
            weight = module._parameters.get("weight")
            device = weight.device if weight is not None else torch.device("cpu")
            module._parameters.pop("weight", None)
            module.__class__ = OstrisLinear
        dtype_name = raw_entry.get("dtype")
        original_dtype = getattr(torch, dtype_name, None)
        if not isinstance(original_dtype, torch.dtype) or not original_dtype.is_floating_point:
            raise ValueError(f"unsupported original dtype {dtype_name!r} for {name!r}")
        module.ostris_orig_dtype = original_dtype

        buffer_names = raw_entry.get("buffers")
        if not isinstance(buffer_names, list) or not all(
            isinstance(value, str) for value in buffer_names
        ):
            raise ValueError(f"invalid buffer list for {name!r}")
        for buffer_name in buffer_names:
            key = f"{name}.{buffer_name}"
            if key not in state_dict:
                raise ValueError(f"missing packed buffer {key!r}")
            module.register_buffer(
                buffer_name,
                state_dict[key].to(device=device),
                persistent=False,
            )
        for attribute, value in attrs.items():
            if attribute in _PACKED_LAYER_ATTRIBUTES and isinstance(
                value, (bool, int, float, str)
            ):
                setattr(module, attribute, value)
        module.ostris_quantizer = quantizer
        module.ostris_backend_name = qtype
        bias_key = f"{name}.bias"
        if bias_key in state_dict:
            if module.bias is None:
                raise ValueError(f"packed layer {name!r} has a bias but the target does not")
            module.bias.data.copy_(state_dict[bias_key].to(device, module.bias.dtype))
        if module.bias is not None:
            module.bias.requires_grad_(False)
        restored += 1
    return restored


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
