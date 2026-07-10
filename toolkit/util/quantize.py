from fnmatch import fnmatch
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union, TYPE_CHECKING
import torch

from optimum.quanto.quantize import _quantize_submodule
from optimum.quanto.tensor import Optimizer, QTensor, qtype, qtypes
from torchao.quantization.quant_api import (
    quantize_ as torchao_quantize_,
    Float8WeightOnlyConfig,
    UIntXWeightOnlyConfig,
    Int8WeightOnlyConfig
)
from optimum.quanto import freeze
from tqdm import tqdm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from toolkit.print import print_acc
from toolkit.util.ostris_quant import (
    OstrisBackendMetadata,
    OstrisBackendOptions,
    OstrisKernel,
    OstrisLinear,
    OstrisQuantizer,
    convert_linear_to_ostris,
    get_ostris_backend_metadata,
    get_ostris_quantizer,
)
import os

if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel

# the quantize function in quanto had a bug where it was using exclude instead of include

Q_MODULES = [
    "QLinear",
    "QConv2d",
    "QEmbedding",
    "QBatchNorm2d",
    "QLayerNorm",
    "QConvTranspose2d",
    "QEmbeddingBag",
    "OstrisLinear",
]

torchao_qtypes = {
    # "int4": Int4WeightOnlyConfig(),
    "uint2": UIntXWeightOnlyConfig(torch.uint2),
    "uint3": UIntXWeightOnlyConfig(torch.uint3),
    "uint4": UIntXWeightOnlyConfig(torch.uint4),
    "uint5": UIntXWeightOnlyConfig(torch.uint5),
    "uint6": UIntXWeightOnlyConfig(torch.uint6),
    "uint7": UIntXWeightOnlyConfig(torch.uint7),
    "uint8": UIntXWeightOnlyConfig(torch.uint8),
    "int8": Int8WeightOnlyConfig(),
    "float8": Float8WeightOnlyConfig(),
}


@dataclass
class QuantizationSkipReason:
    modules: int = 0
    bytes: int = 0
    examples: List[str] = field(default_factory=list)

    def add(self, name: str, byte_count: int) -> None:
        self.modules += 1
        self.bytes += byte_count
        if len(self.examples) < 8:
            self.examples.append(name or "<root>")

    def merge(self, other: "QuantizationSkipReason") -> None:
        self.modules += other.modules
        self.bytes += other.bytes
        for name in other.examples:
            if len(self.examples) >= 8:
                break
            if name not in self.examples:
                self.examples.append(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modules": self.modules,
            "bytes": self.bytes,
            "examples": list(self.examples),
        }


@dataclass
class QuantizationReport:
    """Byte coverage and CUDA observations for one quantization phase."""

    qtype: str
    backend: str
    component: Optional[str] = None
    eligible_bytes: int = 0
    quantized_original_bytes: int = 0
    quantized_weight_count: int = 0
    compressed_bytes: int = 0
    dense_skipped_bytes: int = 0
    metadata_bytes: int = 0
    quantized_modules: int = 0
    skipped_modules: int = 0
    skip_reasons: Dict[str, QuantizationSkipReason] = field(default_factory=dict)
    cuda_device: Optional[str] = None
    cuda_allocated_before_bytes: Optional[int] = None
    cuda_reserved_before_bytes: Optional[int] = None
    cuda_allocated_after_bytes: Optional[int] = None
    cuda_reserved_after_bytes: Optional[int] = None
    cuda_peak_allocated_bytes: Optional[int] = None
    cuda_peak_reserved_bytes: Optional[int] = None

    @property
    def persistent_bytes(self) -> int:
        return self.compressed_bytes + self.metadata_bytes

    @property
    def original_accounted_bytes(self) -> int:
        return self.quantized_original_bytes + self.dense_skipped_bytes

    @property
    def coverage(self) -> float:
        if self.original_accounted_bytes == 0:
            return 0.0
        return self.quantized_original_bytes / self.original_accounted_bytes

    @property
    def storage_bytes_per_weight_byte(self) -> float:
        if self.quantized_original_bytes == 0:
            return 0.0
        return self.persistent_bytes / self.quantized_original_bytes

    @property
    def persistent_bytes_per_weight(self) -> float:
        if self.quantized_weight_count == 0:
            return 0.0
        return self.persistent_bytes / self.quantized_weight_count

    @property
    def compressed_bytes_per_weight(self) -> float:
        if self.quantized_weight_count == 0:
            return 0.0
        return self.compressed_bytes / self.quantized_weight_count

    def add_skip(self, reason: str, name: str, byte_count: int) -> None:
        self.skipped_modules += 1
        self.dense_skipped_bytes += byte_count
        self.skip_reasons.setdefault(reason, QuantizationSkipReason()).add(
            name,
            byte_count,
        )

    def merge(self, other: "QuantizationReport") -> "QuantizationReport":
        if self.qtype != other.qtype or self.backend != other.backend:
            raise ValueError("cannot merge reports from different quantization backends")
        self.eligible_bytes += other.eligible_bytes
        self.quantized_original_bytes += other.quantized_original_bytes
        self.quantized_weight_count += other.quantized_weight_count
        self.compressed_bytes += other.compressed_bytes
        self.dense_skipped_bytes += other.dense_skipped_bytes
        self.metadata_bytes += other.metadata_bytes
        self.quantized_modules += other.quantized_modules
        self.skipped_modules += other.skipped_modules
        for reason, values in other.skip_reasons.items():
            self.skip_reasons.setdefault(reason, QuantizationSkipReason()).merge(values)
        if self.cuda_device is None:
            self.cuda_device = other.cuda_device
        if self.cuda_allocated_before_bytes is None:
            self.cuda_allocated_before_bytes = other.cuda_allocated_before_bytes
        if self.cuda_reserved_before_bytes is None:
            self.cuda_reserved_before_bytes = other.cuda_reserved_before_bytes
        self.cuda_allocated_after_bytes = other.cuda_allocated_after_bytes
        self.cuda_reserved_after_bytes = other.cuda_reserved_after_bytes
        for field_name in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
            current = getattr(self, field_name)
            candidate = getattr(other, field_name)
            if candidate is not None:
                setattr(self, field_name, candidate if current is None else max(current, candidate))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qtype": self.qtype,
            "backend": self.backend,
            "component": self.component,
            "eligible_bytes": self.eligible_bytes,
            "quantized_original_bytes": self.quantized_original_bytes,
            "quantized_weight_count": self.quantized_weight_count,
            "compressed_bytes": self.compressed_bytes,
            "dense_skipped_bytes": self.dense_skipped_bytes,
            "metadata_bytes": self.metadata_bytes,
            "persistent_bytes": self.persistent_bytes,
            "coverage": self.coverage,
            "storage_bytes_per_weight_byte": self.storage_bytes_per_weight_byte,
            "persistent_bytes_per_weight": self.persistent_bytes_per_weight,
            "compressed_bytes_per_weight": self.compressed_bytes_per_weight,
            "quantized_modules": self.quantized_modules,
            "skipped_modules": self.skipped_modules,
            "skip_reasons": {
                key: value.to_dict() for key, value in sorted(self.skip_reasons.items())
            },
            "cuda": {
                "device": self.cuda_device,
                "allocated_before_bytes": self.cuda_allocated_before_bytes,
                "reserved_before_bytes": self.cuda_reserved_before_bytes,
                "allocated_after_bytes": self.cuda_allocated_after_bytes,
                "reserved_after_bytes": self.cuda_reserved_after_bytes,
                "peak_allocated_bytes": self.cuda_peak_allocated_bytes,
                "peak_reserved_bytes": self.cuda_peak_reserved_bytes,
            },
        }

    def summary(self) -> str:
        return (
            f"{self.component or 'model'} {self.qtype}: "
            f"{self.quantized_modules} modules, "
            f"{self.eligible_bytes / (1024 ** 2):.1f} MiB eligible -> "
            f"{self.persistent_bytes / (1024 ** 2):.1f} MiB packed+metadata; "
            f"{self.dense_skipped_bytes / (1024 ** 2):.1f} MiB dense skipped "
            f"({self.coverage:.1%} coverage)"
        )


def enforce_orbit4_low_vram_coverage(
    report: QuantizationReport,
    *,
    minimum_coverage: float = 0.95,
) -> None:
    """Fail early when an Orbit4 job leaves too many linear weights dense."""
    if report.qtype != "orbit4":
        raise ValueError("the low-VRAM coverage gate requires an orbit4 report")
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in the interval (0, 1]")
    if report.original_accounted_bytes == 0:
        raise ValueError("Orbit4 did not find any materialized linear weights to quantize")
    if report.coverage >= minimum_coverage:
        return
    reasons = ", ".join(
        f"{name}={values.bytes / (1024 ** 2):.1f} MiB/{values.modules} modules"
        for name, values in sorted(
            report.skip_reasons.items(),
            key=lambda item: item[1].bytes,
            reverse=True,
        )
    ) or "no skip reasons were recorded"
    raise ValueError(
        f"Orbit4 low-VRAM coverage is {report.coverage:.1%}, below the "
        f"required {minimum_coverage:.1%}; dense leftovers: {reasons}"
    )


def _cuda_memory_snapshot(device: Optional[torch.device] = None) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    try:
        cuda_device = torch.device(device) if device is not None else torch.device(
            "cuda", torch.cuda.current_device()
        )
        if cuda_device.type != "cuda":
            cuda_device = torch.device("cuda", torch.cuda.current_device())
        return {
            "cuda_device": str(cuda_device),
            "cuda_allocated": int(torch.cuda.memory_allocated(cuda_device)),
            "cuda_reserved": int(torch.cuda.memory_reserved(cuda_device)),
            "cuda_peak_allocated": int(torch.cuda.max_memory_allocated(cuda_device)),
            "cuda_peak_reserved": int(torch.cuda.max_memory_reserved(cuda_device)),
        }
    except Exception:
        return {}


def _start_report(
    weights: object,
    component: Optional[str],
    device: Optional[torch.device] = None,
) -> QuantizationReport:
    qtype_name = getattr(weights, "name", None) or str(weights)
    if isinstance(weights, ostristype):
        backend = "ostris"
    elif isinstance(weights, aotype):
        backend = "torchao"
    else:
        backend = "quanto"
    report = QuantizationReport(qtype=qtype_name, backend=backend, component=component)
    memory = _cuda_memory_snapshot(device)
    report.cuda_device = memory.get("cuda_device")
    report.cuda_allocated_before_bytes = memory.get("cuda_allocated")
    report.cuda_reserved_before_bytes = memory.get("cuda_reserved")
    report.cuda_peak_allocated_bytes = memory.get("cuda_peak_allocated")
    report.cuda_peak_reserved_bytes = memory.get("cuda_peak_reserved")
    return report


def _finish_report(
    report: QuantizationReport,
    device: Optional[torch.device] = None,
) -> None:
    memory = _cuda_memory_snapshot(device)
    report.cuda_allocated_after_bytes = memory.get("cuda_allocated")
    report.cuda_reserved_after_bytes = memory.get("cuda_reserved")
    for report_name, memory_name in (
        ("cuda_peak_allocated_bytes", "cuda_peak_allocated"),
        ("cuda_peak_reserved_bytes", "cuda_peak_reserved"),
    ):
        value = memory.get(memory_name)
        current = getattr(report, report_name)
        if value is not None:
            setattr(report, report_name, value if current is None else max(current, value))


class aotype:
    def __init__(self, name: str):
        self.name = name
        self.config = torchao_qtypes[name]


class ostristype:
    """Resolved custom quantization backend and its serializable qtype name."""

    def __init__(
        self,
        name: str,
        quantizer: OstrisQuantizer,
        metadata: OstrisBackendMetadata,
        options: OstrisBackendOptions,
    ):
        self.name = name
        self.quantizer = quantizer
        self.metadata = metadata
        self.options = options


def get_qtype(
    qtype: Union[str, qtype, aotype, ostristype],
    *,
    kernel: OstrisKernel = "auto",
    max_workspace_mb: int = 64,
):
    if isinstance(qtype, (aotype, ostristype)):
        return qtype
    if isinstance(qtype, str) and qtype in torchao_qtypes:
        return aotype(qtype)
    if isinstance(qtype, str):
        options = OstrisBackendOptions(
            kernel=kernel,
            max_workspace_mb=max_workspace_mb,
        )
        metadata = get_ostris_backend_metadata(qtype)
        ostris_quantizer = get_ostris_quantizer(
            qtype,
            kernel=options.kernel,
            max_workspace_mb=options.max_workspace_mb,
        )
        if ostris_quantizer is not None:
            if metadata is None:
                raise RuntimeError(f"backend {qtype!r} has no registry metadata")
            return ostristype(qtype, ostris_quantizer, metadata, options)
        return qtypes[qtype]
    else:
        return qtype


def _normalize_patterns(
    value: Optional[Union[str, Sequence[str]]],
    field_name: str,
) -> Optional[List[str]]:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    if not all(isinstance(item, str) for item in values):
        raise TypeError(f"{field_name} must be a string or a sequence of strings")
    return [item for item in values if item]


def _matches_patterns(
    name: str,
    include: Optional[Sequence[str]],
    exclude: Optional[Sequence[str]],
) -> Optional[str]:
    if include is not None and not any(fnmatch(name, pattern) for pattern in include):
        return "not_included"
    if exclude is not None and any(fnmatch(name, pattern) for pattern in exclude):
        return "excluded"
    return None


def _qualified_name(prefix: str, name: str) -> str:
    if not prefix:
        return name
    if not name:
        return prefix
    return f"{prefix}.{name}"


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _linear_weight_bytes(module: torch.nn.Module) -> int:
    if isinstance(module, OstrisLinear):
        return int(module.logical_weight_numel) * torch.empty(
            (), dtype=module.ostris_orig_dtype
        ).element_size()
    parameter = module._parameters.get("weight")
    if isinstance(parameter, torch.Tensor) and parameter.dtype.is_floating_point:
        return _tensor_bytes(parameter)
    return 0


def _ostris_storage_bytes(module: OstrisLinear) -> tuple[int, int]:
    compressed = 0
    metadata = 0
    for name, buffer in module._buffers.items():
        if not isinstance(buffer, torch.Tensor):
            continue
        if name in ("orbit_packed", "ovq_packed"):
            compressed += _tensor_bytes(buffer)
        else:
            metadata += _tensor_bytes(buffer)
    return compressed, metadata


def is_quantized_tensor(t) -> bool:
    # torchao stores quantized weights as tensor subclasses under torchao.* that
    # still expose dequantize(). Quanto QTensor is handled in dequantize_if_quantized.
    if getattr(t, '_is_ostris_weight', False):
        return True
    return 'torchao' in type(t).__module__ and hasattr(t, 'dequantize')


def dequantize_if_quantized(t):
    return t.dequantize() if isinstance(t, QTensor) or is_quantized_tensor(t) else t


def get_torchao_config(qtype):
    if qtype is None:
        return None
    try:
        q = get_qtype(qtype)
    except Exception:
        return None
    if isinstance(q, aotype):
        return q.config
    if isinstance(q, ostristype):
        return q
    return None


def requantize_module_weight(module, fp_weight, orig_dtype, config) -> None:
    """Write a full precision weight back into module.weight, re-quantizing when possible."""
    if isinstance(module, OstrisLinear):
        module.requantize_(fp_weight)
        return
    if isinstance(config, ostristype):
        # This layer did not satisfy the backend's shape requirements.
        config = None
    module.weight = torch.nn.Parameter(fp_weight.to(orig_dtype), requires_grad=False)
    if config is not None:
        torchao_quantize_(module, config)


def quantize(
    model: torch.nn.Module,
    weights: Optional[Union[str, qtype, aotype, ostristype]] = None,
    activations: Optional[Union[str, qtype]] = None,
    optimizer: Optional[Optimizer] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
    kernel: Optional[OstrisKernel] = None,
    max_workspace_mb: Optional[int] = None,
    *,
    component_label: Optional[str] = None,
    name_prefix: str = "",
) -> QuantizationReport:
    """Quantize the specified model submodules

    Recursively quantize the submodules of the specified parent model.

    Only modules that have quantized counterparts will be quantized.

    If include patterns are specified, the submodule name must match one of them.

    If exclude patterns are specified, the submodule must not match one of them.

    Include or exclude patterns are Unix shell-style wildcards which are NOT regular expressions. See
    https://docs.python.org/3/library/fnmatch.html for more details.

    Note: quantization happens in-place and modifies the original model and its descendants.

    Args:
        model (`torch.nn.Module`): the model whose submodules will be quantized.
        weights (`Optional[Union[str, qtype]]`): the qtype for weights quantization.
        activations (`Optional[Union[str, qtype]]`): the qtype for activations quantization.
        include (`Optional[Union[str, List[str]]]`):
            Patterns constituting the allowlist. If provided, module names must match at
            least one pattern from the allowlist.
        exclude (`Optional[Union[str, List[str]]]`):
            Patterns constituting the denylist. If provided, module names must not match
            any patterns from the denylist.
    """
    include = _normalize_patterns(include, "include")
    exclude = _normalize_patterns(exclude, "exclude")

    inherited_options = weights.options if isinstance(weights, ostristype) else None
    effective_kernel: OstrisKernel = (
        kernel
        if kernel is not None
        else inherited_options.kernel if inherited_options is not None else "auto"
    )
    effective_workspace = (
        max_workspace_mb
        if max_workspace_mb is not None
        else inherited_options.max_workspace_mb if inherited_options is not None else 64
    )
    # Validate options even for Quanto/TorchAO so misspelled low-VRAM settings
    # fail at the public boundary instead of being silently ignored.
    backend_options = OstrisBackendOptions(
        kernel=effective_kernel,
        max_workspace_mb=effective_workspace,
    )
    if isinstance(weights, str):
        weights = get_qtype(
            weights,
            kernel=backend_options.kernel,
            max_workspace_mb=backend_options.max_workspace_mb,
        )
    elif isinstance(weights, ostristype) and weights.options != backend_options:
        weights = get_qtype(
            weights.name,
            kernel=backend_options.kernel,
            max_workspace_mb=backend_options.max_workspace_mb,
        )

    report = _start_report(weights, component_label)
    for name, m in list(model.named_modules()):
        qualified_name = _qualified_name(name_prefix, name)
        weight_bytes = _linear_weight_bytes(m) if isinstance(m, torch.nn.Linear) else 0
        pattern_skip = _matches_patterns(qualified_name, include, exclude)

        if isinstance(weights, ostristype) and isinstance(m, torch.nn.Linear):
            if isinstance(m, OstrisLinear):
                continue
            parameter = m._parameters.get("weight")
            is_candidate = (
                isinstance(parameter, torch.nn.Parameter)
                and parameter.dtype.is_floating_point
                and type(parameter.data) is torch.Tensor
            )
            if not is_candidate:
                if weight_bytes:
                    report.eligible_bytes += weight_bytes
                    report.add_skip("unsupported_weight", qualified_name, weight_bytes)
                continue
            report.eligible_bytes += weight_bytes
            if not weights.quantizer.can_quantize(m):
                report.add_skip("unsupported_shape", qualified_name, weight_bytes)
                continue
            if pattern_skip is not None:
                report.add_skip(pattern_skip, qualified_name, weight_bytes)
                continue
            try:
                if not convert_linear_to_ostris(m, weights.quantizer):
                    report.add_skip("conversion_rejected", qualified_name, weight_bytes)
                    continue
                compressed, metadata = _ostris_storage_bytes(m)
                report.compressed_bytes += compressed
                report.metadata_bytes += metadata
                report.quantized_original_bytes += weight_bytes
                report.quantized_weight_count += int(m.in_features) * int(m.out_features)
                report.quantized_modules += 1
            except Exception as e:
                print(f"Failed to quantize {qualified_name}: {e}")
                raise
            continue

        if pattern_skip is not None:
            if weight_bytes:
                report.add_skip(pattern_skip, qualified_name, weight_bytes)
            continue
        try:
            # check if m is QLinear or QConv2d
            if m.__class__.__name__ in Q_MODULES:
                continue
            else:
                if weight_bytes:
                    report.eligible_bytes += weight_bytes
                if isinstance(weights, aotype):
                    torchao_quantize_(m, weights.config)
                else:
                    _quantize_submodule(
                        model,
                        name,
                        m,
                        weights=weights,
                        activations=activations,
                        optimizer=optimizer,
                    )
                if weight_bytes:
                    report.quantized_modules += 1
        except Exception as e:
            print(f"Failed to quantize {qualified_name}: {e}")
            raise

    _finish_report(report)
    model._aitk_quantization_report = report
    return report


def _resolve_staged_blocks(
    component: torch.nn.Module,
    block_paths: Optional[Sequence[str]],
) -> List[tuple[str, torch.nn.Module]]:
    blocks: List[tuple[str, torch.nn.Module]] = []
    seen: set[int] = set()
    for path in block_paths or ():
        try:
            target = component.get_submodule(path)
        except (AttributeError, KeyError):
            continue
        if isinstance(target, (torch.nn.ModuleList, torch.nn.Sequential)):
            candidates = [
                (f"{path}.{index}", child) for index, child in enumerate(target)
            ]
        else:
            candidates = [(path, target)]
        for name, block in candidates:
            if id(block) in seen:
                continue
            seen.add(id(block))
            blocks.append((name, block))
    return blocks


def _move_for_quantization(
    module: torch.nn.Module,
    device: torch.device,
    dtype: Optional[torch.dtype],
) -> None:
    kwargs: Dict[str, Any] = {"device": device, "non_blocking": True}
    if dtype is not None:
        kwargs["dtype"] = dtype
    module.to(**kwargs)


def quantize_component_in_stages(
    component: torch.nn.Module,
    weights: Union[str, qtype, aotype, ostristype],
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    *,
    block_paths: Optional[Sequence[str]] = None,
    include: Optional[Union[str, Sequence[str]]] = None,
    exclude: Optional[Union[str, Sequence[str]]] = None,
    options: Optional[Mapping[str, Any]] = None,
    component_label: Optional[str] = None,
) -> QuantizationReport:
    """Quantize a CPU component while staging at most one block or linear.

    Include/exclude patterns are evaluated against absolute names within the
    component, even while an individual block is temporarily treated as the
    quantization root.
    """
    raw_options = dict(options or {})
    allowed_options = {"kernel", "max_workspace_mb", "include", "exclude"}
    unknown = sorted(set(raw_options) - allowed_options)
    if unknown:
        raise TypeError(f"unknown quantization options: {', '.join(unknown)}")

    option_include = raw_options.pop("include", None)
    option_exclude = raw_options.pop("exclude", None)
    include_patterns = _normalize_patterns(
        include if include is not None else option_include,
        "include",
    )
    explicit_exclude = _normalize_patterns(exclude, "exclude") or []
    configured_exclude = _normalize_patterns(option_exclude, "exclude") or []
    exclude_patterns = explicit_exclude + [
        pattern for pattern in configured_exclude if pattern not in explicit_exclude
    ]
    if not exclude_patterns:
        exclude_patterns = None

    kernel = raw_options.pop("kernel", None)
    max_workspace_mb = raw_options.pop("max_workspace_mb", None)
    inherited = weights.options if isinstance(weights, ostristype) else None
    effective_kernel = kernel if kernel is not None else (
        inherited.kernel if inherited is not None else "auto"
    )
    effective_workspace = max_workspace_mb if max_workspace_mb is not None else (
        inherited.max_workspace_mb if inherited is not None else 64
    )
    backend_options = OstrisBackendOptions(
        kernel=effective_kernel,
        max_workspace_mb=effective_workspace,
    )
    if isinstance(weights, str) or isinstance(weights, ostristype):
        weights = get_qtype(
            weights.name if isinstance(weights, ostristype) else weights,
            kernel=backend_options.kernel,
            max_workspace_mb=backend_options.max_workspace_mb,
        )

    target_device = torch.device(device)
    meta_parameters = [name for name, value in component.named_parameters() if value.is_meta]
    if meta_parameters:
        raise ValueError(
            "staged quantization requires materialized CPU weights; found meta "
            f"parameters such as {meta_parameters[0]!r}"
        )
    component.to("cpu")

    aggregate = _start_report(weights, component_label, target_device)
    blocks = _resolve_staged_blocks(component, block_paths)
    processed_prefixes: List[str] = []
    for absolute_name, block in blocks:
        _move_for_quantization(block, target_device, dtype)
        block_report = quantize(
            block,
            weights=weights,
            include=include_patterns,
            exclude=exclude_patterns,
            kernel=backend_options.kernel,
            max_workspace_mb=backend_options.max_workspace_mb,
            component_label=component_label,
            name_prefix=absolute_name,
        )
        freeze(block)
        block.to("cpu", non_blocking=True)
        aggregate.merge(block_report)
        processed_prefixes.append(absolute_name)

    if isinstance(weights, ostristype):
        # Quantize non-block linears individually. This is slower than staging a
        # block, but guarantees that an embedding or other large dense sibling
        # never follows it onto the accelerator.
        for absolute_name, module in list(component.named_modules()):
            if not isinstance(module, torch.nn.Linear) or isinstance(module, OstrisLinear):
                continue
            if any(
                absolute_name == prefix or absolute_name.startswith(f"{prefix}.")
                for prefix in processed_prefixes
            ):
                continue
            pattern_skip = _matches_patterns(
                absolute_name,
                include_patterns,
                exclude_patterns,
            )
            if pattern_skip is not None:
                skipped_report = quantize(
                    module,
                    weights=weights,
                    include=include_patterns,
                    exclude=exclude_patterns,
                    kernel=backend_options.kernel,
                    max_workspace_mb=backend_options.max_workspace_mb,
                    component_label=component_label,
                    name_prefix=absolute_name,
                )
                aggregate.merge(skipped_report)
                continue
            _move_for_quantization(module, target_device, dtype)
            module_report = quantize(
                module,
                weights=weights,
                include=include_patterns,
                exclude=exclude_patterns,
                kernel=backend_options.kernel,
                max_workspace_mb=backend_options.max_workspace_mb,
                component_label=component_label,
                name_prefix=absolute_name,
            )
            freeze(module)
            module.to("cpu", non_blocking=True)
            aggregate.merge(module_report)
    else:
        # Quanto/TorchAO retain their established recursive CPU behavior for
        # extras; importantly, the full component is never moved to CUDA.
        extras_report = quantize(
            component,
            weights=weights,
            include=include_patterns,
            exclude=exclude_patterns,
            kernel=backend_options.kernel,
            max_workspace_mb=backend_options.max_workspace_mb,
            component_label=component_label,
        )
        freeze(component)
        aggregate.merge(extras_report)

    _finish_report(aggregate, target_device)
    component._aitk_quantization_report = aggregate
    return aggregate


def quantize_model(
    base_model: "BaseModel",
    model_to_quantize: torch.nn.Module,
):
    from toolkit.dequantize import patch_dequantization_on_save

    if not hasattr(base_model, "get_transformer_block_names"):
        raise ValueError(
            "The model to quantize must have a method `get_transformer_block_names`."
        )

    # patch the state dict method
    patch_dequantization_on_save(model_to_quantize)

    # sensitive modules to keep in full precision (fnmatch patterns)
    exclude_modules = base_model.get_quantization_exclude_modules() or []
    quantize_options = dict(base_model.model_config.quantize_kwargs or {})
    configured_exclude = _normalize_patterns(
        quantize_options.get("exclude"),
        "exclude",
    ) or []
    combined_exclude = list(exclude_modules)
    combined_exclude.extend(
        pattern for pattern in configured_exclude if pattern not in combined_exclude
    )
    backend_call_kwargs = {
        "kernel": quantize_options.get("kernel"),
        "max_workspace_mb": quantize_options.get("max_workspace_mb"),
    }

    if base_model.model_config.accuracy_recovery_adapter is not None:
        from toolkit.config_modules import NetworkConfig
        from toolkit.lora_special import LoRASpecialNetwork

        # we need to load and quantize with an accuracy recovery adapter
        # todo handle hf repos
        load_lora_path = base_model.model_config.accuracy_recovery_adapter

        if not os.path.exists(load_lora_path):
            # not local file, grab from the hub

            path_split = load_lora_path.split("/")
            if len(path_split) > 3:
                raise ValueError(
                    "The accuracy recovery adapter path must be a local path or for a hf repo, 'username/repo_name/filename.safetensors'."
                )
            repo_id = f"{path_split[0]}/{path_split[1]}"
            print_acc(f"Grabbing lora from the hub: {load_lora_path}")
            new_lora_path = hf_hub_download(
                repo_id,
                filename=path_split[-1],
            )
            # replace the path
            load_lora_path = new_lora_path

        # build the lora config based on the lora weights
        lora_state_dict = load_file(load_lora_path)
        
        if hasattr(base_model, "convert_lora_weights_before_load"):
            lora_state_dict = base_model.convert_lora_weights_before_load(lora_state_dict)
        
        network_config = {
            "type": "lora",
            "network_kwargs": {"only_if_contains": []},
            "transformer_only": False,
        }
        first_key = list(lora_state_dict.keys())[0]
        first_weight = lora_state_dict[first_key]
        # if it starts with lycoris and includes lokr
        if any("lokr" in key.lower() for key in lora_state_dict.keys()):
            network_config["type"] = "lokr"
        
        network_kwargs = {}

        # find firse loraA weight
        if network_config["type"] == "lora":
            linear_dim = None
            for key, value in lora_state_dict.items():
                if "lora_A" in key:
                    linear_dim = int(value.shape[0])
                    break
            linear_alpha = linear_dim
            network_config["linear"] = linear_dim
            network_config["linear_alpha"] = linear_alpha

            # we build the keys to match every key
            only_if_contains = []
            for key in lora_state_dict.keys():
                contains_key = key.split(".lora_")[0]
                if contains_key not in only_if_contains:
                    only_if_contains.append(contains_key)

            network_kwargs["only_if_contains"] = only_if_contains
        elif network_config["type"] == "lokr":
            # find the factor
            largest_factor = 0
            inferred_rank = None
            lowered_keys = [key.lower() for key in lora_state_dict.keys()]
            has_w1_a = any("lokr_w1_a" in key for key in lowered_keys)
            has_w2_a = any("lokr_w2_a" in key for key in lowered_keys)
            has_tucker = any("lokr_t2" in key for key in lowered_keys)
            has_dora_scale = any("dora_scale" in key for key in lowered_keys)
            for key, value in lora_state_dict.items():
                if "lokr_w1" in key:
                    factor = int(value.shape[0])
                    if factor > largest_factor:
                        largest_factor = factor
                if inferred_rank is None and "lokr_w1_a" in key:
                    inferred_rank = int(value.shape[1])
                if inferred_rank is None and "lokr_w2_a" in key:
                    inferred_rank = int(value.shape[0] if has_tucker else value.shape[1])
            if inferred_rank is None:
                network_config["lokr_full_rank"] = True
            else:
                network_config["linear"] = inferred_rank
                network_config["linear_alpha"] = inferred_rank
                network_config["lokr_full_matrix"] = not has_w1_a and not has_w2_a
            network_config["lokr_factor"] = largest_factor
            network_config["lokr_use_tucker"] = has_tucker
            network_config["lokr_decompose_both"] = has_w1_a
            network_config["lokr_weight_decompose"] = has_dora_scale
            network_config["lokr_legacy_factorization"] = True

            only_if_contains = []
            for key in lora_state_dict.keys():
                if "lokr_w1" in key:
                    contains_key = key.split(".lokr_w1")[0]
                    contains_key = contains_key.replace("lycoris_", "")
                    if contains_key not in only_if_contains:
                        only_if_contains.append(contains_key)
            network_kwargs["only_if_contains"] = only_if_contains
        
        if hasattr(base_model, 'target_lora_modules'):
            network_kwargs['target_lin_modules'] = base_model.target_lora_modules

        # todo auto grab these
        # get dim and scale
        network_config = NetworkConfig(**network_config)

        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=model_to_quantize,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            # conv_lora_dim=self.network_config.conv,
            # conv_alpha=self.network_config.conv_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=base_model.is_transformer,
            base_model=base_model,
            is_ara=True,
            **network_kwargs
        )
        network.apply_to(
            None, model_to_quantize, apply_text_encoder=False, apply_unet=True
        )
        network.force_to(base_model.device_torch, dtype=base_model.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_state_dict)
        network.eval()
        network.is_active = True
        network.can_merge_in = False
        base_model.accuracy_recovery_adapter = network

        # quantize it
        lora_exclude_modules = []
        quantization_type = get_qtype(
            base_model.model_config.qtype,
            kernel=backend_call_kwargs["kernel"] or "auto",
            max_workspace_mb=backend_call_kwargs["max_workspace_mb"] or 64,
        )
        ara_report = _start_report(
            quantization_type,
            "transformer",
            base_model.device_torch,
        )
        for lora_module in tqdm(network.unet_loras, desc="Attaching quantization"):
            # the lora has already hijacked the original module
            orig_module = lora_module.org_module[0]
            orig_module.to(base_model.torch_dtype)
            # make the params not require gradients
            for param in orig_module.parameters():
                param.requires_grad = False
            module_report = quantize(
                orig_module,
                weights=quantization_type,
                **backend_call_kwargs,
            )
            ara_report.merge(module_report)
            freeze(orig_module)
            module_name = lora_module.lora_name.replace('$$', '.').replace('transformer.', '')
            lora_exclude_modules.append(module_name)
            if base_model.model_config.low_vram:
                # move it back to cpu
                orig_module.to("cpu")
        pass
        # quantize additional layers
        print_acc(" - quantizing additional layers")
        additional_quantization_type = (
            quantization_type
            if isinstance(quantization_type, ostristype)
            else get_qtype("uint8")
        )
        additional_report = quantize(
            model_to_quantize,
            weights=additional_quantization_type,
            include=quantize_options.get("include"),
            exclude=lora_exclude_modules + combined_exclude,
            **backend_call_kwargs,
        )
        if isinstance(quantization_type, ostristype):
            ara_report.merge(additional_report)
            _finish_report(ara_report, base_model.device_torch)
            model_to_quantize._aitk_quantization_report = ara_report
            base_model.quantization_report = ara_report
            if base_model.model_config.low_vram and quantization_type.name == "orbit4":
                enforce_orbit4_low_vram_coverage(ara_report)
            base_model.print_and_status_update(f" - {ara_report.summary()}")
            return ara_report
    else:
        transformer_block_names = base_model.get_transformer_block_names() or []
        all_blocks = _resolve_staged_blocks(
            model_to_quantize,
            transformer_block_names,
        )
        base_model.print_and_status_update(
            f" - quantizing {len(all_blocks)} transformer blocks"
        )
        report = quantize_component_in_stages(
            model_to_quantize,
            weights=base_model.model_config.qtype,
            device=base_model.device_torch,
            dtype=base_model.torch_dtype,
            block_paths=transformer_block_names,
            exclude=exclude_modules,
            options=quantize_options,
            component_label="transformer",
        )
        freeze(model_to_quantize)
        if base_model.model_config.low_vram and base_model.model_config.qtype == "orbit4":
            enforce_orbit4_low_vram_coverage(report)
        base_model.quantization_report = report
        base_model.print_and_status_update(f" - {report.summary()}")
        return report
