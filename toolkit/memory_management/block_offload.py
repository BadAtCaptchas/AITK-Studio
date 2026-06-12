from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Callable, Iterable, Optional, Sequence

import torch


DEFAULT_BLOCK_PATHS = [
    "transformer_blocks",
    "single_transformer_blocks",
    "double_blocks",
    "single_blocks",
    "blocks",
    "layers",
    "model.layers",
    "model.language_model.layers",
    "model.language_model.base_model.layers",
    "language_model.model.layers",
    "encoder.layers",
    "encoder.block",
    "encoder.layer",
    "text_model.encoder.layers",
    "text_model.encoder.layer",
    "transformer.h",
]


def _clamp_fraction(value: float) -> float:
    if value is None:
        return 1.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(parsed):
        return 1.0
    return max(0.0, min(1.0, parsed))


def _module_tensor_nbytes(module: torch.nn.Module) -> int:
    total = 0
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        total += tensor.numel() * tensor.element_size()
    return total


def _tensors_nbytes(tensors: Sequence[torch.Tensor]) -> int:
    total = 0
    for tensor in tensors:
        total += tensor.numel() * tensor.element_size()
    return total


def _is_storage_swap_supported(tensor: torch.Tensor) -> bool:
    try:
        data = tensor.data
        detached = tensor.detach()
    except Exception:
        return False

    for view in (data, detached):
        if type(view) is not torch.Tensor:
            return False
        try:
            if bool(getattr(view, "is_quantized", False)):
                return False
        except Exception:
            return False
    return True


def _get_child(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj[key]
    if isinstance(obj, (list, tuple, torch.nn.ModuleList, torch.nn.Sequential)):
        return obj[int(key)]
    return getattr(obj, key)


def _resolve_path(root: torch.nn.Module, path: str) -> Any:
    current: Any = root
    if path:
        for piece in path.split("."):
            if piece == "":
                continue
            current = _get_child(current, piece)
    return current


def _iter_ordered_modules(value: Any, path: str) -> Iterable[tuple[str, torch.nn.Module]]:
    if isinstance(value, (torch.nn.ModuleList, torch.nn.Sequential, list, tuple)):
        for index, item in enumerate(value):
            if not isinstance(item, torch.nn.Module):
                continue
            name = f"{path}.{index}" if path else str(index)
            yield name, item
    elif isinstance(value, torch.nn.Module):
        yield path, value


def resolve_block_layers(
    module: torch.nn.Module,
    block_paths: Optional[Sequence[str]] = None,
) -> list[tuple[str, torch.nn.Module]]:
    paths = list(block_paths or infer_block_paths(module))
    layers: list[tuple[str, torch.nn.Module]] = []
    seen: set[int] = set()

    for path in paths:
        try:
            value = _resolve_path(module, path)
        except (AttributeError, KeyError, IndexError, ValueError):
            continue
        for name, layer in _iter_ordered_modules(value, path):
            layer_id = id(layer)
            if layer_id in seen:
                continue
            seen.add(layer_id)
            layers.append((name, layer))

    return layers


def infer_block_paths(module: torch.nn.Module) -> list[str]:
    return [path for path in DEFAULT_BLOCK_PATHS if resolve_block_layers_for_path(module, path)]


def resolve_block_layers_for_path(module: torch.nn.Module, path: str) -> list[tuple[str, torch.nn.Module]]:
    try:
        value = _resolve_path(module, path)
    except (AttributeError, KeyError, IndexError, ValueError):
        return []
    return list(_iter_ordered_modules(value, path))


@dataclass(frozen=True)
class LayerOffloadStrategy:
    layer_sizes: tuple[int, ...]
    offload_fraction: float = 1.0
    offloaded_indices: tuple[int, ...] = field(init=False)
    resident_indices: tuple[int, ...] = field(init=False)
    window_byte_budget: int = field(init=False)

    def __post_init__(self):
        fraction = _clamp_fraction(self.offload_fraction)
        object.__setattr__(self, "offload_fraction", fraction)

        total_bytes = sum(max(0, size) for size in self.layer_sizes)
        target_offload_bytes = int(math.ceil(total_bytes * fraction))
        offloaded: list[int] = []
        offloaded_bytes = 0

        if target_offload_bytes > 0:
            # Offload from the tail forward. This is deterministic, keeps early
            # blocks resident for prompt/startup work, and gives the prefetcher
            # time to warm later blocks before they are first used.
            for index in range(len(self.layer_sizes) - 1, -1, -1):
                next_bytes = offloaded_bytes + max(0, self.layer_sizes[index])
                if (
                    offloaded
                    and abs(next_bytes - target_offload_bytes) > abs(offloaded_bytes - target_offload_bytes)
                ):
                    break
                offloaded.append(index)
                offloaded_bytes = next_bytes
                if offloaded_bytes >= target_offload_bytes:
                    break

        offloaded = sorted(offloaded)
        resident = [index for index in range(len(self.layer_sizes)) if index not in set(offloaded)]
        largest_layer = max((max(0, size) for size in self.layer_sizes), default=0)
        retained_bytes = max(0, total_bytes - target_offload_bytes)
        window_budget = max(largest_layer, retained_bytes)

        object.__setattr__(self, "offloaded_indices", tuple(offloaded))
        object.__setattr__(self, "resident_indices", tuple(resident))
        object.__setattr__(self, "window_byte_budget", window_budget)

    @classmethod
    def from_layers(
        cls,
        layers: Sequence[torch.nn.Module],
        offload_fraction: float,
    ) -> "LayerOffloadStrategy":
        return cls(tuple(_module_tensor_nbytes(layer) for layer in layers), offload_fraction)

    @property
    def offloaded_index_set(self) -> set[int]:
        return set(self.offloaded_indices)

    def is_offloaded(self, index: int) -> bool:
        return index in self.offloaded_index_set

    def forward_forward_window(self, index: int) -> tuple[int, ...]:
        return self._directional_window(index=index, step=1)

    def forward_backward_window(self, index: int) -> tuple[int, ...]:
        return self._directional_window(index=index, step=-1)

    def backward_forward_window(self, index: int) -> tuple[int, ...]:
        return self._directional_window(index=index, step=1)

    def _directional_window(self, index: int, step: int) -> tuple[int, ...]:
        if index < 0 or index >= len(self.layer_sizes):
            return tuple()

        offloaded = self.offloaded_index_set
        if index not in offloaded:
            return tuple()

        selected = [index]
        used_bytes = max(0, self.layer_sizes[index])
        cursor = index + step
        while 0 <= cursor < len(self.layer_sizes):
            if cursor in offloaded:
                layer_bytes = max(0, self.layer_sizes[cursor])
                if used_bytes + layer_bytes > self.window_byte_budget and selected:
                    break
                selected.append(cursor)
                used_bytes += layer_bytes
            cursor += step
        return tuple(sorted(selected))


class StaticPinnedCpuAllocator:
    @staticmethod
    def copy(
        tensor: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {"device": "cpu", "non_blocking": non_blocking}
        if dtype is not None and tensor.is_floating_point():
            kwargs["dtype"] = dtype
        result = tensor.detach().to(**kwargs)
        try:
            if torch.cuda.is_available() and not result.is_pinned():
                result = result.pin_memory()
        except RuntimeError:
            pass
        return result


class StaticCudaAllocator:
    @staticmethod
    def copy(
        tensor: torch.Tensor,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {"device": device, "non_blocking": non_blocking}
        if dtype is not None and tensor.is_floating_point():
            kwargs["dtype"] = dtype
        return tensor.to(**kwargs)


@dataclass
class _LayerCandidate:
    name: str
    module: torch.nn.Module
    params: list[torch.nn.Parameter]
    buffers: list[torch.Tensor]


@dataclass
class _ManagedLayer:
    index: int
    name: str
    module: torch.nn.Module
    original_forward: Callable[..., Any]
    params: list[torch.nn.Parameter]
    buffers: list[torch.Tensor]
    hook_handles: list[Any] = field(default_factory=list)
    transfer_event: Optional[torch.cuda.Event] = None
    state: str = "resident"


def _normalize_device(device: Any) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _extract_device_dtype(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Optional[torch.device], Optional[torch.dtype]]:
    device = kwargs.get("device")
    dtype = kwargs.get("dtype")

    for arg in args:
        if isinstance(arg, torch.device):
            device = arg
        elif isinstance(arg, str):
            try:
                parsed = torch.device(arg)
            except (TypeError, RuntimeError):
                parsed = None
            if parsed is not None:
                device = parsed
        elif isinstance(arg, torch.dtype):
            dtype = arg
        elif isinstance(arg, torch.Tensor):
            device = arg.device
            dtype = arg.dtype

    if device is not None and not isinstance(device, torch.device):
        device = torch.device(device)
    return device, dtype


class BlockOffloadManager:
    def __init__(
        self,
        module: torch.nn.Module,
        process_device: torch.device,
        layers: Sequence[_LayerCandidate],
        strategy: LayerOffloadStrategy,
        skipped_layers: Optional[Sequence[str]] = None,
    ):
        self.module = module
        self.process_device = _normalize_device(process_device)
        self.strategy = strategy
        self.skipped_layer_names = tuple(skipped_layers or ())
        self.active = self.process_device.type == "cuda" and torch.cuda.is_available()
        self.transfer_stream = None
        if self.active:
            self.transfer_stream = torch.cuda.Stream(device=self.process_device)

        self.layers: list[_ManagedLayer] = []
        for index, candidate in enumerate(layers):
            params = list(candidate.params)
            buffers = list(candidate.buffers)
            entry = _ManagedLayer(
                index=index,
                name=candidate.name,
                module=candidate.module,
                original_forward=candidate.module.forward,
                params=params,
                buffers=buffers,
                state="device" if self.active and self._entry_device(entry_params=params, entry_buffers=buffers) == self.process_device else "cpu",
            )
            self.layers.append(entry)

        self._managed_modules = {id(entry.module) for entry in self.layers}
        self._original_to = module.to

    @staticmethod
    def _collect_ignored_tensor_ids(ignore_modules: Sequence[torch.nn.Module]) -> set[int]:
        ignored: set[int] = set()
        for item in ignore_modules:
            if isinstance(item, torch.nn.Parameter):
                ignored.add(id(item))
            elif isinstance(item, torch.Tensor):
                ignored.add(id(item))
            elif isinstance(item, torch.nn.Module):
                ignored.update(id(param) for param in item.parameters(recurse=True))
                ignored.update(id(buffer) for buffer in item.buffers(recurse=True))
        return ignored

    @staticmethod
    def _build_layer_candidates(
        layers: Sequence[tuple[str, torch.nn.Module]],
        ignored_tensor_ids: set[int],
    ) -> tuple[list[_LayerCandidate], list[str]]:
        candidates: list[_LayerCandidate] = []
        skipped_layers: list[str] = []

        for name, layer in layers:
            params = [param for param in layer.parameters(recurse=True) if id(param) not in ignored_tensor_ids]
            buffers = [buffer for buffer in layer.buffers(recurse=True) if id(buffer) not in ignored_tensor_ids]
            tensors: list[torch.Tensor] = list(params) + list(buffers)
            if not tensors:
                skipped_layers.append(name)
                continue
            if any(not _is_storage_swap_supported(tensor) for tensor in tensors):
                skipped_layers.append(name)
                continue
            candidates.append(
                _LayerCandidate(
                    name=name,
                    module=layer,
                    params=params,
                    buffers=buffers,
                )
            )

        return candidates, skipped_layers

    @staticmethod
    def _entry_device(
        entry_params: Sequence[torch.nn.Parameter],
        entry_buffers: Sequence[torch.Tensor],
    ) -> Optional[torch.device]:
        tensors: list[torch.Tensor] = list(entry_params) + list(entry_buffers)
        if not tensors:
            return None
        return tensors[0].device

    @classmethod
    def attach(
        cls,
        module: torch.nn.Module,
        device: torch.device,
        offload_fraction: float = 1.0,
        block_paths: Optional[Sequence[str]] = None,
        ignore_modules: Optional[Sequence[torch.nn.Module]] = None,
    ) -> "BlockOffloadManager":
        if hasattr(module, "_block_offload_manager"):
            return module._block_offload_manager

        layers = resolve_block_layers(module, block_paths)
        if not layers:
            path_label = ", ".join(block_paths or DEFAULT_BLOCK_PATHS)
            raise ValueError(f"Block offloading could not find ordered block layers at: {path_label}")

        ignored_tensor_ids = cls._collect_ignored_tensor_ids(ignore_modules or [])
        candidates, skipped_layers = cls._build_layer_candidates(layers, ignored_tensor_ids)
        if not candidates:
            raise ValueError(
                "Block offloading could not find any block layers with storage-swappable tensors. "
                "This usually means the block weights are quantized tensor subclasses; use legacy "
                "layer offloading or disable quantization for block offloading."
            )

        layer_sizes = tuple(_tensors_nbytes(list(candidate.params) + list(candidate.buffers)) for candidate in candidates)
        strategy = LayerOffloadStrategy(layer_sizes, offload_fraction)
        if not strategy.offloaded_indices:
            raise ValueError("Block offloading was requested with a 0% whole-block offload fraction.")

        manager = cls(
            module=module,
            process_device=device,
            layers=candidates,
            strategy=strategy,
            skipped_layers=skipped_layers,
        )
        module._block_offload_manager = manager
        module._block_offload_original_to = module.to
        module._aitk_block_offload_skipped_layers = tuple(skipped_layers)
        module.to = manager.memory_managed_to
        manager._patch_layers()
        if manager.active:
            manager.offload_inactive_layers()
        return manager

    def detach(self):
        for entry in self.layers:
            entry.module.forward = entry.original_forward
            for handle in entry.hook_handles:
                handle.remove()
            entry.hook_handles.clear()

        if hasattr(self.module, "_block_offload_original_to"):
            self.module.to = self.module._block_offload_original_to
            delattr(self.module, "_block_offload_original_to")
        if hasattr(self.module, "_block_offload_manager"):
            delattr(self.module, "_block_offload_manager")
        if hasattr(self.module, "_aitk_block_offload_skipped_layers"):
            delattr(self.module, "_aitk_block_offload_skipped_layers")

    def _patch_layers(self):
        for entry in self.layers:
            if not self.strategy.is_offloaded(entry.index):
                continue

            def wrapped_forward(*args, _entry=entry, **kwargs):
                return self.layer_forward(_entry, *args, **kwargs)

            entry.module.forward = wrapped_forward
            try:
                entry.hook_handles.append(
                    entry.module.register_full_backward_hook(
                        lambda _module, _grad_input, _grad_output, _entry=entry: self.after_backward(_entry)
                    )
                )
            except RuntimeError:
                pass

    def memory_managed_to(self, *args, **kwargs):
        device, dtype = _extract_device_dtype(args, kwargs)
        result = self._original_to(*args, **kwargs)

        if device is not None:
            self.process_device = device
            self.active = self.process_device.type == "cuda" and torch.cuda.is_available()
            if self.active and self.transfer_stream is None:
                self.transfer_stream = torch.cuda.Stream(device=self.process_device)
            elif not self.active:
                self.transfer_stream = None

        if dtype is not None:
            for entry in self.layers:
                self._move_entry(entry, self._entry_target_device(entry), dtype=dtype, async_transfer=False)

        if self.active:
            self.offload_inactive_layers()
        else:
            for entry in self.layers:
                entry.state = "cpu"
        return result

    def _entry_target_device(self, entry: _ManagedLayer) -> torch.device:
        device = self._entry_device(entry.params, entry.buffers)
        return device or torch.device("cpu")

    def _wait_for_entry_transfer(self, entry: _ManagedLayer):
        if entry.transfer_event is None:
            return
        if self.active:
            torch.cuda.current_stream(self.process_device).wait_event(entry.transfer_event)
        entry.transfer_event = None
        if entry.state == "prefetching":
            entry.state = "device"
        elif entry.state == "offloading":
            entry.state = "cpu"

    def offload_inactive_layers(self):
        for entry in self.layers:
            if self.strategy.is_offloaded(entry.index):
                self._offload_entry(entry, async_transfer=False)
            else:
                entry.state = "resident"

    def layer_forward(self, entry: _ManagedLayer, *args, **kwargs):
        self._ensure_entry_on_device(entry)
        self._prefetch_window(self.strategy.forward_forward_window(entry.index))
        result = entry.original_forward(*args, **kwargs)

        if torch.is_grad_enabled() and self._output_requires_grad(result):
            entry.state = "device"
        else:
            self._offload_entry(entry, async_transfer=True)
        return result

    def after_backward(self, entry: _ManagedLayer):
        self._offload_entry(entry, async_transfer=True)
        for next_index in self.strategy.forward_backward_window(entry.index):
            if next_index != entry.index:
                self._prefetch_entry(self.layers[next_index])
        return None

    @staticmethod
    def _output_requires_grad(output: Any) -> bool:
        if isinstance(output, torch.Tensor):
            return output.requires_grad
        if isinstance(output, (list, tuple)):
            return any(BlockOffloadManager._output_requires_grad(item) for item in output)
        if isinstance(output, dict):
            return any(BlockOffloadManager._output_requires_grad(item) for item in output.values())
        return False

    def _prefetch_window(self, indices: Sequence[int]):
        for index in indices:
            if index < 0 or index >= len(self.layers):
                continue
            self._prefetch_entry(self.layers[index])

    def _ensure_entry_on_device(self, entry: _ManagedLayer):
        if not self.active:
            return
        if entry.state == "prefetching":
            self._wait_for_entry_transfer(entry)
            if entry.state == "device":
                return
        if entry.state == "offloading":
            self._wait_for_entry_transfer(entry)
        if entry.state != "device":
            self._move_entry(entry, self.process_device, async_transfer=False)
            entry.state = "device"

    def _prefetch_entry(self, entry: _ManagedLayer):
        if not self.active or entry.state in {"device", "prefetching", "resident"}:
            return
        self._move_entry(entry, self.process_device, async_transfer=True)
        entry.state = "prefetching"

    def _offload_entry(self, entry: _ManagedLayer, async_transfer: bool):
        if not self.active or entry.state in {"cpu", "offloading"}:
            return
        self._move_entry(entry, torch.device("cpu"), async_transfer=async_transfer)
        entry.state = "offloading" if async_transfer and entry.transfer_event is not None else "cpu"

    def _move_entry(
        self,
        entry: _ManagedLayer,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        async_transfer: bool = True,
    ):
        if device.type == "cuda" and not self.active:
            return

        if not async_transfer:
            self._wait_for_entry_transfer(entry)

        stream_context = (
            torch.cuda.stream(self.transfer_stream)
            if async_transfer and self.transfer_stream is not None
            else None
        )
        if stream_context is None:
            self._move_entry_tensors(entry, device, dtype)
        else:
            if device.type == "cpu":
                self.transfer_stream.wait_stream(torch.cuda.current_stream(self.process_device))
            with stream_context:
                self._move_entry_tensors(entry, device, dtype)
                entry.transfer_event = torch.cuda.Event()
                entry.transfer_event.record(self.transfer_stream)

    def _move_entry_tensors(
        self,
        entry: _ManagedLayer,
        device: torch.device,
        dtype: Optional[torch.dtype],
    ):
        for param in entry.params:
            param.data = self._move_tensor(param.data, device, dtype)
            if param.grad is not None:
                param.grad.data = self._move_tensor(param.grad.data, device, dtype)
        for buffer in entry.buffers:
            buffer.data = self._move_tensor(buffer.data, device, dtype)

    @staticmethod
    def _move_tensor(tensor: torch.Tensor, device: torch.device, dtype: Optional[torch.dtype]) -> torch.Tensor:
        if device.type == "cpu":
            return StaticPinnedCpuAllocator.copy(tensor, dtype=dtype)
        return StaticCudaAllocator.copy(tensor, device=device, dtype=dtype)
