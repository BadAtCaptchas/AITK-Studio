from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence

import torch
from tqdm import tqdm

from toolkit.basic import flush


def _as_device(value) -> torch.device:
    if isinstance(value, torch.device):
        return value
    try:
        return torch.device(value)
    except Exception:
        return torch.device("cpu")


def _module_device(module) -> torch.device:
    if module is None:
        return torch.device("cpu")
    device = getattr(module, "device", None)
    if device is not None:
        return _as_device(device)
    try:
        first = next(module.parameters())
        return first.device
    except Exception:
        pass
    try:
        first = next(module.buffers())
        return first.device
    except Exception:
        return torch.device("cpu")


def _module_nbytes(module) -> int:
    if module is None:
        return 0
    total = 0
    try:
        for tensor in list(module.parameters()) + list(module.buffers()):
            total += int(tensor.numel() * tensor.element_size())
    except Exception:
        return 0
    return total


@dataclass
class _ModuleState:
    module: object
    device: torch.device
    training: Optional[bool]
    requires_grad: list[bool]


class SampleMemoryCoordinator:
    """Phase-based module residency for native low-VRAM sampling."""

    def __init__(
        self,
        owner,
        *,
        device: Optional[torch.device] = None,
        offload_device: torch.device | str = "cpu",
        status_callback=None,
    ):
        self.owner = owner
        self.device = _as_device(device or getattr(owner, "device_torch", "cpu"))
        self.offload_device = _as_device(offload_device)
        self.status_callback = status_callback
        self._states: dict[int, _ModuleState] = {}
        self._active_components: tuple[str, ...] = ()

    def __enter__(self) -> "SampleMemoryCoordinator":
        self.capture()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()

    @property
    def active_components(self) -> tuple[str, ...]:
        return self._active_components

    def capture(self) -> None:
        if self._states:
            return
        for module in self.iter_known_modules():
            self._states[id(module)] = _ModuleState(
                module=module,
                device=_module_device(module),
                training=getattr(module, "training", None),
                requires_grad=self._requires_grad_state(module),
            )

    def restore(self) -> None:
        for state in reversed(list(self._states.values())):
            self._move_module(state.module, state.device)
            if state.training is not None:
                try:
                    state.module.train(state.training)
                except Exception:
                    pass
            self._restore_requires_grad(state.module, state.requires_grad)
        self._states.clear()
        self._active_components = ()
        flush()

    def iter_known_modules(self) -> Iterator[object]:
        seen: set[int] = set()
        for name in (
            "model",
            "unet",
            "transformer",
            "vae",
            "adapter",
            "refiner_unet",
            "unconditional_transformer",
        ):
            yield from self._dedupe(self._component_modules(name), seen)
        yield from self._dedupe(self._component_modules("text_encoder"), seen)

    @contextmanager
    def phase(
        self,
        name: str,
        active_components: Sequence[str] | None = None,
        *,
        message: Optional[str] = None,
    ):
        self.activate(active_components or (), phase_name=name, message=message)
        try:
            yield self
        finally:
            flush()

    def activate(
        self,
        active_components: Sequence[str],
        *,
        phase_name: str,
        message: Optional[str] = None,
    ) -> None:
        self.capture()
        active = tuple(active_components)
        active_ids: set[int] = set()
        for component in active:
            for module in self._component_modules(component):
                if module is not None:
                    active_ids.add(id(module))

        if message:
            self._status(message)
        elif active != self._active_components:
            self._status(f"Low-VRAM sample: {phase_name}")

        for module in self.iter_known_modules():
            target = self.device if id(module) in active_ids else self.offload_device
            self._move_module(module, target)
        self._active_components = active
        flush()

    def offload_all(self, phase_name: str = "idle") -> None:
        self.activate((), phase_name=phase_name)

    def module_nbytes(self, component: str) -> int:
        return sum(_module_nbytes(module) for module in self._component_modules(component))

    def free_device_bytes(self) -> Optional[int]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
            return int(free_bytes)
        except Exception:
            return None

    def _component_modules(self, component: str) -> list[object]:
        owner = self.owner
        if component == "text_encoder":
            encoders = getattr(owner, "text_encoder", None)
            if encoders is None:
                return []
            if isinstance(encoders, (list, tuple)):
                return [encoder for encoder in encoders if encoder is not None]
            return [encoders]
        if component in {"unet", "transformer"}:
            module = getattr(owner, "model", None)
        else:
            module = getattr(owner, component, None)
        return [] if module is None else [module]

    @staticmethod
    def _dedupe(modules: Iterable[object], seen: set[int]) -> Iterator[object]:
        for module in modules:
            if module is None:
                continue
            key = id(module)
            if key in seen:
                continue
            seen.add(key)
            yield module

    @staticmethod
    def _requires_grad_state(module) -> list[bool]:
        try:
            return [bool(param.requires_grad) for param in module.parameters()]
        except Exception:
            return []

    @staticmethod
    def _restore_requires_grad(module, state: list[bool]) -> None:
        if not state:
            return
        try:
            for param, requires_grad in zip(module.parameters(), state):
                param.requires_grad_(requires_grad)
        except Exception:
            pass

    @staticmethod
    def _move_module(module, device: torch.device) -> None:
        if module is None:
            return
        block_manager = getattr(module, "_block_offload_manager", None)
        if block_manager is not None:
            if device.type == "cuda":
                activate = getattr(block_manager, "activate_for_forward", None)
                if activate is not None:
                    activate(device)
                    return
            elif device.type == "cpu":
                deactivate = getattr(block_manager, "deactivate_to_cpu", None)
                if deactivate is not None:
                    deactivate()
                    return
        if _module_device(module) == device:
            return
        try:
            module.to(device)
        except TypeError:
            module.to(str(device))

    def _status(self, message: str) -> None:
        text = str(message)
        status_update = getattr(self.owner, "_status_update", None)
        has_owner_status = callable(status_update)
        if self.status_callback is None and not has_owner_status:
            return

        wrote_to_tqdm = False
        try:
            tqdm.write(text)
            wrote_to_tqdm = True
        except Exception:
            pass

        if has_owner_status:
            try:
                status_update(text)
                return
            except Exception:
                pass

        if self.status_callback is None:
            return
        try:
            self.status_callback(text if wrote_to_tqdm else f"\n{text}")
        except Exception:
            pass
