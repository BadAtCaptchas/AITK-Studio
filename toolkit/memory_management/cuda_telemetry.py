from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

import torch


CUDA_MEMORY_PHASES = (
    "loading",
    "caching",
    "forward",
    "backward",
    "optimizer",
    "saving",
    "sampling",
)


@dataclass(frozen=True)
class CudaPhaseMemory:
    """Allocator state captured around one or more executions of a phase."""

    calls: int
    allocated_start_bytes: int
    allocated_end_bytes: int
    allocated_peak_bytes: int
    reserved_start_bytes: int
    reserved_end_bytes: int
    reserved_peak_bytes: int


class CudaPhaseTelemetry:
    """Best-effort CUDA allocator telemetry with named, bounded phase scopes.

    The reporter deliberately does not synchronize CUDA. Allocator counters are
    updated when allocations occur, so synchronization would add training-time
    stalls without making these memory measurements more useful. Telemetry must
    never make a CPU job, an unsupported accelerator, or a CUDA stats failure
    fail the workload.
    """

    def __init__(
        self,
        device: Optional[torch.device | str] = None,
        *,
        enabled: bool = True,
        cuda_api: Any = None,
    ) -> None:
        self.device = torch.device(device or "cuda")
        self._cuda = cuda_api if cuda_api is not None else torch.cuda
        try:
            cuda_available = bool(self._cuda.is_available())
        except Exception:
            cuda_available = False
        self.enabled = bool(enabled and self.device.type == "cuda" and cuda_available)
        self._latest: Dict[str, CudaPhaseMemory] = {}
        self._summary: Dict[str, CudaPhaseMemory] = {}
        self._overall_peak_allocated_bytes = 0
        self._overall_peak_reserved_bytes = 0

    @staticmethod
    def _validate_phase(phase: str) -> None:
        if phase not in CUDA_MEMORY_PHASES:
            raise ValueError(
                f"Unknown CUDA memory phase {phase!r}; expected one of {CUDA_MEMORY_PHASES}"
            )

    def _current(self) -> tuple[int, int]:
        return (
            int(self._cuda.memory_allocated(self.device)),
            int(self._cuda.memory_reserved(self.device)),
        )

    def _peaks(self, allocated: int, reserved: int) -> tuple[int, int]:
        peak_allocated = int(self._cuda.max_memory_allocated(self.device))
        max_reserved = getattr(self._cuda, "max_memory_reserved", None)
        peak_reserved = int(max_reserved(self.device)) if callable(max_reserved) else reserved
        return max(allocated, peak_allocated), max(reserved, peak_reserved)

    def _store(self, phase: str, record: CudaPhaseMemory) -> None:
        self._latest[phase] = record
        previous = self._summary.get(phase)
        if previous is None:
            self._summary[phase] = record
            return
        self._summary[phase] = CudaPhaseMemory(
            calls=previous.calls + 1,
            allocated_start_bytes=record.allocated_start_bytes,
            allocated_end_bytes=record.allocated_end_bytes,
            allocated_peak_bytes=max(
                previous.allocated_peak_bytes, record.allocated_peak_bytes
            ),
            reserved_start_bytes=record.reserved_start_bytes,
            reserved_end_bytes=record.reserved_end_bytes,
            reserved_peak_bytes=max(
                previous.reserved_peak_bytes, record.reserved_peak_bytes
            ),
        )

    @contextmanager
    def phase(self, phase: str) -> Iterator[None]:
        """Measure a named phase, yielding a no-op scope when CUDA is unavailable."""

        self._validate_phase(phase)
        if not self.enabled:
            yield
            return

        try:
            allocated_start, reserved_start = self._current()
            previous_peak_allocated, previous_peak_reserved = self._peaks(
                allocated_start, reserved_start
            )
            self._overall_peak_allocated_bytes = max(
                self._overall_peak_allocated_bytes, previous_peak_allocated
            )
            self._overall_peak_reserved_bytes = max(
                self._overall_peak_reserved_bytes, previous_peak_reserved
            )
            self._cuda.reset_peak_memory_stats(self.device)
        except Exception:
            # Some alternative CUDA runtimes expose torch.cuda but not every
            # allocator statistic. Keep the workload functional in that case.
            yield
            return

        try:
            yield
        finally:
            try:
                allocated_end, reserved_end = self._current()
                allocated_peak, reserved_peak = self._peaks(
                    allocated_end, reserved_end
                )
                self._overall_peak_allocated_bytes = max(
                    self._overall_peak_allocated_bytes, allocated_peak
                )
                self._overall_peak_reserved_bytes = max(
                    self._overall_peak_reserved_bytes, reserved_peak
                )
                self._store(
                    phase,
                    CudaPhaseMemory(
                        calls=1,
                        allocated_start_bytes=allocated_start,
                        allocated_end_bytes=allocated_end,
                        allocated_peak_bytes=allocated_peak,
                        reserved_start_bytes=reserved_start,
                        reserved_end_bytes=reserved_end,
                        reserved_peak_bytes=reserved_peak,
                    ),
                )
            except Exception:
                pass

    def latest(self, phase: str) -> Optional[CudaPhaseMemory]:
        self._validate_phase(phase)
        return self._latest.get(phase)

    def report(self) -> Dict[str, Dict[str, int]]:
        """Return aggregate byte counters suitable for logs or JSON reports."""

        return {phase: asdict(record) for phase, record in self._summary.items()}

    def metrics(self, prefix: str = "train/vram") -> Mapping[str, float]:
        """Return aggregate phase counters in GiB for the existing metrics path."""

        if not self.enabled:
            return {}
        gib = float(1024**3)
        metrics: Dict[str, float] = {}
        for phase, record in self._summary.items():
            base = f"{prefix}/{phase}"
            metrics[f"{base}/allocated_gb"] = record.allocated_end_bytes / gib
            metrics[f"{base}/reserved_gb"] = record.reserved_end_bytes / gib
            metrics[f"{base}/peak_allocated_gb"] = record.allocated_peak_bytes / gib
            metrics[f"{base}/peak_reserved_gb"] = record.reserved_peak_bytes / gib
            metrics[f"{base}/calls"] = float(record.calls)
        metrics[f"{prefix}/overall/peak_allocated_gb"] = (
            self._overall_peak_allocated_bytes / gib
        )
        metrics[f"{prefix}/overall/peak_reserved_gb"] = (
            self._overall_peak_reserved_bytes / gib
        )
        return metrics

    @property
    def overall_peak_allocated_bytes(self) -> int:
        return self._overall_peak_allocated_bytes

    @property
    def overall_peak_reserved_bytes(self) -> int:
        return self._overall_peak_reserved_bytes

    def clear(self) -> None:
        self._latest.clear()
        self._summary.clear()
        self._overall_peak_allocated_bytes = 0
        self._overall_peak_reserved_bytes = 0


def record_cuda_phase(phase: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Measure an instance method when it exposes ``cuda_phase_telemetry``."""

    CudaPhaseTelemetry._validate_phase(phase)

    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            reporter = getattr(self, "cuda_phase_telemetry", None)
            if reporter is None:
                return function(self, *args, **kwargs)
            with reporter.phase(phase):
                return function(self, *args, **kwargs)

        return wrapped

    return decorator
