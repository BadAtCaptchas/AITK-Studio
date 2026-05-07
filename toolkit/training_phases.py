import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


PHASE_CONTROL_FIELDS = {"name", "steps", "auto_advance"}

PHASE_OVERRIDE_FIELDS = {
    "adapter_lr",
    "content_or_style",
    "content_or_style_reg",
    "embedding_lr",
    "img_multiplier",
    "latent_multiplier",
    "loss_type",
    "lr",
    "lr_scheduler",
    "lr_scheduler_params",
    "max_denoising_steps",
    "max_grad_norm",
    "min_denoising_steps",
    "min_snr_gamma",
    "noise_multiplier",
    "noise_offset",
    "noisy_latent_multiplier",
    "optimizer",
    "optimizer_params",
    "pred_scaler",
    "prompt_dropout_prob",
    "random_noise_multiplier",
    "random_noise_shift",
    "refiner_lr",
    "reg_weight",
    "snr_gamma",
    "target_noise_multiplier",
    "text_encoder_lr",
    "timestep_type",
    "unet_lr",
}

PHASE_ALLOWED_FIELDS = PHASE_CONTROL_FIELDS | PHASE_OVERRIDE_FIELDS
PHASE_MERGE_FIELDS = {"optimizer_params", "lr_scheduler_params"}


@dataclass
class PlateauState:
    best_value: Optional[float] = None
    stale_windows: int = 0
    windows_seen: int = 0


@dataclass
class AutoAdvanceConfig:
    type: str = "loss_plateau"
    metric: str = "loss/loss"
    mode: str = "min"
    min_steps: Optional[int] = None
    window: int = 100
    patience: int = 2
    min_delta_pct: float = 1.0

    @classmethod
    def from_raw(cls, raw: Optional[Dict[str, Any]]) -> Optional["AutoAdvanceConfig"]:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("train.phases[].auto_advance must be an object")

        config = cls(
            type=str(raw.get("type", "loss_plateau")),
            metric=str(raw.get("metric", "loss/loss")),
            mode=str(raw.get("mode", "min")),
            min_steps=raw.get("min_steps", None),
            window=int(raw.get("window", 100)),
            patience=int(raw.get("patience", 2)),
            min_delta_pct=float(raw.get("min_delta_pct", 1.0)),
        )
        if config.type != "loss_plateau":
            raise ValueError(f"Unsupported phase auto_advance type: {config.type}")
        if config.mode not in {"min", "max"}:
            raise ValueError("train.phases[].auto_advance.mode must be 'min' or 'max'")
        if config.window < 1:
            raise ValueError("train.phases[].auto_advance.window must be at least 1")
        if config.patience < 1:
            raise ValueError("train.phases[].auto_advance.patience must be at least 1")
        if config.min_steps is None:
            config.min_steps = max(200, config.window * 2)
        else:
            config.min_steps = int(config.min_steps)
        if config.min_steps < 1:
            raise ValueError("train.phases[].auto_advance.min_steps must be at least 1")
        if config.min_delta_pct < 0:
            raise ValueError("train.phases[].auto_advance.min_delta_pct must be 0 or greater")
        return config


@dataclass
class TrainingPhase:
    name: str
    steps: int
    overrides: Dict[str, Any]
    auto_advance: Optional[AutoAdvanceConfig]
    planned_start_step: int
    planned_end_step: int


@dataclass
class PhaseAdvanceResult:
    changed: bool = False
    should_stop: bool = False
    reason: str = ""


class TrainingPhaseManager:
    def __init__(self, train_config: Any):
        self.base_config = copy.deepcopy(train_config)
        raw_phases = copy.deepcopy(getattr(train_config, "phases", None) or [])
        self.phases = self._parse_phases(raw_phases)
        self.enabled = len(self.phases) > 0
        self.save_on_phase_change = bool(getattr(train_config, "save_on_phase_change", True))
        self.current_index = 0
        self.current_phase_start_step = 0
        self.current_reason = "initial"
        self.should_stop_training = False
        self.metric_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.plateau_state: Dict[int, PlateauState] = defaultdict(PlateauState)

        if self.enabled:
            total_phase_steps = sum(phase.steps for phase in self.phases)
            if int(getattr(train_config, "steps", 0)) != total_phase_steps:
                raise ValueError(
                    f"train.steps must equal the sum of train.phases[].steps "
                    f"({getattr(train_config, 'steps', None)} != {total_phase_steps})"
                )

    def _parse_phases(self, raw_phases: List[Dict[str, Any]]) -> List[TrainingPhase]:
        if not isinstance(raw_phases, list):
            raise ValueError("train.phases must be a list")

        phases: List[TrainingPhase] = []
        cursor = 0
        for idx, raw_phase in enumerate(raw_phases):
            if not isinstance(raw_phase, dict):
                raise ValueError("train.phases[] entries must be objects")

            unknown_fields = sorted(set(raw_phase.keys()) - PHASE_ALLOWED_FIELDS)
            if unknown_fields:
                raise ValueError(
                    f"Unsupported train.phases[{idx}] fields: {', '.join(unknown_fields)}"
                )

            if "steps" not in raw_phase:
                raise ValueError(f"train.phases[{idx}].steps is required")
            steps = int(raw_phase["steps"])
            if steps < 1:
                raise ValueError(f"train.phases[{idx}].steps must be at least 1")

            name = str(raw_phase.get("name") or f"Phase {idx + 1}")
            overrides = {
                key: copy.deepcopy(value)
                for key, value in raw_phase.items()
                if key in PHASE_OVERRIDE_FIELDS
            }
            for merge_key in PHASE_MERGE_FIELDS:
                if merge_key in overrides and not isinstance(overrides[merge_key], dict):
                    raise ValueError(f"train.phases[{idx}].{merge_key} must be an object")

            phases.append(
                TrainingPhase(
                    name=name,
                    steps=steps,
                    overrides=overrides,
                    auto_advance=AutoAdvanceConfig.from_raw(raw_phase.get("auto_advance")),
                    planned_start_step=cursor,
                    planned_end_step=cursor + steps,
                )
            )
            cursor += steps

        return phases

    @property
    def current_phase(self) -> Optional[TrainingPhase]:
        if not self.enabled:
            return None
        return self.phases[self.current_index]

    def index_for_step(self, step: int) -> int:
        if not self.enabled:
            return 0
        if step >= self.phases[-1].planned_end_step:
            return len(self.phases) - 1
        for idx, phase in enumerate(self.phases):
            if step < phase.planned_end_step:
                return idx
        return len(self.phases) - 1

    def restore_from_training_info(self, training_info: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        raw_index = training_info.get("phase_index", None)
        if raw_index is None:
            self.current_index = self.index_for_step(int(training_info.get("step", 0)))
            self.current_phase_start_step = self.phases[self.current_index].planned_start_step
        else:
            self.current_index = max(0, min(int(raw_index), len(self.phases) - 1))
            self.current_phase_start_step = int(
                training_info.get(
                    "phase_start_step",
                    self.phases[self.current_index].planned_start_step,
                )
            )
        self.current_reason = str(training_info.get("phase_reason", "resume"))

    def apply(self, train_config: Any, step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        if step is not None and self.current_reason == "initial":
            self.current_index = self.index_for_step(int(step))
            self.current_phase_start_step = self.phases[self.current_index].planned_start_step

        for key in PHASE_OVERRIDE_FIELDS:
            if hasattr(self.base_config, key):
                setattr(train_config, key, copy.deepcopy(getattr(self.base_config, key)))

        phase = self.phases[self.current_index]
        for key, value in phase.overrides.items():
            if key in PHASE_MERGE_FIELDS:
                merged = copy.deepcopy(getattr(self.base_config, key, {}) or {})
                merged.update(copy.deepcopy(value))
                setattr(train_config, key, merged)
            else:
                setattr(train_config, key, copy.deepcopy(value))

        base_lr = getattr(self.base_config, "lr", None)
        phase_lr = getattr(train_config, "lr", base_lr)
        for lr_key in ("unet_lr", "text_encoder_lr", "refiner_lr", "embedding_lr", "adapter_lr"):
            if lr_key in phase.overrides:
                continue
            if hasattr(self.base_config, lr_key) and getattr(self.base_config, lr_key) == base_lr:
                setattr(train_config, lr_key, copy.deepcopy(phase_lr))

    def get_phase_local_step(self, global_step: int) -> int:
        if not self.enabled:
            return int(global_step)
        return max(0, int(global_step) - self.current_phase_start_step)

    def metrics_for_step(self, global_step: int) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        phase = self.current_phase
        return {
            "phase/index": self.current_index,
            "phase/step": self.get_phase_local_step(global_step),
            "phase/name": phase.name if phase else "",
            "phase/reason": self.current_reason,
        }

    def training_info(self, global_step: int) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        phase = self.current_phase
        return {
            "phase_index": self.current_index,
            "phase_name": phase.name if phase else "",
            "phase_step": self.get_phase_local_step(global_step),
            "phase_start_step": self.current_phase_start_step,
            "phase_reason": self.current_reason,
        }

    def observe_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)):
                continue
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                continue
            self.metric_history[key].append((int(step), numeric_value))
            if len(self.metric_history[key]) > 10000:
                self.metric_history[key] = self.metric_history[key][-10000:]

    def maybe_advance_after_step(self, completed_step: int) -> PhaseAdvanceResult:
        if not self.enabled or self.should_stop_training:
            return PhaseAdvanceResult()

        next_step = int(completed_step) + 1
        phase = self.current_phase
        if phase is None:
            return PhaseAdvanceResult()

        elapsed = next_step - self.current_phase_start_step
        reason = ""
        if elapsed >= phase.steps:
            reason = "steps"
        elif self._has_plateaued(next_step):
            reason = "loss_plateau"

        if not reason:
            return PhaseAdvanceResult()

        if self.current_index >= len(self.phases) - 1:
            self.should_stop_training = True
            self.current_reason = reason
            return PhaseAdvanceResult(should_stop=True, reason=reason)

        self.current_index += 1
        self.current_phase_start_step = next_step
        self.current_reason = reason
        return PhaseAdvanceResult(changed=True, reason=reason)

    def _has_plateaued(self, next_step: int) -> bool:
        phase = self.current_phase
        if phase is None or phase.auto_advance is None:
            return False

        config = phase.auto_advance
        elapsed = next_step - self.current_phase_start_step
        if elapsed < int(config.min_steps or 1):
            return False

        points = [
            value
            for step, value in self.metric_history.get(config.metric, [])
            if self.current_phase_start_step <= step < next_step
        ]
        window_count = len(points) // config.window
        state = self.plateau_state[self.current_index]
        if window_count <= state.windows_seen:
            return False

        for window_idx in range(state.windows_seen, window_count):
            start = window_idx * config.window
            end = start + config.window
            window_values = points[start:end]
            if len(window_values) < config.window:
                continue
            window_average = sum(window_values) / len(window_values)

            if state.best_value is None:
                state.best_value = window_average
                state.stale_windows = 0
                continue

            if self._is_improved(window_average, state.best_value, config):
                state.best_value = window_average
                state.stale_windows = 0
            else:
                state.stale_windows += 1

        state.windows_seen = window_count
        return state.stale_windows >= config.patience

    def _is_improved(
        self,
        value: float,
        best_value: float,
        config: AutoAdvanceConfig,
    ) -> bool:
        min_delta = abs(best_value) * (config.min_delta_pct / 100.0)
        if config.mode == "min":
            return value < best_value - min_delta
        return value > best_value + min_delta
