import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


PHASE_CONTROL_FIELDS = {"name", "steps"}


def validate_finite_training_config(config: Any) -> None:
    """Reject retired automatic training settings before model loading."""
    values = config if isinstance(config, dict) else vars(config)
    migration = (
        "Auto Learn is no longer supported. Remove auto_train/auto_learn and "
        "auto_advance, then set positive integer train.steps and steps for every "
        "manual phase (train.steps must equal their sum)."
    )
    if values.get('auto_train') or values.get('auto_learn') or 'auto_advance' in values:
        raise ValueError(migration)
    if type(values.get('steps', 1000)) is not int or values.get('steps', 1000) < 1:
        raise ValueError("train.steps must be a positive finite integer")
    phases = values.get('phases')
    if phases is not None and not isinstance(phases, list):
        raise ValueError("train.phases must be a list")
    for index, phase in enumerate(phases or []):
        if not isinstance(phase, dict):
            raise ValueError("train.phases[] entries must be objects")
        if 'auto_advance' in phase:
            raise ValueError(migration)
        if type(phase.get('steps')) is not int or phase['steps'] < 1:
            raise ValueError(f"train.phases[{index}].steps must be a positive finite integer")

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
class TrainingPhase:
    name: str
    steps: int
    overrides: Dict[str, Any]
    planned_start_step: int
    planned_end_step: int


@dataclass
class PhaseAdvanceResult:
    changed: bool = False
    should_stop: bool = False
    reason: str = ""


class TrainingPhaseManager:
    def __init__(self, train_config: Any):
        validate_finite_training_config(train_config)
        self.base_config = copy.deepcopy(train_config)
        raw_phases = copy.deepcopy(getattr(train_config, "phases", None) or [])
        self.phases = self._parse_phases(raw_phases)
        self.enabled = len(self.phases) > 0
        self.save_on_phase_change = bool(getattr(train_config, "save_on_phase_change", True))
        self.current_index = 0
        self.current_phase_start_step = 0
        self.current_reason = "initial"
        self.should_stop_training = False
        if self.enabled:
            total_phase_steps = sum(phase.steps for phase in self.phases)
            if getattr(train_config, "steps", 0) != total_phase_steps:
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

            steps = raw_phase["steps"]

            name = str(raw_phase.get("name") or f"Phase {idx + 1}")
            overrides = {
                key: copy.deepcopy(value)
                for key, value in raw_phase.items()
                if key in PHASE_OVERRIDE_FIELDS
            }
            for merge_key in PHASE_MERGE_FIELDS:
                if merge_key in overrides and not isinstance(overrides[merge_key], dict):
                    raise ValueError(f"train.phases[{idx}].{merge_key} must be an object")

            planned_end_step = cursor + steps
            phases.append(
                TrainingPhase(
                    name=name,
                    steps=steps,
                    overrides=overrides,
                    planned_start_step=cursor,
                    planned_end_step=planned_end_step,
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
        if step >= int(self.phases[-1].planned_end_step or 0):
            return len(self.phases) - 1
        for idx, phase in enumerate(self.phases):
            if phase.planned_end_step is not None and step < phase.planned_end_step:
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

    def current_phase_scheduler_steps(self, fallback_steps: int) -> int:
        phase = self.current_phase
        return phase.steps if phase is not None else max(1, int(fallback_steps))
