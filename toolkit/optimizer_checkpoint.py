import os
import tempfile

import torch


OPTIMIZER_CHECKPOINT_FORMAT_VERSION = 1


class OptimizerCheckpointValidationError(ValueError):
    pass


def _validate_step(step, label):
    if type(step) is not int or step < 0:
        raise OptimizerCheckpointValidationError(f"{label} must be a non-negative integer")
    return step


def build_optimizer_checkpoint(state_dict, step, checkpoint_path=None):
    _validate_step(step, "Optimizer checkpoint step")
    if not isinstance(state_dict, dict):
        raise OptimizerCheckpointValidationError("Optimizer state must be a dictionary")
    return {
        "format_version": OPTIMIZER_CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "model_checkpoint": os.path.basename(checkpoint_path) if checkpoint_path else None,
        "optimizer_state_dict": state_dict,
    }


def invalidate_optimizer_checkpoint(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        return


def atomic_save_optimizer_checkpoint(path, state_dict, step, checkpoint_path=None):
    """Atomically replace optimizer state, invalidating stale state on failure."""
    payload = build_optimizer_checkpoint(state_dict, step, checkpoint_path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    os.close(file_descriptor)

    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    except BaseException as error:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass

        # The model checkpoint has already been written by the caller.  An
        # older optimizer file is now unsafe to pair with it, so remove it.
        try:
            invalidate_optimizer_checkpoint(path)
        except OSError as invalidation_error:
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(f"Could not invalidate stale optimizer state: {invalidation_error}")
        raise


def parse_optimizer_checkpoint(payload, expected_step):
    _validate_step(expected_step, "Expected model checkpoint step")
    if not isinstance(payload, dict):
        raise OptimizerCheckpointValidationError("Optimizer checkpoint is not a dictionary")
    if payload.get("format_version") != OPTIMIZER_CHECKPOINT_FORMAT_VERSION:
        raise OptimizerCheckpointValidationError(
            "Optimizer checkpoint is legacy or has an unsupported format version"
        )

    saved_step = payload.get("step")
    _validate_step(saved_step, "Saved optimizer checkpoint step")
    if saved_step != expected_step:
        raise OptimizerCheckpointValidationError(
            f"Optimizer checkpoint step {saved_step} does not match model checkpoint step {expected_step}"
        )

    state_dict = payload.get("optimizer_state_dict")
    if not isinstance(state_dict, dict):
        raise OptimizerCheckpointValidationError("Optimizer checkpoint has no valid state dictionary")
    if "state" not in state_dict or "param_groups" not in state_dict:
        raise OptimizerCheckpointValidationError("Optimizer state dictionary is incomplete")
    return state_dict


def load_optimizer_checkpoint(path, expected_step):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return parse_optimizer_checkpoint(payload, expected_step)
