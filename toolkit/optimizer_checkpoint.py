import hashlib
import os
import tempfile

import torch


OPTIMIZER_CHECKPOINT_FORMAT_VERSION = 1
RAW_ADAPTER_CHECKPOINT_FORMAT_VERSION = 2
EMA_CHECKPOINT_FORMAT_VERSION = 3


class OptimizerCheckpointValidationError(ValueError):
    pass


def _validate_step(step, label):
    if type(step) is not int or step < 0:
        raise OptimizerCheckpointValidationError(f"{label} must be a non-negative integer")
    return step


def _checkpoint_digest(path):
    if not path or not os.path.exists(path):
        raise OptimizerCheckpointValidationError("Resume state requires a model checkpoint file or directory")
    digest = hashlib.sha256()
    if os.path.isdir(path):
        files = sorted(os.path.join(root, name) for root, _, names in os.walk(path) for name in names)
        if not files:
            raise OptimizerCheckpointValidationError("Model checkpoint directory is empty")
    else:
        files = [path]
    for filename in files:
        if os.path.isdir(path):
            digest.update(os.path.relpath(filename, path).replace(os.sep, '/').encode('utf-8'))
            digest.update(b'\0')
            digest.update(str(os.path.getsize(filename)).encode('ascii'))
            digest.update(b'\0')
        with open(filename, "rb") as checkpoint:
            while chunk := checkpoint.read(4 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_set_digest(paths):
    if not paths:
        raise OptimizerCheckpointValidationError("Checkpoint artifact set is empty")
    paths = [os.path.abspath(path) for path in paths]
    root = os.path.commonpath([os.path.dirname(path) for path in paths])
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(os.path.relpath(path, root).replace(os.sep, '/').encode('utf-8'))
        digest.update(b'\0')
        digest.update(_checkpoint_digest(path).encode('ascii'))
        digest.update(b'\0')
    return digest.hexdigest()


def build_optimizer_checkpoint(state_dict, step, checkpoint_path=None, adapter_resume_state=None, ema_resume_state=None, *, checkpoint_paths=None, trainer_resume_state=None):
    _validate_step(step, "Optimizer checkpoint step")
    if not isinstance(state_dict, dict):
        raise OptimizerCheckpointValidationError("Optimizer state must be a dictionary")
    payload = {
        "format_version": OPTIMIZER_CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "model_checkpoint": os.path.basename(checkpoint_path) if checkpoint_path else None,
        "optimizer_state_dict": state_dict,
    }
    if adapter_resume_state is not None:
        payload["format_version"] = RAW_ADAPTER_CHECKPOINT_FORMAT_VERSION
        payload["adapter_resume_state"] = adapter_resume_state
        payload["model_checkpoint_sha256"] = _checkpoint_digest(checkpoint_path)
    if ema_resume_state is not None:
        payload["format_version"] = EMA_CHECKPOINT_FORMAT_VERSION
        payload["ema_resume_state"] = ema_resume_state
        if "model_checkpoint_sha256" not in payload:
            payload["model_checkpoint_sha256"] = _checkpoint_digest(checkpoint_path)
    if checkpoint_paths is not None:
        payload['model_checkpoint_set_sha256'] = _checkpoint_set_digest(checkpoint_paths)
    if trainer_resume_state is not None:
        payload['trainer_resume_state'] = trainer_resume_state
    return payload


def invalidate_optimizer_checkpoint(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        return


def atomic_save_optimizer_checkpoint(path, state_dict, step, checkpoint_path=None, adapter_resume_state=None, ema_resume_state=None, *, checkpoint_paths=None, trainer_resume_state=None):
    """Atomically replace optimizer state, invalidating stale state on failure."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    os.close(file_descriptor)

    try:
        payload = build_optimizer_checkpoint(state_dict, step, checkpoint_path, adapter_resume_state, ema_resume_state,
                                             checkpoint_paths=checkpoint_paths, trainer_resume_state=trainer_resume_state)
        torch.save(payload, temp_path)
        with open(temp_path, 'r+b') as handle:
            os.fsync(handle.fileno())
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
    version = payload.get("format_version")
    if type(version) is not int or version not in (
        OPTIMIZER_CHECKPOINT_FORMAT_VERSION, RAW_ADAPTER_CHECKPOINT_FORMAT_VERSION, EMA_CHECKPOINT_FORMAT_VERSION,
    ):
        raise OptimizerCheckpointValidationError(
            "Optimizer checkpoint is legacy or has an unsupported format version"
        )
    if version == RAW_ADAPTER_CHECKPOINT_FORMAT_VERSION or payload.get("adapter_resume_state") is not None:
        raw_state = payload.get("adapter_resume_state")
        if not isinstance(raw_state, dict) or raw_state.get("format_version") != 1:
            raise OptimizerCheckpointValidationError("Optimizer checkpoint has no valid raw adapter state")
    if version == EMA_CHECKPOINT_FORMAT_VERSION:
        ema_state = payload.get("ema_resume_state")
        if not isinstance(ema_state, dict) or ema_state.get("format_version") != 1:
            raise OptimizerCheckpointValidationError("Optimizer checkpoint has no valid EMA resume state")
    if version in (RAW_ADAPTER_CHECKPOINT_FORMAT_VERSION, EMA_CHECKPOINT_FORMAT_VERSION):
        digest = payload.get("model_checkpoint_sha256")
        if (
            not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise OptimizerCheckpointValidationError("Optimizer checkpoint has no valid adapter file digest")

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
    state_dict = parse_optimizer_checkpoint(payload, expected_step)
    if payload.get("adapter_resume_state") is not None or payload.get("ema_resume_state") is not None:
        raise OptimizerCheckpointValidationError(
            "Optimizer checkpoint requires live training parameters; use "
            "load_optimizer_checkpoint_for_resume"
        )
    return state_dict


def load_optimizer_checkpoint_for_resume(path, expected_step, checkpoint_path=None, *, include_ema_state=False, checkpoint_paths=None, include_trainer_state=False):
    """Read moments and optional adapter/EMA state bound to the exported model.

    The digest accepts renamed, identical checkpoints while rejecting a
    different adapter at the same step. Legacy payloads still load normally;
    the network decides whether their parameterization can safely resume.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = parse_optimizer_checkpoint(payload, expected_step)
    if payload.get('model_checkpoint_set_sha256') is not None:
        if checkpoint_paths is None:
            from toolkit.checkpoint_transaction import generation_for_path
            generation = generation_for_path(checkpoint_path) if checkpoint_path else None
            if generation is not None:
                checkpoint_paths = generation.model_files
            elif checkpoint_path is not None:
                checkpoint_paths = ([os.path.join(root, name) for root, _, names in os.walk(checkpoint_path) for name in names]
                                    if os.path.isdir(checkpoint_path) else [checkpoint_path])
        if payload['model_checkpoint_set_sha256'] != _checkpoint_set_digest(checkpoint_paths):
            raise OptimizerCheckpointValidationError("Optimizer state does not match the complete checkpoint artifact set")
        checkpoint_absolute = os.path.abspath(checkpoint_path) if checkpoint_path else None
        if checkpoint_absolute is None or not any(
                os.path.abspath(member) == checkpoint_absolute or
                (os.path.isdir(checkpoint_absolute) and os.path.commonpath([checkpoint_absolute, os.path.abspath(member)]) == checkpoint_absolute)
                for member in checkpoint_paths):
            raise OptimizerCheckpointValidationError('Loaded checkpoint is not a member of the saved artifact set')
    adapter_state = payload.get("adapter_resume_state")
    ema_state = payload.get("ema_resume_state")
    if (adapter_state is not None or ema_state is not None) and payload.get('model_checkpoint_set_sha256') is None:
        # Legacy checkpoints bind raw/EMA state to one file. A complete set digest
        # already binds every split-stage member, whichever member initiated load.
        if payload.get("model_checkpoint_sha256") != _checkpoint_digest(checkpoint_path):
            raise OptimizerCheckpointValidationError(
                "Training resume state does not match the loaded model checkpoint"
            )
    if include_ema_state:
        if include_trainer_state:
            return state_dict, adapter_state, ema_state, payload.get('trainer_resume_state')
        return state_dict, adapter_state, ema_state
    if ema_state is not None:
        raise OptimizerCheckpointValidationError("EMA resume state requires include_ema_state=True")
    return state_dict, adapter_state
