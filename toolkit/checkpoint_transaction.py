"""Publish complete checkpoint generations without exposing partial resume state.

Portable files keep their historical names. Resume uses immutable generations in
the same training folder, committed by one directory rename after all writers
have finished. Export links/copies are a convenience, never the commit record.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
import uuid

from safetensors import safe_open


GENERATIONS = ".aitk-checkpoints"
MANIFEST = "checkpoint.json"
FORMAT_VERSION = 1


class InvalidCheckpointError(ValueError):
    """A checkpoint is incomplete, corrupt, or outside its declared generation."""


@dataclass(frozen=True)
class CheckpointGeneration:
    directory: str
    step: int
    created_ns: int
    primary: str
    files: tuple[str, ...]
    model_files: tuple[str, ...]
    optimizer: str | None


def _is_link(path):
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise InvalidCheckpointError("Invalid checkpoint member path")
    candidate = Path(root) / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise InvalidCheckpointError("Checkpoint member escapes its generation")
    resolved_root = Path(root).resolve()
    if not candidate.resolve().is_relative_to(resolved_root):
        raise InvalidCheckpointError("Checkpoint member escapes its generation")
    current = candidate
    while current != Path(root):
        if _is_link(current):
            raise InvalidCheckpointError("Checkpoint members cannot be links to other paths")
        current = current.parent
    return candidate


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_safetensors(state_dict, path, metadata=None):
    """Keep an existing portable file intact when its serializer fails."""
    from safetensors.torch import save_file
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-write-", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        save_file(state_dict, temporary, metadata=metadata)
        validate_safetensors(temporary)
        _sync_file(temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def validate_safetensors(path):
    """Check the format/complete tensor extent without loading tensors into RAM."""
    try:
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            if not checkpoint.keys():
                raise InvalidCheckpointError("Checkpoint contains no tensors")
            return checkpoint.metadata() or {}
    except (OSError, ValueError, RuntimeError) as error:
        raise InvalidCheckpointError(f"Invalid checkpoint {path}: {error}") from error
    except Exception as error:
        # SafetensorError is a native exception, not a ValueError on all versions.
        raise InvalidCheckpointError(f"Invalid checkpoint {path}: {error}") from error


def checkpoint_step(path):
    """Prefer recorded progress; file timestamps never determine training order."""
    metadata = {}
    if os.path.isdir(path):
        metadata_path = Path(path) / "aitk_meta.yaml"
        if metadata_path.is_file():
            metadata = _read_metadata(metadata_path)
    else:
        metadata = validate_safetensors(path)
    info = metadata.get("training_info") if isinstance(metadata, dict) else None
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except ValueError as error:
            raise InvalidCheckpointError("Invalid checkpoint training metadata") from error
    if isinstance(info, dict) and "step" in info:
        step = info["step"]
        if type(step) is not int or step < 0:
            raise InvalidCheckpointError("Invalid checkpoint training step")
        return step
    match = re.search(r"_(\d+)(?:_(?:high|low)_noise)?(?:\.safetensors)?$", os.path.basename(path))
    return int(match[1]) if match else 0


def is_managed_export(path):
    """Portable exports are not an alternate commit when a generation is corrupt."""
    if os.path.isdir(path):
        metadata_path = Path(path) / 'aitk_meta.yaml'
        if not metadata_path.exists():
            return False
        metadata = _read_metadata(metadata_path)
    else:
        metadata = validate_safetensors(path)
    return isinstance(metadata, dict) and bool(metadata.get('aitk_checkpoint_generation'))


def _read_metadata(path):
    import yaml
    try:
        with path.open(encoding='utf-8') as handle:
            return yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as error:
        raise InvalidCheckpointError(f'Invalid checkpoint metadata {path}: {error}') from error


def validate_legacy_checkpoint(path, required_stages=()):
    if not os.path.exists(path):
        raise InvalidCheckpointError(f"Checkpoint does not exist: {path}")
    if os.path.isdir(path):
        tensors = [member for member in Path(path).rglob('*') if member.is_file()
                   and member.suffix in ('.safetensors', '.bin', '.pt') and member.name != 'optimizer.pt']
        if not tensors:
            raise InvalidCheckpointError(f"Checkpoint directory has no model weights: {path}")
        for tensor in tensors:
            _inside(path, tensor.relative_to(path).as_posix())
            if tensor.suffix == '.safetensors':
                validate_safetensors(tensor)
            else:
                _validate_torch_weights(tensor)
        for index_path in Path(path).rglob('*.index.json'):
            with index_path.open(encoding='utf-8') as handle:
                index = json.load(handle)
            weight_map = index.get('weight_map') if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise InvalidCheckpointError(f'Invalid checkpoint shard index: {index_path}')
            for member in weight_map.values():
                shard = _inside(index_path.parent, member)
                if not shard.is_file() or shard not in tensors:
                    raise InvalidCheckpointError(f'Missing checkpoint shard: {member}')
    else:
        if Path(path).suffix in ('.pt', '.bin'):
            _validate_torch_weights(path)
            match = re.search(r'_(\d+)\.(?:pt|bin)$', str(path))
            return int(match[1]) if match else 0
        validate_safetensors(path)
        match = re.search(r"_(?:high|low)_noise\.safetensors$", str(path))
        if match:
            base = str(path)[:match.start()]
            for stage in required_stages:
                validate_safetensors(f"{base}_{stage}_noise.safetensors")
    return checkpoint_step(path)


def _validate_torch_weights(path):
    import torch
    def contains_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(contains_tensor(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_tensor(item) for item in value)
        return False
    try:
        try:
            state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        except RuntimeError as error:
            if 'mmap' not in str(error):
                raise
            state = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or not contains_tensor(state):
            raise InvalidCheckpointError(f'Checkpoint has no tensor state: {path}')
    except Exception as error:
        raise InvalidCheckpointError(f'Invalid model weights {path}: {error}') from error


def begin_checkpoint(save_root):
    root = Path(save_root).resolve()
    generations = root / GENERATIONS
    if _is_link(generations):
        raise InvalidCheckpointError("Checkpoint generation folder cannot be a link")
    generations.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix=".tmp-", dir=generations)


def _sync_file(path):
    with open(path, "r+b") as handle:
        os.fsync(handle.fileno())


def _sync_directory(path):
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def commit_checkpoint(staging, primary, step):
    if type(step) is not int or step < 0:
        raise InvalidCheckpointError("Checkpoint step must be a non-negative integer")
    staging = Path(staging)
    primary_relative = Path(primary).relative_to(staging).as_posix()
    if not Path(primary).exists():
        raise InvalidCheckpointError("Checkpoint exporter did not create its primary output")
    validate_legacy_checkpoint(primary)
    members = []
    for path in sorted(staging.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(staging).as_posix()
        _inside(staging, relative)
        if path.suffix == ".safetensors":
            validate_safetensors(path)
        _sync_file(path)
        members.append({"path": relative, "size": path.stat().st_size, "sha256": file_digest(path)})
    model_files = [item["path"] for item in members if item["path"] != "optimizer.pt"]
    if not model_files:
        raise InvalidCheckpointError("Checkpoint has no model artifacts")
    payload = {"version": FORMAT_VERSION, "step": step, "created_ns": time.time_ns(),
               "primary": primary_relative, "files": members, "model_files": model_files,
               "optimizer": "optimizer.pt" if (staging / "optimizer.pt").is_file() else None}
    with (staging / MANIFEST).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(staging)
    destination = staging.parent / f"{step:09d}-{uuid.uuid4().hex}"
    os.replace(staging, destination)
    _sync_directory(destination.parent)
    return read_generation(destination)


def read_generation(directory, verify_hashes=True):
    directory = Path(directory)
    try:
        if _is_link(directory) or _is_link(directory.parent):
            raise InvalidCheckpointError("Checkpoint generation cannot be a link")
        # Normalize Windows short names only after rejecting linked generations.
        directory = directory.resolve()
        with (directory / MANIFEST).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("version") != FORMAT_VERSION:
            raise InvalidCheckpointError("Unsupported checkpoint manifest")
        step, created = payload.get("step"), payload.get("created_ns")
        if type(step) is not int or step < 0 or type(created) is not int or created < 0:
            raise InvalidCheckpointError("Invalid checkpoint progress")
        entries = payload.get("files")
        if not isinstance(entries, list) or not entries:
            raise InvalidCheckpointError("Checkpoint manifest has no members")
        files = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise InvalidCheckpointError("Invalid checkpoint member")
            relative = entry.get("path")
            path = _inside(directory, relative)
            digest = entry.get('sha256')
            if type(entry.get('size')) is not int or entry['size'] < 0 or not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
                raise InvalidCheckpointError('Invalid checkpoint member size or checksum')
            if relative in files or not path.is_file() or path.stat().st_size != entry.get("size"):
                raise InvalidCheckpointError(f"Incomplete checkpoint member: {relative}")
            if verify_hashes and file_digest(path) != entry.get("sha256"):
                raise InvalidCheckpointError(f"Checkpoint checksum mismatch: {relative}")
            if path.suffix == ".safetensors":
                validate_safetensors(path)
            files.append(relative)
        models = payload.get("model_files")
        if not isinstance(models, list) or not models or any(item not in files for item in models):
            raise InvalidCheckpointError("Invalid model artifact set")
        primary = _inside(directory, payload.get("primary"))
        if not primary.exists():
            raise InvalidCheckpointError("Missing primary checkpoint")
        primary_relative = primary.relative_to(directory).as_posix()
        if primary.is_file() and primary_relative not in models:
            raise InvalidCheckpointError('Primary checkpoint is not a verified model member')
        if primary.is_dir() and not any(item.startswith(primary_relative + '/') for item in models):
            raise InvalidCheckpointError('Primary checkpoint directory has no verified model members')
        actual_members = {member.relative_to(directory).as_posix() for member in directory.rglob('*')
                          if member.is_file() and member != directory / MANIFEST}
        if actual_members != set(files):
            raise InvalidCheckpointError('Checkpoint contains unlisted or missing members')
        if verify_hashes:
            validate_legacy_checkpoint(primary)
        optimizer = payload.get("optimizer")
        if optimizer is not None and (optimizer != "optimizer.pt" or optimizer not in files):
            raise InvalidCheckpointError("Invalid optimizer artifact")
        return CheckpointGeneration(str(directory), step, created, str(primary),
            tuple(str(directory / item) for item in files),
            tuple(str(directory / item) for item in models),
            str(directory / optimizer) if optimizer else None)
    except (OSError, ValueError, TypeError, KeyError) as error:
        if isinstance(error, InvalidCheckpointError):
            raise
        raise InvalidCheckpointError(f"Invalid checkpoint generation {directory}: {error}") from error


def generation_for_path(path, verify_hashes=True):
    current = Path(path).absolute()
    for parent in (current, *current.parents):
        if parent.parent.name == GENERATIONS and not parent.name.startswith(".tmp-"):
            return read_generation(parent, verify_hashes=verify_hashes)
    return None


def committed_generations(save_root, warn=print, verify_hashes=True):
    root = Path(save_root) / GENERATIONS
    if _is_link(root):
        raise InvalidCheckpointError('Checkpoint generation folder cannot be a link')
    if not root.exists():
        return [], False
    valid = []
    had_committed = False
    for directory in root.iterdir():
        if directory.name.startswith(".tmp-"):
            continue
        had_committed = True
        try:
            valid.append(read_generation(directory, verify_hashes=verify_hashes))
        except InvalidCheckpointError as error:
            warn(f"Skipping invalid checkpoint: {error}")
    return valid, had_committed


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def publish_exports(generation, save_root, warn=print):
    """Refresh portable exports after commit; an export failure cannot undo it."""
    source_root = Path(generation.directory)
    for source in source_root.iterdir():
        if source.name == MANIFEST:
            continue
        destination = Path(save_root) / source.name
        temporary = Path(save_root) / f".checkpoint-export-{uuid.uuid4().hex}"
        backup = None
        try:
            if source.is_dir():
                shutil.copytree(source, temporary, copy_function=_link_or_copy)
                if destination.exists():
                    backup = Path(save_root) / f".checkpoint-old-export-{uuid.uuid4().hex}"
                    os.replace(destination, backup)
            else:
                _link_or_copy(source, temporary)
            os.replace(temporary, destination)
        except OSError as error:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            warn(f"Checkpoint committed, but portable export could not be refreshed: {destination}: {error}")
        finally:
            for leftover in (temporary, backup):
                if leftover is not None and leftover.exists():
                    try:
                        if leftover.is_dir():
                            shutil.rmtree(leftover)
                        else:
                            leftover.unlink()
                    except OSError as error:
                        warn(f"Could not clean checkpoint export temporary file: {error}")


def prune_generations(save_root, keep, warn=print, current_generation=None):
    if type(keep) is not int or keep < 1:
        return
    try:
        generations, _ = committed_generations(save_root, warn, verify_hashes=current_generation is None)
    except OSError as error:
        warn(f'Could not inspect old checkpoint generations; training continues: {error}')
        return
    generations.sort(key=lambda item: (item.step, item.created_ns), reverse=True)
    retained = []
    if current_generation is not None:
        retained.append(os.path.normcase(os.path.realpath(current_generation.directory)))
    for generation in generations:
        if len(retained) >= keep:
            break
        if os.path.normcase(os.path.realpath(generation.directory)) in retained:
            continue
        try:
            read_generation(generation.directory)
            retained.append(os.path.normcase(os.path.realpath(generation.directory)))
        except InvalidCheckpointError as error:
            warn(f'Ignoring invalid generation during retention: {error}')
    for generation in generations:
        if os.path.normcase(os.path.realpath(generation.directory)) in retained:
            continue
        try:
            shutil.rmtree(generation.directory)
        except OSError as error:
            warn(f"Could not remove old checkpoint generation; training continues: {error}")
