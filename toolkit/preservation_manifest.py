"""Snapshot selected preservation data and verify it before training or resume."""
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from toolkit.continuation_assets import pin_json

MANIFEST_NAME = '.aitk-preservation.json'
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')


def _dataset_root(folder):
    try:
        root = Path(folder).resolve(strict=True)
        if root.is_dir():
            return root
    except OSError as error:
        raise ValueError('Preservation dataset is missing or inaccessible. Select an available dataset.') from error
    raise ValueError('Preservation dataset must be an image folder')


def _asset(root, name, field):
    if (not isinstance(name, str) or not name or any(c in name for c in ('\\', '\0', ':'))
            or PurePosixPath(name).is_absolute() or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError('Unsafe preservation asset name')
    try:
        file = (root / name).resolve(strict=True)
        limit = 64 * 1024 ** 2 if field == 'image' else 512 * 1024
        if not file.is_relative_to(root) or not file.is_file() or file.stat().st_size > limit:
            raise ValueError('Invalid preservation asset path or size')
        return file
    except OSError as error:
        message = (f"Preservation image '{name.removesuffix('.txt')}' needs a matching .txt caption."
                   if field == 'caption' else f"Preservation image '{name}' is missing or inaccessible.")
        raise ValueError(message) from error


def _images(root):
    images = []
    for file in root.rglob('*'):
        # A recursive loader must see the same tree as this snapshot. Do not
        # follow linked directories into another dataset or omit their images.
        if file.is_dir() and file.resolve() != file:
            raise ValueError('Preservation datasets cannot contain linked subfolders')
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS + ('.bmp', '.jxl'):
            images.append(file.relative_to(root).as_posix())
            if len(images) > 512:
                raise ValueError('Preservation requires 1-512 selected image-caption pairs')
    return sorted(images)


def _checksum(file):
    digest = hashlib.sha256()
    with file.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(root, manifest, source_sha256):
    if not isinstance(manifest, dict) or manifest.get('version') != 1 or manifest.get('source_sha256') != source_sha256:
        raise ValueError('Preservation manifest does not match the source LoRA')
    examples = manifest.get('examples')
    if not isinstance(examples, list) or not 1 <= len(examples) <= 512:
        raise ValueError('Preservation manifest requires 1-512 curated examples')
    seen, images, total = set(), set(), 0
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError('Invalid preservation example')
        for field in ('image', 'caption'):
            name, digest = example.get(field), example.get(field + '_sha256')
            file = _asset(root, name, field)
            if name in seen:
                raise ValueError('Duplicate preservation asset name')
            if not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest):
                raise ValueError('Missing preservation asset checksum')
            seen.add(name)
            total += file.stat().st_size
            if total > 2 * 1024 ** 3:
                raise ValueError('Preservation dataset exceeds the 2 GiB size limit')
            if field == 'image':
                if file.suffix.lower() not in IMAGE_EXTENSIONS:
                    raise ValueError('Preservation requires PNG, JPEG or WebP image examples')
                images.add(name)
            elif name != str(PurePosixPath(example['image']).with_suffix('.txt')):
                raise ValueError('Preservation captions must match their images')
            if _checksum(file) != digest:
                raise ValueError('Preservation examples changed after curation or selection; create a new job for changed data')
    if set(_images(root)) != images:
        raise ValueError('Preservation dataset contains uncurated or missing images')


def validate_preservation_manifest(folder, source_sha256, manifest_path=None):
    root = _dataset_root(folder)
    manifest_path = Path(manifest_path) if manifest_path is not None else root / MANIFEST_NAME
    try:
        if (manifest_path.resolve(strict=True).parent != manifest_path.parent.resolve(strict=True)
                or not manifest_path.is_file() or manifest_path.stat().st_size > 4 * 1024 ** 2):
            raise ValueError('Invalid preservation manifest path or size')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except OSError as error:
        raise ValueError('Preservation manifest is missing or inaccessible. Restore the job snapshot or select preservation data for a new job.') from error
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError('Preservation manifest is corrupt. Restore the original manifest; it cannot be recreated on resume.') from error
    _validate(root, manifest, source_sha256)
    return manifest_path


def prepare_preservation_manifest(folder, source_sha256, snapshot_path, previous_snapshot=None):
    """Use curated provenance, or pin a user-selected image/caption set once.

    Manual selections are not claimed to be source-generated samples. Their
    manifest belongs to the job, so the dataset remains untouched/read-only.
    A resume never regenerates metadata from possibly changed training data.
    """
    root = _dataset_root(folder)
    curated = root / MANIFEST_NAME
    snapshot_path = Path(snapshot_path)
    if curated.exists() or curated.is_symlink():
        return validate_preservation_manifest(root, source_sha256, curated)
    if previous_snapshot is not None:
        previous_snapshot = Path(previous_snapshot)
        candidate = previous_snapshot if previous_snapshot.exists() else snapshot_path
        return validate_preservation_manifest(root, source_sha256, candidate)
    if snapshot_path.exists() or snapshot_path.is_symlink():
        return validate_preservation_manifest(root, source_sha256, snapshot_path)

    images = _images(root)
    if not images:
        raise ValueError('Preservation requires 1-512 selected image-caption pairs')
    examples, total = [], 0
    for image in images:
        caption = str(PurePosixPath(image).with_suffix('.txt'))
        image_path, caption_path = _asset(root, image, 'image'), _asset(root, caption, 'caption')
        total += image_path.stat().st_size + caption_path.stat().st_size
        if total > 2 * 1024 ** 3:
            raise ValueError('Preservation dataset exceeds the 2 GiB size limit')
        try:
            if not caption_path.read_text(encoding='utf-8-sig').strip():
                raise ValueError(f"Preservation caption '{caption}' must describe behavior to preserve; it cannot be blank.")
        except UnicodeError as error:
            raise ValueError(f"Preservation caption '{caption}' must be UTF-8 text.") from error
        examples.append(dict(image=image, caption=caption, image_sha256=_checksum(image_path), caption_sha256=_checksum(caption_path)))
    manifest = dict(version=1, origin='user_selected', source_sha256=source_sha256, examples=examples)
    _validate(root, manifest, source_sha256)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    pin_json(snapshot_path, manifest, 'Preservation selection changed during setup; create a new job for changed data')
    return validate_preservation_manifest(root, source_sha256, snapshot_path)
