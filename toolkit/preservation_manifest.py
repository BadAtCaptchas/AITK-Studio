"""Validate curated replay manifests before any examples reach the trainer."""
import hashlib
import json
import re
from pathlib import Path


def validate_preservation_manifest(folder, source_sha256):
    root = Path(folder).resolve(strict=True)
    manifest_path = root / '.aitk-preservation.json'
    if manifest_path.resolve(strict=True).parent != root or manifest_path.stat().st_size > 4 * 1024 ** 2:
        raise ValueError('Invalid preservation manifest path or size')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or manifest.get('version') != 1 or manifest.get('source_sha256') != source_sha256:
        raise ValueError('Preservation manifest does not match the source LoRA')
    examples = manifest.get('examples')
    if not isinstance(examples, list) or not 1 <= len(examples) <= 512:
        raise ValueError('Preservation manifest requires 1-512 curated examples')
    seen = set()
    images = set()
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError('Invalid preservation example')
        for field in ('image', 'caption'):
            name, digest = example.get(field), example.get(field + '_sha256')
            if not isinstance(name, str) or not name or Path(name).name != name or any(c in name for c in ('/', '\\', '\0')) or name in seen:
                raise ValueError('Unsafe or duplicate preservation asset name')
            if not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest):
                raise ValueError('Missing preservation asset checksum')
            seen.add(name)
            file = (root / name).resolve(strict=True)
            limit = 64 * 1024 ** 2 if field == 'image' else 512 * 1024
            if file.parent != root or not file.is_file() or file.stat().st_size > limit:
                raise ValueError('Invalid preservation asset path or size')
            if field == 'image':
                if file.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
                    raise ValueError('Preservation requires ordinary image examples')
                images.add(name)
            elif file.suffix != '.txt' or file.stem != Path(example['image']).stem:
                raise ValueError('Preservation captions must match their images')
            if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                raise ValueError('Preservation examples changed after curation')
    # Dataset loaders discover images recursively. Unlisted examples must never
    # silently join the curated set, even if their captions look plausible.
    discovered = {str(file.relative_to(root)) for file in root.rglob('*')
                  if file.is_file() and file.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.jxl')}
    if discovered != images:
        raise ValueError('Preservation dataset contains uncurated or missing images')
    return manifest_path
