"""Validated, portable grouped RGBA samples for native layered-image training."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from PIL import Image

FORMAT = "aitk.layered-image"
ASSETS_DIRECTORY = ".layers"
MAX_LAYERS = 32
MAX_CANVAS_PIXELS = 32_000_000
MAX_DIMENSION = 8192
MAX_MANIFEST_BYTES = 1024 * 1024


class LayeredDatasetError(ValueError):
    pass


def text_length(value: str) -> int:
    """Match the UI contract's UTF-16 string length for portable limits."""
    return len(value.encode('utf-16-le', errors='surrogatepass')) // 2


def contained_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    """Resolve a portable relative path, including symlinks, within its owner."""
    if not isinstance(relative, str) or not relative or text_length(relative) >= 1024 or re.search(r"[\\:\x00-\x1f]", relative):
        raise LayeredDatasetError("Invalid layered sample path")
    parts = PurePosixPath(relative).parts
    if relative.startswith("/") or any(part in ("", ".", "..") for part in relative.split("/")):
        raise LayeredDatasetError("Layered sample paths must be relative and cannot traverse directories")
    root = root.resolve(strict=True)
    candidate = root.joinpath(*parts).resolve(strict=must_exist)
    if not candidate.is_relative_to(root) or candidate == root:
        raise LayeredDatasetError("Layered sample path escapes its dataset")
    if must_exist and not candidate.is_file():
        raise LayeredDatasetError("Layered sample asset is not a regular file")
    return candidate


def validate_canvas(width: Any, height: Any) -> tuple[int, int]:
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 < v <= MAX_DIMENSION for v in (width, height)):
        raise LayeredDatasetError(f"Canvas dimensions must be between 1 and {MAX_DIMENSION}")
    if width * height > MAX_CANVAS_PIXELS:
        raise LayeredDatasetError("Canvas exceeds the 32 megapixel limit")
    return width, height


def assert_plain_dataset(root: Path) -> None:
    if (root / ".aitk_encrypted_dataset.json").exists():
        raise LayeredDatasetError("Grouped layered-image datasets do not support encrypted storage yet")


@dataclass(frozen=True)
class LayeredSample:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]

    @property
    def composite_path(self) -> Path:
        return contained_path(self.root, self.manifest["composite"])

    @property
    def num_frames(self) -> int:
        return len(self.manifest["layers"]) + 1

    def image_paths(self) -> list[Path]:
        return [self.composite_path] + [contained_path(self.root, row["path"]) for row in self.manifest["layers"]]

    def signature(self) -> str:
        # Include contents, order, alpha, captions and source metadata. A copied
        # file with the same timestamp must not reuse a different layer's cache.
        digest = hashlib.sha256(json.dumps(self.manifest, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        for filename in self.image_paths():
            with filename.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def caption(self, default: str = "") -> str:
        filename = contained_path(self.root, self.manifest["caption"])
        if filename.stat().st_size > MAX_MANIFEST_BYTES:
            raise LayeredDatasetError("Layered sample caption exceeds 1 MiB")
        return filename.read_text(encoding="utf-8").strip() or default


def load_layered_sample(root: str | Path, manifest_path: str | Path, *, check_images: bool = True) -> LayeredSample:
    root = Path(root).resolve(strict=True)
    assert_plain_dataset(root)
    filename = Path(manifest_path).resolve(strict=True)
    if not filename.is_relative_to(root / ASSETS_DIRECTORY) or filename.name != "manifest.json":
        raise LayeredDatasetError("Layered manifest must belong to the dataset's .layers directory")
    if filename.stat().st_size > MAX_MANIFEST_BYTES:
        raise LayeredDatasetError("Layered manifest exceeds 1 MiB")
    value = json.loads(filename.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") != FORMAT or isinstance(value.get("version"), bool) or value.get("version") != 1 or value.get("order") != "bottom-to-top":
        raise LayeredDatasetError("Unsupported layered-image manifest")
    sample_id = value.get("id")
    if not isinstance(sample_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", sample_id):
        raise LayeredDatasetError("Invalid layered sample ID")
    assets = contained_path(root, f"{ASSETS_DIRECTORY}/{sample_id}/manifest.json")
    if assets != filename:
        raise LayeredDatasetError("Layered sample ID does not match its directory")
    width, height = validate_canvas(value.get("width"), value.get("height"))
    layers = value.get("layers")
    if not isinstance(layers, list) or not 1 <= len(layers) <= MAX_LAYERS:
        raise LayeredDatasetError(f"A layered sample requires 1 to {MAX_LAYERS} targets")
    composite = contained_path(root, value.get("composite"))
    caption = contained_path(root, value.get("caption"))
    if composite.parent != root or value["composite"].startswith('.') or '/' in value["composite"] or composite.suffix != ".png" or caption != composite.with_suffix(".txt"):
        raise LayeredDatasetError("Composite and matching text caption must be at the dataset root")
    seen = {composite}
    seen_names = set()
    for row in layers:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or text_length(row["name"]) > 512 or not isinstance(row.get("caption", ""), str) or text_length(row.get("caption", "")) > 65_536:
            raise LayeredDatasetError("Invalid layered target metadata")
        row.setdefault("caption", "")
        target = contained_path(root, row.get("path"))
        if target.parent != filename.parent or target.suffix != ".png" or target in seen or row["path"].casefold() in seen_names:
            raise LayeredDatasetError("Layer targets must be unique PNGs owned by this sample")
        seen.add(target)
        seen_names.add(row["path"].casefold())
    source = value.get("source")
    if "source" in value and (not isinstance(source, dict) or source.get("format") not in ("psd", "ora") or not isinstance(source.get("filename"), str) or text_length(source["filename"]) > 512 or not isinstance(source.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"])):
        raise LayeredDatasetError("Invalid layered sample source metadata")
    sample = LayeredSample(root, filename, value)
    if check_images:
        for target in sample.image_paths():
            with Image.open(target) as image:
                if image.format != "PNG" or image.mode != "RGBA" or image.size != (width, height) or getattr(image, "n_frames", 1) != 1:
                    raise LayeredDatasetError("Every target and composite must be a full-canvas static RGBA PNG")
                image.verify()
    return sample


def discover_layered_samples(root: str | Path) -> list[LayeredSample]:
    root = Path(root).resolve(strict=True)
    assert_plain_dataset(root)
    assets = root / ASSETS_DIRECTORY
    if not assets.is_dir():
        raise LayeredDatasetError("Layered-image training requires imported PSD/ORA groups in .layers")
    samples = [load_layered_sample(root, path) for path in sorted(assets.glob("*/manifest.json"))]
    if not samples:
        raise LayeredDatasetError("No layered-image samples found")
    composites = [sample.composite_path for sample in samples]
    if len(set(composites)) != len(composites):
        raise LayeredDatasetError("Multiple layered manifests refer to the same composite")
    return samples


def format_layered_caption(sample: LayeredSample, caption: str) -> str:
    parts = [f"Decompose the image into {len(sample.manifest['layers'])} layers."]
    if caption.strip():
        parts.append(caption.strip())
    for index, layer in enumerate(sample.manifest["layers"], 1):
        if layer["caption"].strip():
            parts.append(f"Layer {index}: {layer['caption'].strip()}")
    return "\n".join(parts)


def validate_layered_config(config: Any, *, batch_size: int) -> None:
    if config.encrypted:
        raise LayeredDatasetError("Grouped layered-image datasets do not support encrypted storage yet")
    incompatible = ("standardize_images", "augments", "augmentations", "poi", "control_path", "control_from_same_folder", "mask_path", "alpha_mask", "inpaint_path", "clip_image_path", "unconditional_path", "controls", "do_audio", "auto_frame_count")
    for key in incompatible:
        if getattr(config, key, None):
            raise LayeredDatasetError(f"Layered-image datasets require {key}=false or empty")
    if config.num_frames != 1:
        raise LayeredDatasetError("Layer counts come from manifests; leave dataset num_frames=1")
    if not config.buckets and batch_size > 1:
        raise LayeredDatasetError("Layered-image batches larger than one require buckets")
    if config.cache_text_embeddings and (config.random_crop or config.random_scale):
        raise LayeredDatasetError("Cached layered conditioning requires fixed geometry; disable random_crop and random_scale")
    if not config.buckets and (config.random_crop or config.random_scale):
        raise LayeredDatasetError("Random layered-image geometry requires buckets")
    if config.shuffle_tokens or config.token_dropout_rate or config.random_triggers or config.caption_dropout_rate:
        raise LayeredDatasetError("Layered-image prompts require fixed layer-count instructions; disable caption/token shuffling and dropout")


def load_layered_tensor(item: Any, *, composite_only: bool = False):
    """Apply one geometry to every RGBA plane; controls share frame zero exactly."""
    import numpy as np
    import torch
    from PIL import ImageOps

    sample: LayeredSample = item.layered_sample
    paths = sample.image_paths()[:1] if composite_only else sample.image_paths()
    tensors = []
    for path in paths:
        with Image.open(path) as original:
            image = original.convert("RGBA")
        if image.size != (sample.manifest["width"], sample.manifest["height"]):
            raise LayeredDatasetError("Layered asset dimensions changed after dataset initialization")
        if item.flip_x:
            image = ImageOps.mirror(image)
        if item.flip_y:
            image = ImageOps.flip(image)
        if item.dataset_config.buckets:
            image = image.resize((item.scale_to_width, item.scale_to_height), Image.Resampling.BICUBIC)
            image = image.crop((item.crop_x, item.crop_y, item.crop_x + item.crop_width, item.crop_y + item.crop_height))
        else:
            # Deterministic aligned center crop for the non-bucketed batch-one path.
            side = min(image.size)
            left, top = (image.width - side) // 2, (image.height - side) // 2
            image = image.crop((left, top, left + side, top + side)).resize((item.dataset_config.resolution, item.dataset_config.resolution), Image.Resampling.BICUBIC)
        tensors.append(torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).permute(2, 0, 1).div_(127.5).sub_(1))
    stacked = torch.stack(tensors)
    item.control_tensor = (stacked[0] + 1).div(2).clamp(0, 1)
    if not composite_only:
        item.tensor = stacked
    return stacked
