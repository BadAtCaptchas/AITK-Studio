"""Bounded PSD/OpenRaster conversion to grouped, full-canvas RGBA samples."""
from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import sys
import tempfile
import uuid
import zipfile

from PIL import Image

from toolkit.layered_dataset import (
    ASSETS_DIRECTORY, FORMAT, MAX_LAYERS, LayeredDatasetError,
    assert_plain_dataset, load_layered_sample, text_length, validate_canvas,
)

MAX_INPUT_BYTES = 512 * 1024 ** 2
MAX_EXPANDED_BYTES = 2 * 1024 ** 3
MAX_DECODED_BYTES = 1024 ** 3
MAX_NODES = 256
MAX_DEPTH = 16
MAX_ZIP_ENTRIES = 2048
MAX_XML_BYTES = 1024 ** 2


def _safe_archive_name(name: str) -> str:
    if not name or "\\" in name or name.startswith("/") or any(p in ("", ".", "..") for p in name.rstrip("/").split("/")):
        raise LayeredDatasetError("Unsafe OpenRaster archive path")
    for part in PurePosixPath(name).parts:
        if re.search(r'[<>:"|?*\x00-\x1f]', part) or part.endswith((" ", ".")) or re.match(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", part, re.I):
            raise LayeredDatasetError("Nonportable OpenRaster archive path")
    return name


def _opacity(value: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise LayeredDatasetError("Invalid layer opacity") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise LayeredDatasetError("Layer opacity must be between zero and one")
    return number


def _apply_opacity(image: Image.Image, opacity: float) -> Image.Image:
    if opacity != 1:
        image.putalpha(image.getchannel("A").point([round(value * opacity) for value in range(256)]))
    return image


def _read_ora(source: Path) -> tuple[list[tuple[str, Image.Image]], tuple[int, int], list[str]]:
    from defusedxml.ElementTree import fromstring

    warnings: list[str] = []
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_ZIP_ENTRIES:
            raise LayeredDatasetError("OpenRaster archive has too many entries")
        by_name = {}
        portable_names = set()
        expanded = 0
        for info in entries:
            name = _safe_archive_name(info.filename)
            key = name.rstrip("/").casefold()
            mode = stat.S_IFMT(info.external_attr >> 16)
            if key in portable_names or mode not in (0, stat.S_IFREG, stat.S_IFDIR) or info.flag_bits & 1:
                raise LayeredDatasetError("OpenRaster archive contains duplicate paths, links, special files or encrypted entries")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise LayeredDatasetError("Unsupported OpenRaster compression")
            portable_names.add(key)
            expanded += info.file_size
            if expanded > MAX_EXPANDED_BYTES:
                raise LayeredDatasetError("OpenRaster expanded data exceeds 2 GiB")
            by_name[name] = info

        consumed = 0

        def read(name: str, limit: int) -> bytes:
            nonlocal consumed
            _safe_archive_name(name)
            entry = by_name.get(name)
            if entry is None or entry.is_dir() or entry.file_size > limit:
                raise LayeredDatasetError(f"Missing or oversized OpenRaster asset: {name}")
            with archive.open(entry) as stream:
                data = stream.read(limit + 1)
            consumed += len(data)
            if len(data) > limit or len(data) != entry.file_size or consumed > MAX_EXPANDED_BYTES:
                raise LayeredDatasetError("OpenRaster actual expansion limit exceeded")
            return data

        if not entries or entries[0].filename != "mimetype" or entries[0].compress_type != zipfile.ZIP_STORED or read("mimetype", 64) != b"image/openraster":
            raise LayeredDatasetError("Invalid OpenRaster mimetype header")
        root = fromstring(read("stack.xml", MAX_XML_BYTES), forbid_dtd=True, forbid_entities=True, forbid_external=True)
        if root.tag != "image" or len(root) != 1 or root[0].tag != "stack":
            raise LayeredDatasetError("OpenRaster requires one image and root stack")
        if set(root.attrib) - {"w", "h", "name", "version", "xres", "yres"}:
            raise LayeredDatasetError("Unsupported OpenRaster document attributes")
        try:
            canvas = validate_canvas(int(root.get("w", "0")), int(root.get("h", "0")))
        except ValueError as exc:
            raise LayeredDatasetError("Invalid OpenRaster canvas") from exc
        nodes = 0
        decoded = 0

        def render(node, depth: int, label: str) -> Image.Image | None:
            nonlocal nodes, decoded
            nodes += 1
            if nodes > MAX_NODES or depth > MAX_DEPTH:
                raise LayeredDatasetError("OpenRaster layer count or nesting limit exceeded")
            visibility = node.get("visibility", "visible")
            if visibility not in ("visible", "hidden"):
                raise LayeredDatasetError(f"Invalid visibility: {label}")
            if visibility == "hidden":
                return None
            allowed = {"name", "opacity", "visibility", "composite-op", "x", "y"}
            allowed.add("isolation" if node.tag == "stack" else "src")
            if set(node.attrib) - allowed:
                raise LayeredDatasetError(f"Unsupported OpenRaster layer attributes on {label}")
            if node.get("composite-op", "svg:src-over") != "svg:src-over":
                raise LayeredDatasetError(f"Unsupported blend mode on {label}; merge/rasterize it in your editor")
            opacity = _opacity(node.get("opacity", "1"))
            if node.tag == "stack":
                if node.get("x", "0") != "0" or node.get("y", "0") != "0":
                    raise LayeredDatasetError(f"Unsupported group offsets on {label}; bake offsets into the child layers")
                isolation = node.get("isolation", "isolate")
                if isolation not in ("isolate", "auto") or (isolation == "auto" and opacity != 1):
                    raise LayeredDatasetError(f"Unsupported pass-through group on {label}; isolate/rasterize it in your editor")
                image = Image.new("RGBA", canvas)
                for index, child in enumerate(reversed(list(node))):
                    rendered = render(child, depth + 1, f"{label}/{child.get('name', str(index))}")
                    if rendered is not None:
                        image = Image.alpha_composite(image, rendered)
                return _apply_opacity(image, opacity)
            if node.tag != "layer" or len(node):
                raise LayeredDatasetError(f"Unsupported OpenRaster element on {label}; rasterize it in your editor")
            src = node.get("src", "")
            if not src.lower().endswith(".png"):
                raise LayeredDatasetError(f"OpenRaster target {label} requires an embedded PNG")
            data = read(src, MAX_INPUT_BYTES)
            with Image.open(io.BytesIO(data)) as pixels:
                validate_canvas(*pixels.size)
                decoded += pixels.width * pixels.height * 4
                if decoded > MAX_DECODED_BYTES:
                    raise LayeredDatasetError("OpenRaster decoded pixels exceed 1 GiB")
                if pixels.format != "PNG" or getattr(pixels, "n_frames", 1) != 1 or pixels.mode not in ("RGB", "RGBA", "P", "L", "LA"):
                    raise LayeredDatasetError(f"Unsupported PNG data on {label}")
                pixels = pixels.convert("RGBA")
            try:
                offset = (int(node.get("x", "0")), int(node.get("y", "0")))
            except ValueError as exc:
                raise LayeredDatasetError(f"Invalid layer offset on {label}") from exc
            if any(abs(value) > 2 ** 31 - 1 for value in offset):
                raise LayeredDatasetError("Layer offset exceeds supported range")
            image = Image.new("RGBA", canvas)
            # Paste without an alpha mask: using it here squares semi-transparency.
            image.paste(_apply_opacity(pixels, opacity), offset)
            return image

        # Count hidden descendants too, before allocating image buffers.
        def check_tree(node, depth=0):
            if depth > MAX_DEPTH:
                raise LayeredDatasetError("OpenRaster nesting limit exceeded")
            return 1 + sum(check_tree(child, depth + 1) for child in node)
        if check_tree(root[0]) > MAX_NODES:
            raise LayeredDatasetError("OpenRaster layer count limit exceeded")
        if any(key in root[0].attrib for key in ("opacity", "visibility", "composite-op", "isolation")):
            raise LayeredDatasetError("OpenRaster root stack cannot carry compositing attributes")
        targets = []
        for index, node in enumerate(reversed(list(root[0]))):
            name = node.get("name", f"Layer {index + 1}")
            image = render(node, 1, name)
            if image is None or image.getchannel("A").getbbox() is None:
                warnings.append(f"Excluded hidden or empty target: {name}")
            else:
                targets.append((name, image))
                if len(targets) > MAX_LAYERS:
                    raise LayeredDatasetError(f"Document exceeds {MAX_LAYERS} exported targets")
                if (len(targets) + 1) * canvas[0] * canvas[1] * 4 > MAX_DECODED_BYTES:
                    raise LayeredDatasetError("Full-canvas targets exceed the 1 GiB decoded pixel limit")
        return targets, canvas, warnings


def _read_psd(source: Path) -> tuple[list[tuple[str, Image.Image]], tuple[int, int], list[str]]:
    from psd_tools import PSDImage
    from psd_tools.constants import BlendMode, Tag
    from psd_tools.composite import composite_pil

    with source.open("rb") as stream:
        header = stream.read(26)
    if len(header) != 26 or header[:4] != b"8BPS":
        raise LayeredDatasetError("Invalid Photoshop document")
    version, _reserved, _channels, height, width, depth, mode = struct.unpack(">H6sHIIHH", header[4:])
    canvas = validate_canvas(width, height)
    if version != 1 or depth != 8 or mode != 3:
        raise LayeredDatasetError("PSD import supports 8-bit RGB PSD files; convert PSB/CMYK/16-bit documents in your editor")
    document = PSDImage.open(source)
    descendants = list(document.descendants())
    if len(descendants) > MAX_NODES:
        raise LayeredDatasetError("PSD layer count limit exceeded")
    decoded = sum(max(0, item.width) * max(0, item.height) * 4 for item in descendants if not item.is_group())
    if decoded > MAX_DECODED_BYTES:
        raise LayeredDatasetError("PSD decoded pixels exceed 1 GiB")

    def validate(layer, nesting=1, label=""):
        label = f"{label}/{layer.name}" if label else layer.name
        if nesting > MAX_DEPTH:
            raise LayeredDatasetError("PSD nesting limit exceeded")
        if not layer.is_visible():
            return
        if layer.clipping or layer.has_effects():
            raise LayeredDatasetError(f"Unsupported clipping/effects on {label}; rasterize/merge it in your editor")
        if layer.has_vector_mask():
            raise LayeredDatasetError(f"Unsupported vector mask on {label}; rasterize it in your editor")
        ranges = layer._record.blending_ranges
        range_rows = list(ranges.composite_ranges or [])
        range_rows.extend(pair for channel in (ranges.channel_ranges or []) for pair in channel)
        if any(tuple(pair) != (0, 65535) for pair in range_rows):
            raise LayeredDatasetError(f"Unsupported Blend If ranges on {label}; rasterize/merge it in your editor")
        if layer.tagged_blocks.get_data(Tag.KNOCKOUT_SETTING, 0) or layer.tagged_blocks.get_data(Tag.CHANNEL_BLENDING_RESTRICTIONS_SETTING, []):
            raise LayeredDatasetError(f"Unsupported channel blending/knockout on {label}; rasterize/merge it in your editor")
        if layer.is_group():
            if layer.blend_mode == BlendMode.PASS_THROUGH:
                if layer.opacity != 255 or layer.fill_opacity != 255 or layer.has_mask():
                    raise LayeredDatasetError(f"Unsupported pass-through group on {label}; isolate/rasterize it in your editor")
            elif layer.blend_mode != BlendMode.NORMAL:
                raise LayeredDatasetError(f"Unsupported group blend mode on {label}")
            for child in layer:
                validate(child, nesting + 1, label)
        else:
            if layer.kind == "smartobject" and layer.smart_object.kind != "data":
                raise LayeredDatasetError(f"External/linked smart object on {label}; embed or rasterize it in your editor")
            if layer.blend_mode != BlendMode.NORMAL or layer.kind not in ("pixel", "type", "smartobject") or not layer.has_pixels() or layer.has_vector_mask():
                raise LayeredDatasetError(f"Unsupported layer {label}; rasterize/merge it in your editor")

    targets = []
    warnings = []
    for layer in document:
        validate(layer)
        if not layer.is_visible():
            warnings.append(f"Excluded hidden target: {layer.name}")
            continue
        # as_layer includes the group's own mask/opacity, unlike rendering
        # just its children. Nothing opens smart_object external references.
        image = composite_pil(layer, viewport=(0, 0, width, height), color=0.0, alpha=0.0,
                              layer_filter=None, force=False, as_layer=True, apply_icc=True)
        if image is None:
            warnings.append(f"Excluded empty target: {layer.name}")
            continue
        image = image.convert("RGBA")
        if image.getchannel("A").getbbox() is None:
            warnings.append(f"Excluded empty target: {layer.name}")
            continue
        targets.append((layer.name, image))
        if len(targets) > MAX_LAYERS:
            raise LayeredDatasetError(f"Document exceeds {MAX_LAYERS} exported targets")
        if (len(targets) + 1) * width * height * 4 > MAX_DECODED_BYTES:
            raise LayeredDatasetError("Full-canvas targets exceed the 1 GiB decoded pixel limit")
    return targets, canvas, warnings


def _publish_file(source: Path, target: Path) -> None:
    """Same-volume publication with a no-replacement fallback for other filesystems."""
    try:
        os.link(source, target)
        return
    except OSError as error:
        if error.errno not in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV) and getattr(error, 'winerror', None) != 1:
            raise
    created = False
    try:
        with target.open('xb') as output:
            created = True
            with source.open('rb') as content:
                shutil.copyfileobj(content, output)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise


def import_layered_document(input_path: str | Path, output_path: str | Path, name: str | None = None) -> dict:
    source = Path(input_path).resolve(strict=True)
    destination = Path(output_path).resolve(strict=True)
    if not source.is_file() or not destination.is_dir():
        raise LayeredDatasetError("Import requires a regular source file and an existing dataset directory")
    assert_plain_dataset(destination)
    if source.stat().st_size > MAX_INPUT_BYTES:
        raise LayeredDatasetError("Layered document exceeds the 512 MiB upload limit")
    original_name = (name or source.name).replace("\\", "/").split("/")[-1]
    if text_length(original_name) > 512:
        raise LayeredDatasetError("Source filename exceeds the 512-character limit")
    file_format = Path(original_name).suffix.lower().lstrip(".")
    if file_format not in ("psd", "ora"):
        raise LayeredDatasetError("Layered import accepts .psd and .ora documents")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    targets, canvas, warnings = _read_psd(source) if file_format == "psd" else _read_ora(source)
    if any(text_length(layer_name) > 512 for layer_name, _ in targets):
        raise LayeredDatasetError("Layer name exceeds the 512-character limit; shorten it in the source document")
    if not targets:
        raise LayeredDatasetError("Document has no visible nonempty layer targets")
    sample_id = uuid.uuid4().hex
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(original_name).stem).strip("_")[:80] or "document"
    stem = f"{stem}_{sample_id[:8]}"
    composite_name, caption_name = f"{stem}.png", f"{stem}.txt"
    assets_root = destination / ASSETS_DIRECTORY
    assets_root.mkdir(exist_ok=True)
    if assets_root.is_symlink() or assets_root.resolve() != assets_root:
        raise LayeredDatasetError("Layer asset directory must not be a symlink or junction")
    staging = Path(tempfile.mkdtemp(prefix=".layered-import-", dir=destination))
    published = []
    assets = assets_root / sample_id
    try:
        staged_assets = staging / "assets"
        staged_assets.mkdir()
        composite = Image.new("RGBA", canvas)
        rows = []
        for index, (layer_name, pixels) in enumerate(targets):
            filename = f"{index:03d}.png"
            pixels.save(staged_assets / filename)
            composite = Image.alpha_composite(composite, pixels)
            rows.append({"path": f"{ASSETS_DIRECTORY}/{sample_id}/{filename}", "name": layer_name, "caption": ""})
        composite.save(staging / composite_name)
        (staging / caption_name).write_text("", encoding="utf-8")
        manifest = {"format": FORMAT, "version": 1, "id": sample_id, "composite": composite_name, "caption": caption_name, "width": canvas[0], "height": canvas[1], "order": "bottom-to-top", "layers": rows, "source": {"format": file_format, "filename": original_name, "sha256": digest.hexdigest()}}
        (staged_assets / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # Reserve the asset directory before publishing its files; the manifest
        # is committed last. Exclusive creation preserves pre-existing files.
        assets.mkdir()
        published.append(assets)
        for child in staged_assets.iterdir():
            if child.name == 'manifest.json':
                continue
            shutil.copyfile(child, assets / child.name)
        for filename in (caption_name, composite_name):
            target = destination / filename
            _publish_file(staging / filename, target)
            published.append(target)
        # A manifest is the grouped-discovery commit marker. Until this point,
        # a concurrent grouped loader cannot observe an incomplete sample.
        _publish_file(staged_assets / 'manifest.json', assets / 'manifest.json')
        load_layered_sample(destination, assets / "manifest.json")
        return {"id": sample_id, "manifest": f"{ASSETS_DIRECTORY}/{sample_id}/manifest.json", "composite": composite_name, "caption": caption_name, "width": canvas[0], "height": canvas[1], "layer_count": len(rows), "warnings": warnings}
    except BaseException:
        for entry in reversed(published):
            if entry.parent in (destination, assets_root) and entry.resolve().is_relative_to(destination):
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name")
    args = parser.parse_args()
    try:
        result = import_layered_document(args.input, args.output, args.name)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
