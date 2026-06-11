import argparse
import io
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover - surfaced as a clear runtime error in main
    PILImage = None


CAPTION_PRIORITY = ["caption", "captions", "prompt", "text", "description", "title"]
IMAGE_FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "GIF": ".gif",
    "BMP": ".bmp",
}


class HfDatasetImportError(Exception):
    pass


def get_token():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip() or None

    token_path = os.getenv("HF_TOKEN_PATH")
    if token_path:
        try:
            return Path(token_path).read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    return None


def import_datasets():
    try:
        from datasets import (
            Image as HfImage,
            Value,
            get_dataset_config_names,
            get_dataset_split_names,
            load_dataset,
        )
    except ImportError as error:
        raise HfDatasetImportError(
            "The Python package datasets[vision]==5.0.0 is required. Install project requirements and try again."
        ) from error

    return {
        "HfImage": HfImage,
        "Value": Value,
        "get_dataset_config_names": get_dataset_config_names,
        "get_dataset_split_names": get_dataset_split_names,
        "load_dataset": load_dataset,
    }


def hf_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except TypeError as first_error:
        trimmed = dict(kwargs)
        trimmed.pop("trust_remote_code", None)
        try:
            return fn(*args, **trimmed)
        except TypeError:
            trimmed.pop("token", None)
            try:
                return fn(*args, **trimmed)
            except TypeError:
                raise first_error


def dataset_kwargs(config: str | None, split: str | None, token: str | None, streaming: bool):
    kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "streaming": streaming,
    }
    if config:
        kwargs["name"] = config
    if split:
        kwargs["split"] = split
    if token:
        kwargs["token"] = token
    return kwargs


def get_configs(dataset_id: str, token: str | None):
    api = import_datasets()
    kwargs = {"trust_remote_code": False}
    if token:
        kwargs["token"] = token
    configs = hf_call(api["get_dataset_config_names"], dataset_id, **kwargs)
    return list(configs or ["default"])


def get_splits(dataset_id: str, config: str | None, token: str | None):
    api = import_datasets()
    kwargs = {"trust_remote_code": False}
    if token:
        kwargs["token"] = token
    splits = hf_call(api["get_dataset_split_names"], dataset_id, config, **kwargs)
    return list(splits or [])


def choose_default_config(configs: list[str], requested: str | None):
    if requested and requested in configs:
        return requested
    if "default" in configs:
        return "default"
    return configs[0] if configs else None


def choose_default_split(splits: list[str], requested: str | None):
    if requested and requested in splits:
        return requested
    if "train" in splits:
        return "train"
    return splits[0] if splits else None


def feature_kind(feature: Any):
    api = import_datasets()
    if isinstance(feature, api["HfImage"]):
        return "image"
    if isinstance(feature, api["Value"]):
        dtype = str(getattr(feature, "dtype", "")).lower()
        if dtype in ("string", "large_string"):
            return "string"
        return dtype or "value"
    if isinstance(feature, dict) and feature.get("_type") == "Image":
        return "image"
    return type(feature).__name__


def feature_columns(features: Any):
    if not features:
        return []
    if hasattr(features, "items"):
        return [{"name": name, "kind": feature_kind(feature)} for name, feature in features.items()]
    return []


def is_image_like_value(value: Any):
    if PILImage is not None and isinstance(value, PILImage.Image):
        return True
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return True
        path_value = value.get("path") or value.get("src")
        return isinstance(path_value, str) and bool(path_value)
    if isinstance(value, (str, Path)):
        extension = Path(str(value).split("?", 1)[0]).suffix.lower()
        return extension in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
    return False


def is_string_like_value(value: Any):
    return isinstance(value, str)


def rank_caption_columns(columns: Iterable[str]):
    names = list(dict.fromkeys(column for column in columns if column))
    lower_to_name = {name.lower(): name for name in names}
    ranked: list[str] = []
    for preferred in CAPTION_PRIORITY:
        if preferred in lower_to_name:
            ranked.append(lower_to_name[preferred])
    ranked.extend(name for name in names if name not in ranked)
    return ranked


def detect_columns(features: Any, samples: list[dict[str, Any]]):
    feature_rows = feature_columns(features)
    image_columns = [item["name"] for item in feature_rows if item["kind"] == "image"]
    text_columns = [item["name"] for item in feature_rows if item["kind"] == "string"]

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        for key, value in sample.items():
            if key not in image_columns and is_image_like_value(value):
                image_columns.append(key)
            if key not in text_columns and is_string_like_value(value):
                text_columns.append(key)

    ranked_captions = rank_caption_columns(text_columns)
    return {
        "features": feature_rows,
        "imageColumns": image_columns,
        "textColumns": text_columns,
        "suggestedImageColumn": image_columns[0] if image_columns else None,
        "suggestedCaptionColumn": ranked_captions[0] if ranked_captions else None,
    }


def summarize_value(value: Any):
    if PILImage is not None and isinstance(value, PILImage.Image):
        return {
            "type": "image",
            "width": value.width,
            "height": value.height,
            "format": value.format,
        }
    if isinstance(value, dict) and is_image_like_value(value):
        return {
            "type": "image",
            "path": value.get("path") or value.get("src"),
            "hasBytes": value.get("bytes") is not None,
        }
    if isinstance(value, str):
        return value[:180]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:180]


def preview_samples(dataset: Iterable[dict[str, Any]], limit=5):
    rows = []
    for index, row in enumerate(dataset):
        if index >= limit:
            break
        if isinstance(row, dict):
            rows.append({key: summarize_value(value) for key, value in row.items()})
    return rows


def load_selected_dataset(dataset_id: str, config: str | None, split: str | None, token: str | None):
    api = import_datasets()
    kwargs = dataset_kwargs(config, split, token, streaming=True)
    return hf_call(api["load_dataset"], dataset_id, **kwargs)


def normalize_caption(value: Any):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def extension_from_source(source: str | None):
    if not source:
        return None
    extension = Path(source.split("?", 1)[0]).suffix.lower()
    if extension == ".jpeg":
        return ".jpg"
    if extension in IMAGE_FORMAT_EXTENSIONS.values():
        return extension
    return None


def pil_image_from_value(value: Any):
    if PILImage is None:
        raise HfDatasetImportError("Pillow is required to import image datasets.")

    source = None
    if isinstance(value, PILImage.Image):
        return value, source

    if isinstance(value, dict):
        source_value = value.get("path") or value.get("src")
        source = str(source_value) if source_value else None
        if value.get("bytes") is not None:
            image = PILImage.open(io.BytesIO(value["bytes"]))
            return image, source
        if source and os.path.isfile(source):
            image = PILImage.open(source)
            return image, source

    if isinstance(value, (str, Path)) and os.path.isfile(str(value)):
        source = str(value)
        image = PILImage.open(source)
        return image, source

    return None, source


def image_extension(image: Any, source: str | None):
    source_extension = extension_from_source(source)
    if source_extension:
        return source_extension
    image_format = str(getattr(image, "format", "") or "").upper()
    return IMAGE_FORMAT_EXTENSIONS.get(image_format, ".png")


def save_pil_image(image: Any, target_path: Path, extension: str):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    save_image = image
    save_kwargs: dict[str, Any] = {}
    if extension in (".jpg", ".jpeg"):
        if image.mode not in ("RGB", "L"):
            save_image = image.convert("RGB")
        save_format = "JPEG"
        save_kwargs["quality"] = 95
    else:
        save_format = IMAGE_FORMAT_EXTENSIONS.get(str(getattr(image, "format", "") or "").upper(), None)
        if extension == ".png":
            save_format = "PNG"
        elif extension == ".webp":
            save_format = "WEBP"
        elif extension == ".gif":
            save_format = "GIF"
        elif extension == ".bmp":
            save_format = "BMP"
    save_image.save(target_path, format=save_format, **save_kwargs)


def write_image_file(row: dict[str, Any], image_column: str, output_dir: Path, row_index: int):
    image, source = pil_image_from_value(row.get(image_column))
    if image is None:
        return None
    extension = image_extension(image, source)
    file_path = output_dir / f"row_{row_index:06d}{extension}"
    save_pil_image(image, file_path, extension)
    return file_path


def write_caption_file(image_path: Path, caption: str):
    if not caption:
        return False
    caption_path = image_path.with_suffix(".txt")
    caption_path.write_text(caption, encoding="utf-8")
    return True


def resolve_import_columns(dataset: Any, requested_image: str | None, caption_mode: str, requested_caption: str | None):
    samples = preview_samples(iter(dataset), limit=5)
    detected = detect_columns(getattr(dataset, "features", None), samples)
    image_column = requested_image or detected["suggestedImageColumn"]
    if not image_column:
        raise HfDatasetImportError("No image column was found in the selected dataset split.")

    caption_column = None
    if caption_mode == "column":
        caption_column = requested_caption
        if not caption_column:
            raise HfDatasetImportError("A caption column is required when caption mode is set to column.")
    elif caption_mode == "auto":
        caption_column = detected["suggestedCaptionColumn"]

    return image_column, caption_column


def run_preview(payload: dict[str, Any]):
    token = get_token()
    dataset_id = payload["dataset"]
    configs = get_configs(dataset_id, token)
    selected_config = choose_default_config(configs, payload.get("config"))
    splits = get_splits(dataset_id, selected_config, token) if selected_config else []
    selected_split = choose_default_split(splits, payload.get("split"))
    if not selected_config or not selected_split:
        raise HfDatasetImportError("No importable config and split were found for this dataset.")

    dataset = load_selected_dataset(dataset_id, selected_config, selected_split, token)
    samples = preview_samples(iter(dataset), limit=5)
    detected = detect_columns(getattr(dataset, "features", None), samples)
    row_count = None
    if hasattr(dataset, "info") and getattr(dataset.info, "splits", None):
        split_info = dataset.info.splits.get(selected_split)
        row_count = getattr(split_info, "num_examples", None) if split_info else None

    return {
        "datasetID": dataset_id,
        "configs": configs,
        "splits": splits,
        "selectedConfig": selected_config,
        "selectedSplit": selected_split,
        "rowCount": row_count,
        "samples": samples,
        **detected,
    }


def run_import(payload: dict[str, Any]):
    token = get_token()
    dataset_id = payload["dataset"]
    config = payload.get("config")
    split = payload.get("split")
    if not config or not split:
        configs = get_configs(dataset_id, token)
        config = choose_default_config(configs, config)
        splits = get_splits(dataset_id, config, token) if config else []
        split = choose_default_split(splits, split)
    if not config or not split:
        raise HfDatasetImportError("No importable config and split were found for this dataset.")

    preview_dataset = load_selected_dataset(dataset_id, config, split, token)
    image_column, caption_column = resolve_import_columns(
        preview_dataset,
        payload.get("imageColumn"),
        payload.get("captionMode") or "auto",
        payload.get("captionColumn"),
    )

    output_path = Path(payload["outputPath"])
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=False)

    dataset = load_selected_dataset(dataset_id, config, split, token)
    max_rows = payload.get("maxRows")
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows <= 0:
            max_rows = None

    rows_scanned = 0
    rows_skipped = 0
    images_written = 0
    captions_written = 0
    warnings: list[str] = []

    for row_index, row in enumerate(dataset):
        if max_rows is not None and rows_scanned >= max_rows:
            break
        rows_scanned += 1
        if not isinstance(row, dict):
            rows_skipped += 1
            continue
        try:
            image_path = write_image_file(row, image_column, output_path, row_index)
        except Exception as error:
            rows_skipped += 1
            if len(warnings) < 10:
                warnings.append(f"Skipped row {row_index}: {error}")
            continue
        if image_path is None:
            rows_skipped += 1
            if len(warnings) < 10:
                warnings.append(f"Skipped row {row_index}: no image data in column {image_column}")
            continue
        images_written += 1
        if caption_column:
            caption = normalize_caption(row.get(caption_column))
            if write_caption_file(image_path, caption):
                captions_written += 1

    if images_written == 0:
        raise HfDatasetImportError("No images were imported from the selected dataset split.")

    return {
        "datasetID": dataset_id,
        "config": config,
        "split": split,
        "imageColumn": image_column,
        "captionColumn": caption_column,
        "imagesWritten": images_written,
        "captionsWritten": captions_written,
        "rowsScanned": rows_scanned,
        "rowsSkipped": rows_skipped,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    action = payload.get("action")
    try:
        if action == "preview":
            result = run_preview(payload)
        elif action == "import":
            result = run_import(payload)
        else:
            raise HfDatasetImportError("Invalid action.")
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        print(str(error), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
