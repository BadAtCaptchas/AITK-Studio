import asyncio
from collections import OrderedDict

import importlib.util
import json
import math
import os
import re
from typing import Literal, Optional
import threading
import time
import signal
import concurrent.futures
from PIL import Image

import torch
import torchaudio
from jobs.process import BaseExtensionProcess
import tqdm

from toolkit.exceptions import JobStopRequested
from toolkit.encrypted_dataset import EncryptedDatasetReader, is_encrypted_dataset_path
from toolkit.train_tools import get_torch_dtype
from toolkit.ui_database import UIJobStore

AITK_Status = Literal["running", "stopped", "error", "completed"]

IDEOGRAM_JSON_OUTPUT_FORMAT = "ideogram_json"

IDEOGRAM_JSON_FORMAT_REQUIREMENTS = """Use this exact JSON contract:
- Top-level key order: high_level_description, style_description, compositional_deconstruction.
- For photo captions, style_description key order must be: aesthetics, lighting, photo, medium, color_palette.
- For non-photo captions, style_description key order must be: aesthetics, lighting, medium, art_style, color_palette.
- Include exactly one of style_description.photo or style_description.art_style.
- compositional_deconstruction key order must be: background, elements.
- Object element key order must be: type, bbox, desc, color_palette.
- Text element key order must be: type, bbox, text, desc, color_palette.
- Omit bbox or color_palette only when unavailable; if present, keep them in the listed position.
- Bounding boxes must be [ymin, xmin, ymax, xmax] normalized to 0-1000.
- Colors must be uppercase #RRGGBB hex strings."""

DEFAULT_IDEOGRAM_JSON_PROMPT = """Create an Ideogram 4 training caption for this image as a JSON object.
Return only valid JSON. Do not wrap it in markdown.

high_level_description should be a concise but detailed one-paragraph description.
For each important visible element, include type ("obj" or "text"), desc, optional color_palette, and bbox when you can estimate it. For text elements, include text with the readable text content when known, or an empty string when unreadable.

""" + IDEOGRAM_JSON_FORMAT_REQUIREMENTS + """

Preserve important details from any existing caption, but correct it when the image contradicts it."""

DEFAULT_TEXT_CAPTION_PROMPTS = {
    "Describe this image in detail.",
    'Caption this image as if you were going to try to generate it with an image generator. Be thurough and describe everything in the image. Be decisive by stating things as they are. Do not say things like "It appears that" Or "possibly". Start out with things like "A person on the beach" or "A black dragon". No preamble. Just get to the point.',
}

IDEOGRAM_JSON_CONTRACT_MARKERS = (
    "photo, medium, color_palette",
    "medium, art_style, color_palette",
    "type, bbox, desc, color_palette",
)

IDEOGRAM_JSON_TOP_LEVEL_KEY_ORDER = (
    "high_level_description",
    "style_description",
    "compositional_deconstruction",
)
IDEOGRAM_JSON_STYLE_KEY_ORDER_PHOTO = (
    "aesthetics",
    "lighting",
    "photo",
    "medium",
    "color_palette",
)
IDEOGRAM_JSON_STYLE_KEY_ORDER_NON_PHOTO = (
    "aesthetics",
    "lighting",
    "medium",
    "art_style",
    "color_palette",
)
IDEOGRAM_JSON_COMPOSITION_KEY_ORDER = ("background", "elements")
IDEOGRAM_JSON_ELEMENT_KEY_ORDER_OBJ = ("type", "bbox", "desc", "color_palette")
IDEOGRAM_JSON_ELEMENT_KEY_ORDER_TEXT = (
    "type",
    "bbox",
    "text",
    "desc",
    "color_palette",
)
IDEOGRAM_JSON_STYLE_PALETTE_MAX = 16
IDEOGRAM_JSON_ELEMENT_PALETTE_MAX = 5
IDEOGRAM_JSON_PHOTO_CUES = (
    "photo",
    "photograph",
    "photographic",
    "camera",
    "lens",
    "depth of field",
    "bokeh",
    "aperture",
    "shutter",
    "exposure",
    "focal length",
    "35mm",
    "50mm",
    "85mm",
    "dslr",
    "mirrorless",
)

IDEOGRAM_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "high_level_description": {
            "type": "string",
            "description": "A concise detailed description of the whole image.",
        },
        "style_description": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "aesthetics": {"type": "string"},
                "lighting": {"type": "string"},
                "photo": {
                    "type": "string",
                    "description": "Photographic qualities. Use this instead of art_style for photographs.",
                },
                "medium": {"type": "string"},
                "art_style": {
                    "type": "string",
                    "description": "Artistic style. Use this instead of photo for non-photographs.",
                },
                "color_palette": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^#[0-9A-F]{6}$"},
                    "maxItems": 16,
                },
            },
            "required": ["aesthetics", "lighting", "medium", "color_palette"],
        },
        "compositional_deconstruction": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "background": {"type": "string"},
                "elements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "enum": ["obj", "text"]},
                            "bbox": {
                                "type": "array",
                                "items": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 1000,
                                },
                                "minItems": 4,
                                "maxItems": 4,
                            },
                            "text": {"type": "string"},
                            "desc": {"type": "string"},
                            "color_palette": {
                                "type": "array",
                                "items": {"type": "string", "pattern": "^#[0-9A-F]{6}$"},
                                "maxItems": 5,
                            },
                        },
                        "required": ["type", "desc"],
                    },
                },
            },
            "required": ["background", "elements"],
        },
    },
    "required": [
        "high_level_description",
        "style_description",
        "compositional_deconstruction",
    ],
}


def _normalize_caption_extension(value: Optional[str], default: str = "txt") -> str:
    normalized = (value or default).strip().lstrip(".").lower()
    return normalized or default


class CaptionConfig:
    def __init__(self, **kwargs):
        self.model_name_or_path = kwargs.get("model_name_or_path", None)
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required in config")
        self.model_name_or_path2 = kwargs.get("model_name_or_path2", None)
        self.extensions = kwargs.get("extensions", [])
        if self.extensions is None or len(self.extensions) == 0:
            raise ValueError("At least one extension is required in config")
        self.path_to_caption = kwargs.get("path_to_caption", None)
        if self.path_to_caption is None:
            raise ValueError("path_to_caption is required in config")
        self.dtype = kwargs.get("dtype", "bf16")
        self.device = kwargs.get("device", "cuda")
        self.quantize = kwargs.get("quantize", False)
        self.qtype = kwargs.get("qtype", "float8")
        self.low_vram = kwargs.get("low_vram", False)
        self.caption_extension = kwargs.get("caption_extension", "txt")
        self.caption_extension = _normalize_caption_extension(self.caption_extension)
        self.recaption = kwargs.get("recaption", False)
        self.max_res = kwargs.get("max_res", 512)
        self.max_new_tokens = kwargs.get("max_new_tokens", 128)
        self.output_format = kwargs.get("output_format", "text")
        self.source_caption_extension = _normalize_caption_extension(
            kwargs.get("source_caption_extension", "txt")
        )
        self.delete_source_caption = kwargs.get("delete_source_caption", False)
        self.caption_prompt = kwargs.get(
            "caption_prompt", "Describe this image in detail."
        )


class BaseCaptioner(BaseExtensionProcess):
    caption_config_class = CaptionConfig

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(BaseCaptioner, self).__init__(process_id, job, config, **kwargs)
        self.sqlite_db_path = self.config.get("sqlite_db_path", "./aitk_db.db")
        self.job_id = os.environ.get("AITK_JOB_ID", None)
        self.job_id = self.job_id.strip() if self.job_id is not None else None
        self.ui_job_store = UIJobStore(self.job_id, self.sqlite_db_path)
        self.is_ui_captioner = self.ui_job_store.available
        if self.is_ui_captioner:
            print(f"Using {self.ui_job_store.description}")
            print(f'Job ID: "{self.job_id}"')

        self.is_stopping = False

        if self.is_ui_captioner:
            self.is_stopping = False
            # Create a thread pool for database operations
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            # Track all async tasks
            self._async_tasks = []
            # Initialize the status
            self._run_async_operation(self._update_status("running", "Starting"))
            self._stop_watcher_started = False
            # self.start_stop_watcher(interval_sec=2.0)

        self.caption_config = self.caption_config_class(**self.get_conf("caption", {}))
        self.encrypted_reader = None
        self.encrypted_items_by_path = {}
        if os.path.isdir(self.caption_config.path_to_caption) and is_encrypted_dataset_path(self.caption_config.path_to_caption):
            self.encrypted_reader = EncryptedDatasetReader(self.caption_config.path_to_caption)
        self.model = None
        self.processor = None
        self.model2 = None
        self.processor2 = None
        self.file_paths = []
        self.caption_success_count = 0
        self.caption_failure_count = 0
        self.device_torch = torch.device(self.caption_config.device)
        self.torch_dtype = get_torch_dtype(self.caption_config.dtype)

    def run(self):
        super(BaseCaptioner, self).run()
        with torch.no_grad():
            self.start_stop_watcher()
            self.update_status("running", "Loading Model")
            self.load_model()
            self.update_status("running", "Looking for files")
            self.find_files()
            self.update_status("running", f"Captioning {len(self.file_paths)} files")
            self.run_caption_loop()
            self.update_status("completed", "Captioning completed")
            if self.is_ui_captioner:
                asyncio.run(self.wait_for_all_async())
                self.thread_pool.shutdown(wait=True)
                self.ui_job_store.close()
            print("")

            print("****************************************************")
            print("Captioning complete")
            print("****************************************************")

    def run_caption_loop(self):
        self.caption_success_count = 0
        self.caption_failure_count = 0
        for file_path in tqdm.tqdm(
            self.file_paths, desc="Captioning files", unit="file"
        ):
            if self.is_ui_captioner:
                self.maybe_stop()
                if self.is_stopping:
                    break
            try:
                file_caption = self.get_preserved_ideogram_json_caption_for_file(
                    file_path
                )
                if file_caption is None:
                    file_caption = self.get_caption_for_file(file_path)
                if file_caption is None or str(file_caption).strip() == "":
                    self.caption_failure_count += 1
                    print(f"Error captioning file {file_path}: captioner returned no text")
                    continue
                self.save_caption_for_file(file_path, str(file_caption).strip())
                self.caption_success_count += 1
            except Exception as e:
                self.caption_failure_count += 1
                print(f"Error captioning file {file_path}: {e}")
                continue
        if self.file_paths and self.caption_success_count == 0 and self.caption_failure_count > 0:
            raise RuntimeError(
                f"Captioning failed: no captions were generated for {self.caption_failure_count} attempted file(s)."
            )
        if self.caption_failure_count > 0:
            print(
                f"Captioning completed with {self.caption_failure_count} failed file(s) "
                f"and {self.caption_success_count} saved caption(s)."
            )

    def load_pil_image(self, file_path: str, max_res: Optional[int] = None) -> Image:
        if self.encrypted_reader is not None:
            image = self.encrypted_reader.open_image(self.encrypted_items_by_path[file_path]).convert("RGB")
        else:
            image = Image.open(file_path).convert("RGB")
        if max_res is not None:
            max_pixels = max_res * max_res
            image_pixels = image.width * image.height
            if image_pixels > max_pixels:
                scale_factor = (max_pixels / image_pixels) ** 0.5
                new_width = int(image.width * scale_factor)
                new_height = int(image.height * scale_factor)
                image = image.resize((new_width, new_height), resample=Image.BICUBIC)
        return image

    def save_caption_for_file(self, file_path: str, caption: str):
        if self.encrypted_reader is not None:
            self.encrypted_reader.save_caption(self.encrypted_items_by_path[file_path], caption)
            return
        filename_no_ext = os.path.splitext(file_path)[0]
        caption_file_path = f"{filename_no_ext}.{self.caption_config.caption_extension}"
        # delete it if it already exists
        if os.path.exists(caption_file_path):
            os.remove(caption_file_path)
        with open(caption_file_path, "w", encoding="utf-8") as f:
            f.write(caption)
        self.delete_source_caption_for_file(file_path, caption_file_path)

    def get_caption_for_file(self, file_path: str) -> str:
        raise NotImplementedError("Captioning not implemented for this captioner")

    def is_ideogram_json_output(self) -> bool:
        return str(self.caption_config.output_format or "").strip().lower() in {
            IDEOGRAM_JSON_OUTPUT_FORMAT,
            "json",
        }

    def get_source_caption_for_file(self, file_path: str) -> str:
        if self.encrypted_reader is not None:
            try:
                return (
                    self.encrypted_reader.get_caption(
                        self.encrypted_items_by_path[file_path]
                    )
                    or ""
                ).strip()
            except Exception:
                return ""

        source_ext = self.caption_config.source_caption_extension
        if not source_ext:
            return ""
        source_path = f"{os.path.splitext(file_path)[0]}.{source_ext}"
        if not os.path.exists(source_path):
            return ""
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""

    def build_caption_prompt(self, file_path: str) -> str:
        if not self.is_ideogram_json_output():
            return self.caption_config.caption_prompt.strip()

        prompt = (self.caption_config.caption_prompt or "").strip()
        if not prompt or prompt in DEFAULT_TEXT_CAPTION_PROMPTS:
            prompt = DEFAULT_IDEOGRAM_JSON_PROMPT
        elif not self._caption_prompt_has_ideogram_json_contract(prompt):
            prompt = f"{prompt}\n\n{IDEOGRAM_JSON_FORMAT_REQUIREMENTS}"

        source_caption = self.get_source_caption_for_file(file_path)
        if source_caption:
            if "{existing_caption}" in prompt:
                prompt = prompt.replace("{existing_caption}", source_caption)
            else:
                prompt = (
                    f"{prompt}\n\nExisting caption to preserve and refine:\n"
                    f"{source_caption}"
                )
        return prompt

    def normalize_caption_output(self, file_path: str, caption: str) -> str:
        if not self.is_ideogram_json_output():
            return str(caption).strip()

        parsed = self._parse_json_caption(str(caption))
        parsed = self._normalize_ideogram_json_caption(parsed)
        self._warn_for_ideogram_json_issues(parsed, file_path)
        return json.dumps(parsed, ensure_ascii=False, indent=2)

    def get_preserved_ideogram_json_caption_for_file(
        self, file_path: str
    ) -> Optional[str]:
        if not self.is_ideogram_json_output():
            return None

        source_caption = self.get_source_caption_for_file(file_path)
        if not source_caption:
            return None

        try:
            parsed = self._parse_json_caption(source_caption)
            parsed = self._normalize_ideogram_json_caption(parsed)
            if self._ideogram_caption_verifier().verify(parsed):
                return None
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            return None

    def _parse_json_caption(self, caption: str) -> dict:
        text = caption.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Captioner did not return a JSON object")
            parsed = json.loads(text[start : end + 1])

        if isinstance(parsed, dict) and set(parsed.keys()) == {"caption"}:
            nested = parsed.get("caption")
            if isinstance(nested, str):
                parsed = json.loads(nested)

        if not isinstance(parsed, dict):
            raise ValueError("Captioner returned JSON, but the root value is not an object")
        return parsed

    @staticmethod
    def _caption_prompt_has_ideogram_json_contract(prompt: str) -> bool:
        normalized = " ".join(prompt.lower().split())
        return all(marker in normalized for marker in IDEOGRAM_JSON_CONTRACT_MARKERS)

    def _normalize_ideogram_json_caption(self, caption: dict) -> dict:
        normalized = OrderedDict()
        for key in IDEOGRAM_JSON_TOP_LEVEL_KEY_ORDER:
            if key not in caption:
                continue
            value = caption[key]
            if key == "style_description":
                value = self._normalize_ideogram_style_description(value)
            elif key == "compositional_deconstruction":
                value = self._normalize_ideogram_compositional_deconstruction(value)
            normalized[key] = value

        for key, value in caption.items():
            if key not in normalized:
                normalized[key] = value
        return normalized

    def _normalize_ideogram_style_description(self, style_description):
        if not isinstance(style_description, dict):
            return style_description

        normalized = OrderedDict(style_description)
        if "color_palette" in normalized:
            normalized["color_palette"] = self._normalize_ideogram_color_palette(
                normalized["color_palette"], IDEOGRAM_JSON_STYLE_PALETTE_MAX
            )
        has_photo = "photo" in normalized
        has_art_style = "art_style" in normalized

        if not has_photo and not has_art_style:
            style_key = (
                "photo"
                if self._style_description_has_photo_cues(normalized)
                else "art_style"
            )
            normalized[style_key] = self._infer_style_description_value(
                normalized, style_key
            )
        elif has_photo and has_art_style:
            style_key = (
                "photo"
                if self._style_description_has_photo_cues(normalized)
                else "art_style"
            )
            drop_key = "art_style" if style_key == "photo" else "photo"
            normalized.pop(drop_key, None)

        if "photo" in normalized and "art_style" not in normalized:
            key_order = IDEOGRAM_JSON_STYLE_KEY_ORDER_PHOTO
        elif "art_style" in normalized and "photo" not in normalized:
            key_order = IDEOGRAM_JSON_STYLE_KEY_ORDER_NON_PHOTO
        else:
            return normalized

        return self._ordered_ideogram_dict(normalized, key_order)

    @staticmethod
    def _style_description_has_photo_cues(style_description: dict) -> bool:
        text_parts = []
        for key in ("medium", "photo", "art_style", "aesthetics", "lighting"):
            value = style_description.get(key)
            if isinstance(value, str):
                text_parts.append(value.lower())
        text = " ".join(text_parts)
        return any(
            re.search(rf"(?<![a-z0-9]){re.escape(cue)}(?![a-z0-9])", text)
            for cue in IDEOGRAM_JSON_PHOTO_CUES
        )

    @staticmethod
    def _infer_style_description_value(style_description: dict, style_key: str) -> str:
        for key in (style_key, "medium", "aesthetics"):
            value = style_description.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "photograph" if style_key == "photo" else "artwork"

    def _normalize_ideogram_compositional_deconstruction(self, composition):
        if not isinstance(composition, dict):
            return composition

        normalized = OrderedDict(composition)
        elements = normalized.get("elements")
        if isinstance(elements, list):
            normalized["elements"] = [
                self._normalize_ideogram_element(element) for element in elements
            ]

        return self._ordered_ideogram_dict(
            normalized, IDEOGRAM_JSON_COMPOSITION_KEY_ORDER
        )

    def _normalize_ideogram_element(self, element):
        if not isinstance(element, dict):
            return element

        element = OrderedDict(element)
        if "bbox" in element:
            element["bbox"] = self._normalize_ideogram_bbox(element["bbox"])
        if "color_palette" in element:
            element["color_palette"] = self._normalize_ideogram_color_palette(
                element["color_palette"], IDEOGRAM_JSON_ELEMENT_PALETTE_MAX
            )

        if element.get("type") == "text":
            if "text" not in element:
                element["text"] = ""
            key_order = IDEOGRAM_JSON_ELEMENT_KEY_ORDER_TEXT
        else:
            key_order = IDEOGRAM_JSON_ELEMENT_KEY_ORDER_OBJ
        return self._ordered_ideogram_dict(element, key_order)

    @staticmethod
    def _normalize_ideogram_color_palette(palette, max_colors: int):
        if not isinstance(palette, list):
            return palette

        normalized = []
        for color in palette[:max_colors]:
            if isinstance(color, str):
                stripped = color.strip()
                if re.fullmatch(r"#[0-9a-fA-F]{6}", stripped):
                    normalized.append(stripped.upper())
                    continue
            normalized.append(color)
        return normalized

    @staticmethod
    def _normalize_ideogram_bbox(bbox):
        if not isinstance(bbox, list) or len(bbox) != 4:
            return bbox

        values = []
        has_unit_scale_hint = False
        for value in bbox:
            if isinstance(value, bool):
                return bbox
            if isinstance(value, int):
                number = float(value)
            elif isinstance(value, float):
                number = value
                has_unit_scale_hint = True
            elif isinstance(value, str):
                try:
                    number = float(value.strip())
                except ValueError:
                    return bbox
                has_unit_scale_hint = True
            else:
                return bbox
            if not math.isfinite(number):
                return bbox
            values.append(number)

        if has_unit_scale_hint and all(0 <= value <= 1 for value in values):
            values = [value * 1000 for value in values]

        return [int(round(value)) for value in values]

    @staticmethod
    def _ordered_ideogram_dict(source: dict, key_order: tuple[str, ...]) -> OrderedDict:
        ordered = OrderedDict()
        for key in key_order:
            if key in source:
                ordered[key] = source[key]
        for key, value in source.items():
            if key not in ordered:
                ordered[key] = value
        return ordered

    @staticmethod
    def _ideogram_caption_verifier():
        try:
            from extensions_built_in.diffusion_models.ideogram4.src.caption_verifier import (
                CaptionVerifier,
            )

            return CaptionVerifier()
        except Exception:
            verifier_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "diffusion_models",
                "ideogram4",
                "src",
                "caption_verifier.py",
            )
            spec = importlib.util.spec_from_file_location(
                "aitk_ideogram_caption_verifier", verifier_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Could not load Ideogram caption verifier at {verifier_path}"
                )
            verifier_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(verifier_module)
            return verifier_module.CaptionVerifier()

    def _warn_for_ideogram_json_issues(self, parsed: dict, file_path: str):
        try:
            warnings = self._ideogram_caption_verifier().verify(parsed)
            if warnings:
                print(
                    f"Warning: Ideogram JSON caption verifier found issues for {file_path}:"
                )
                for warning in warnings:
                    print(f"  - {warning}")
        except Exception:
            return

    def delete_source_caption_for_file(self, file_path: str, saved_caption_path: str):
        if not self.is_ideogram_json_output():
            return
        if not self.caption_config.delete_source_caption:
            return
        source_ext = self.caption_config.source_caption_extension
        output_ext = self.caption_config.caption_extension
        if not source_ext or source_ext == output_ext:
            return
        source_path = f"{os.path.splitext(file_path)[0]}.{source_ext}"
        if os.path.abspath(source_path) == os.path.abspath(saved_caption_path):
            return
        try:
            if os.path.exists(source_path):
                os.remove(source_path)
        except Exception as exc:
            print(f"Warning: failed to remove source caption {source_path}: {exc}")

    def print_and_status_update(self, status: str):
        print(status)
        self.update_status("running", status)

    def find_files(self):
        if self.encrypted_reader is not None:
            items = self.encrypted_reader.list_items(extensions=self.caption_config.extensions)
            if not self.caption_config.recaption:
                items = [item for item in items if not item.captionObjectPath]
            self.encrypted_items_by_path = {self.encrypted_reader.virtual_path(item): item for item in items}
            self.file_paths = sorted(self.encrypted_items_by_path.keys())
            print(f"Found {len(self.file_paths)} encrypted files to caption")
            return
        # recursivly find all the files in the path_to_caption with the specified extensions and save the paths to self.file_paths
        for root, dirs, files in os.walk(self.caption_config.path_to_caption):
            dirs[:] = [d for d in dirs if d != "_controls"]
            for file in files:
                if any(
                    file.lower().endswith(f".{ext}") and not file.startswith(".")
                    for ext in self.caption_config.extensions
                ):
                    full_path = os.path.join(root, file)
                    self.file_paths.append(full_path)
        # sort
        self.file_paths.sort()
        # it not recaption, remove the ones with captions
        if not self.caption_config.recaption:
            filtered_file_paths = []
            for file_path in self.file_paths:
                filename_no_ext = os.path.splitext(file_path)[0]
                caption_file_path = (
                    f"{filename_no_ext}.{self.caption_config.caption_extension}"
                )
                if not os.path.exists(caption_file_path):
                    filtered_file_paths.append(file_path)
            print(
                f"Found {len(self.file_paths)} files. {len(filtered_file_paths)} need captioning."
            )
            self.file_paths = filtered_file_paths
        else:
            print(f"Found {len(self.file_paths)} files to caption")

    def load_model(self):
        raise NotImplementedError("Model loading not implemented for this captioner")

    def load_audio_tensor_for_caption(self, file_path: str):
        if self.encrypted_reader is not None:
            return self.encrypted_reader.load_audio_waveform(self.encrypted_items_by_path[file_path])
        return torchaudio.load(file_path)

    def start_stop_watcher(self, interval_sec: float = 5.0):
        """
        Start a daemon thread that periodically checks should_stop()
        and terminates the process immediately when triggered.
        """
        if not self.is_ui_captioner:
            return
        if getattr(self, "_stop_watcher_started", False):
            return
        self._stop_watcher_started = True
        t = threading.Thread(
            target=self._stop_watcher_thread, args=(interval_sec,), daemon=True
        )
        t.start()

    def _stop_watcher_thread(self, interval_sec: float):
        while True:
            try:
                if self.should_stop():
                    # Mark and update status (non-blocking; uses existing infra)
                    self.is_stopping = True
                    self._run_async_operation(
                        self._update_status("stopped", "Job stopped (remote)")
                    )
                    # Best-effort flush pending async ops
                    try:
                        asyncio.run(self.wait_for_all_async())
                    except RuntimeError:
                        pass
                    # Try to stop DB thread pool quickly
                    try:
                        self.thread_pool.shutdown(wait=False, cancel_futures=True)
                    except TypeError:
                        self.thread_pool.shutdown(wait=False)
                    print("")
                    print("****************************************************")
                    print("    Stop signal received; terminating process.      ")
                    print("****************************************************")
                    os.kill(os.getpid(), signal.SIGINT)
                time.sleep(interval_sec)
            except Exception:
                time.sleep(interval_sec)

    def _run_async_operation(self, coro):
        """Helper method to run an async coroutine and track the task."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Create a task and track it
        if loop.is_running():
            task = asyncio.run_coroutine_threadsafe(coro, loop)
            self._async_tasks.append(asyncio.wrap_future(task))
        else:
            task = loop.create_task(coro)
            self._async_tasks.append(task)
            loop.run_until_complete(task)

    async def _execute_db_operation(self, operation_func):
        """Execute a database operation in a separate thread with retry on lock."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, operation_func)

    def should_stop(self):
        if not self.is_ui_captioner:
            return False
        return self.ui_job_store.should_stop()

    def should_return_to_queue(self):
        if not self.is_ui_captioner:
            return False
        return self.ui_job_store.should_return_to_queue()

    def maybe_stop(self):
        if not self.is_ui_captioner:
            return
        if self.should_stop():
            self._run_async_operation(self._update_status("stopped", "Job stopped"))
            self.is_stopping = True
            raise JobStopRequested("Job stopped")
        if self.should_return_to_queue():
            self._run_async_operation(self._update_status("queued", "Job queued"))
            self.is_stopping = True
            raise JobStopRequested("Job returning to queue", return_to_queue=True)

    async def _update_key(self, key, value):
        def _do_update():
            self.ui_job_store.update_key(key, value)

        await self._execute_db_operation(_do_update)

    def update_step(self):
        """Non-blocking update of the step count."""
        if self.is_ui_captioner:
            self._run_async_operation(self._update_key("step", self.step_num))

    def update_db_key(self, key, value):
        """Non-blocking update a key in the database."""
        if self.is_ui_captioner:
            self._run_async_operation(self._update_key(key, value))

    async def _update_status(self, status: AITK_Status, info: Optional[str] = None):
        if not self.is_ui_captioner:
            return

        def _do_update():
            self.ui_job_store.update_status(status, info)

        await self._execute_db_operation(_do_update)

    def update_status(self, status: AITK_Status, info: Optional[str] = None):
        if self.is_ui_captioner:
            """Non-blocking update of status."""
            self._run_async_operation(self._update_status(status, info))

    def on_error(self, e: Exception):
        super(BaseCaptioner, self).on_error(e)
        if self.is_ui_captioner:
            try:
                if not self.is_stopping:
                    self.update_status("error", str(e))
                asyncio.run(self.wait_for_all_async())
            except Exception as db_err:
                print(
                    f"[AITK] Warning: failed to update DB during error handling: {db_err}"
                )
            finally:
                self.thread_pool.shutdown(wait=True)
                self.ui_job_store.close()

    async def wait_for_all_async(self):
        """Wait for all tracked async operations to complete."""
        if not self._async_tasks:
            return

        try:
            await asyncio.gather(*self._async_tasks)
        except Exception as e:
            pass
        finally:
            # Clear the task list after completion
            self._async_tasks.clear()
