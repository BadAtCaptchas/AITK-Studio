import json
import re
from collections import OrderedDict


MAX_IMAGE_PALETTE = 16
MAX_ELEMENT_PALETTE = 5

MEDIUM_OPTIONS = (
    "photograph",
    "illustration",
    "3d_render",
    "painting",
    "graphic_design",
)

_MEDIUM_ALIASES = {
    "photograph": "photograph",
    "photo": "photograph",
    "illustration": "illustration",
    "3d render": "3d_render",
    "3d_render": "3d_render",
    "3d-render": "3d_render",
    "3drender": "3d_render",
    "render": "3d_render",
    "3d": "3d_render",
    "painting": "painting",
    "graphic design": "graphic_design",
    "graphic_design": "graphic_design",
    "graphic-design": "graphic_design",
    "graphic": "graphic_design",
}

_HEX6_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_HEX3_RE = re.compile(r"^#[0-9a-fA-F]{3}$")
_PHOTO_CUES = (
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


def canon_medium(medium):
    if not isinstance(medium, str):
        return medium
    key = medium.strip().rstrip(".").strip().lower()
    return _MEDIUM_ALIASES.get(key, medium.strip())


def is_photo_medium(medium):
    return canon_medium(medium) == "photograph"


def normalize_hex(color):
    if not isinstance(color, str):
        return None
    value = color.strip()
    if _HEX6_RE.match(value):
        return "#" + value[1:].upper()
    if _HEX3_RE.match(value):
        return "#" + "".join(ch * 2 for ch in value[1:]).upper()
    return None


def sanitize_palette(palette, max_len):
    if not isinstance(palette, (list, tuple)):
        return None
    seen = set()
    colors = []
    for color in palette:
        normalized = normalize_hex(color)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        colors.append(normalized)
        if len(colors) >= max_len:
            break
    return colors or None


def _style_has_photo_cues(style):
    text_parts = []
    for key in ("medium", "photo", "art_style", "aesthetics", "lighting"):
        value = style.get(key)
        if isinstance(value, str):
            text_parts.append(value.lower())
    text = " ".join(text_parts)
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(cue)}(?![a-z0-9])", text)
        for cue in _PHOTO_CUES
    )


def _infer_style_value(style, style_key):
    for key in (style_key, "medium", "aesthetics"):
        value = style.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "photograph" if style_key == "photo" else "artwork"


def normalize_style(style):
    if not isinstance(style, dict):
        return style

    raw_medium = style.get("medium")
    medium = canon_medium(raw_medium) if raw_medium is not None else None
    has_photo = "photo" in style and style.get("photo") is not None
    has_art = "art_style" in style and style.get("art_style") is not None

    if medium in MEDIUM_OPTIONS:
        photo_branch = medium == "photograph"
    elif has_art and not has_photo:
        photo_branch = False
    elif has_photo and not has_art:
        photo_branch = True
    else:
        photo_branch = _style_has_photo_cues(style)

    photo_value = style.get("photo") if has_photo else None
    art_value = style.get("art_style") if has_art else None
    style_key = "photo" if photo_branch else "art_style"
    render_value = (
        (photo_value if photo_branch else art_value)
        or (art_value if photo_branch else photo_value)
        or _infer_style_value(style, style_key)
    )

    out = OrderedDict()
    if "aesthetics" in style:
        out["aesthetics"] = style["aesthetics"]
    if "lighting" in style:
        out["lighting"] = style["lighting"]

    if photo_branch:
        out["photo"] = render_value
        if medium is not None:
            out["medium"] = medium
    else:
        if medium is not None:
            out["medium"] = medium
        out["art_style"] = render_value

    palette = sanitize_palette(style.get("color_palette"), MAX_IMAGE_PALETTE)
    if palette is not None:
        out["color_palette"] = palette

    known = {
        "aesthetics",
        "lighting",
        "photo",
        "art_style",
        "medium",
        "color_palette",
    }
    for key, value in style.items():
        if key not in known:
            out[key] = value
    return out


def normalize_element(element):
    if not isinstance(element, dict):
        return element

    element_type = element.get("type", "obj")
    out = OrderedDict()
    out["type"] = element_type
    if element.get("bbox") is not None:
        out["bbox"] = element["bbox"]
    if element_type == "text":
        out["text"] = element.get("text", "")
        if "desc" in element:
            out["desc"] = element["desc"]
    elif "desc" in element:
        out["desc"] = element["desc"]

    palette = sanitize_palette(element.get("color_palette"), MAX_ELEMENT_PALETTE)
    if palette is not None:
        out["color_palette"] = palette

    for key, value in element.items():
        if key not in out and key != "color_palette":
            out[key] = value
    return out


def normalize_caption_dict(data):
    if not isinstance(data, dict):
        return data

    data = OrderedDict(data)
    data.pop("aspect_ratio", None)

    out = OrderedDict()
    if "high_level_description" in data:
        out["high_level_description"] = data["high_level_description"]
    if "style_description" in data:
        out["style_description"] = normalize_style(data["style_description"])

    composition = data.get("compositional_deconstruction")
    if isinstance(composition, dict):
        normalized_composition = OrderedDict()
        if "background" in composition:
            normalized_composition["background"] = composition["background"]
        elements = composition.get("elements")
        if isinstance(elements, list):
            normalized_composition["elements"] = [
                normalize_element(element) for element in elements
            ]
        for key, value in composition.items():
            if key not in ("background", "elements"):
                normalized_composition[key] = value
        out["compositional_deconstruction"] = normalized_composition
    elif composition is not None:
        out["compositional_deconstruction"] = composition

    for key, value in data.items():
        if key not in out and key != "aspect_ratio":
            out[key] = value
    return out


def is_ideogram_caption_str(text):
    value = (text or "").strip()
    if not value.startswith("{"):
        return False
    try:
        data = json.loads(value)
    except Exception:
        return False
    return isinstance(data, dict) and isinstance(
        data.get("compositional_deconstruction"), dict
    )


def to_model_string(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def digest_caption_string(text):
    value = (text or "").strip()
    if not value.startswith("{"):
        return text
    try:
        data = json.loads(value, object_pairs_hook=OrderedDict)
    except Exception:
        return text
    if not (
        isinstance(data, dict)
        and isinstance(data.get("compositional_deconstruction"), dict)
    ):
        return text
    return to_model_string(normalize_caption_dict(data))
