import io
import os
from typing import Optional

from PIL import Image as PILImage
from PIL.ImageOps import exif_transpose


class UnsupportedAnimatedImageError(ValueError):
    pass


class JpegXLSupportError(RuntimeError):
    pass


JPEGXL_EXTENSIONS = {".jxl"}
JPEGXL_FORMATS = {"JXL", "JPEGXL", "JPEG XL"}
JPEGXL_MISSING_DEPENDENCY_MESSAGE = (
    "JPEG XL (.jxl) support requires pillow-jxl-plugin. "
    "Install it with `pip install pillow-jxl-plugin==1.3.7` and restart the process."
)
_jpegxl_plugin_available: Optional[bool] = None


def register_jpegxl_plugin() -> bool:
    global _jpegxl_plugin_available
    if _jpegxl_plugin_available is not None:
        return _jpegxl_plugin_available
    try:
        import pillow_jxl  # noqa: F401
        _jpegxl_plugin_available = True
    except Exception:
        _jpegxl_plugin_available = False
    return _jpegxl_plugin_available


def is_jpegxl_header(data: bytes) -> bool:
    return data.startswith(b"\xff\x0a") or data.startswith(b"\x00\x00\x00\x0cJXL \r\n\x87\n")


def is_jpegxl_path(path: str) -> bool:
    return os.path.splitext(str(path).split("?", 1)[0])[1].lower() in JPEGXL_EXTENSIONS


def ensure_jpegxl_plugin_for_source(source: str = None, data: bytes = None) -> None:
    if (source and is_jpegxl_path(source)) or (data is not None and is_jpegxl_header(data)):
        if not register_jpegxl_plugin():
            raise JpegXLSupportError(JPEGXL_MISSING_DEPENDENCY_MESSAGE)


register_jpegxl_plugin()


def image_has_alpha(image: PILImage.Image) -> bool:
    return "A" in image.getbands() or (image.mode == "P" and "transparency" in image.info)


def _finalize_static_image(
    image: PILImage.Image,
    source: str,
    mode: str = None,
    require_alpha: bool = False,
) -> PILImage.Image:
    image_format = image.format
    is_animated = getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) > 1
    if image_format == "WEBP" and is_animated:
        raise UnsupportedAnimatedImageError(
            f"Animated WebP is not supported for image datasets yet: {source}. "
            "Use a video dataset format or convert it to static WebP/PNG/JPEG."
        )
    if image_format in JPEGXL_FORMATS and is_animated:
        raise UnsupportedAnimatedImageError(
            f"Animated JPEG XL is not supported for image datasets yet: {source}. "
            "Use a video dataset format or convert it to static JXL/PNG/JPEG."
        )

    image = exif_transpose(image)
    if require_alpha and not image_has_alpha(image):
        raise ValueError(f"Image requires an alpha channel but none was found: {source}")
    if mode is not None:
        image = image.convert(mode)
    image.load()
    return image.copy()


def open_static_image(
    path: str,
    mode: str = None,
    require_alpha: bool = False,
) -> PILImage.Image:
    ensure_jpegxl_plugin_for_source(path)
    with PILImage.open(path) as image:
        return _finalize_static_image(
            image,
            path,
            mode=mode,
            require_alpha=require_alpha,
        )


def open_static_image_from_bytes(
    data: bytes,
    source: str = "<bytes>",
    mode: str = None,
    require_alpha: bool = False,
) -> PILImage.Image:
    ensure_jpegxl_plugin_for_source(source, data)
    with PILImage.open(io.BytesIO(data)) as image:
        return _finalize_static_image(
            image,
            source,
            mode=mode,
            require_alpha=require_alpha,
        )


def save_static_image(image: PILImage.Image, path: str, **save_kwargs) -> None:
    ensure_jpegxl_plugin_for_source(path)
    image.save(path, **save_kwargs)
