import io

from PIL import Image as PILImage
from PIL.ImageOps import exif_transpose


class UnsupportedAnimatedImageError(ValueError):
    pass


def image_has_alpha(image: PILImage.Image) -> bool:
    return "A" in image.getbands() or (image.mode == "P" and "transparency" in image.info)


def _finalize_static_image(
    image: PILImage.Image,
    source: str,
    mode: str = None,
    require_alpha: bool = False,
) -> PILImage.Image:
    image_format = image.format
    if image_format == "WEBP" and (getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) > 1):
        raise UnsupportedAnimatedImageError(
            f"Animated WebP is not supported for image datasets yet: {source}. "
            "Use a video dataset format or convert it to static WebP/PNG/JPEG."
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
    with PILImage.open(io.BytesIO(data)) as image:
        return _finalize_static_image(
            image,
            source,
            mode=mode,
            require_alpha=require_alpha,
        )
