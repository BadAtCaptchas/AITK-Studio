"""Bounded decode-only lookahead for cold latent caches.

Random crops, augmentation, transforms and model execution stay on the caller.
One decoder and one pending image avoid a dataset-sized queue or GPU prefetch.
"""
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from toolkit.image_io import _finalize_static_image, ensure_jpegxl_plugin_for_source

MAX_STAGED_PIXELS = 4 * 1024 * 1024


def can_stage_latent_image(item):
    return not any(getattr(item, flag, False) for flag in (
        'is_video', 'is_audio_model', 'is_layered', 'is_encrypted',
        'has_control_image', 'has_inpaint_image', 'has_clip_image',
        'has_mask_image', 'has_unconditional', 'is_text_embedding_cached',
        'dopsd_self_ref', 'control_path', 'control_video_paths',
    ))


def decode_latent_image(item):
    """Return a detached image or None for the serial large-image fallback."""
    ensure_jpegxl_plugin_for_source(item.path)
    with Image.open(item.path) as image:
        # Check the actual opened header, not stale dataset dimensions. Restrict
        # modes so retained storage is at most four bytes per source pixel.
        if image.width * image.height > MAX_STAGED_PIXELS or image.mode not in ('1', 'L', 'LA', 'P', 'RGB', 'RGBA'):
            return None
        return _finalize_static_image(
            image, item.path,
            mode='RGBA' if item.use_alpha_as_mask else None,
            require_alpha=item.use_alpha_as_mask,
        )


class LatentImageStager:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.executor = None
        self.pending = None

    def __enter__(self):
        return self

    def submit(self, item):
        if self.enabled and item is not None and can_stage_latent_image(item):
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='latent-decode')
            self.pending = self.executor.submit(decode_latent_image, item)

    def take_and_stage(self, next_item):
        pending, self.pending = self.pending, None
        try:
            return pending.result() if pending is not None else None
        except BaseException:
            if pending is not None and not pending.cancel():
                pending.add_done_callback(_close_unused_image)
            raise
        finally:
            # Also advance after a corrupt decode: failures retain their exact
            # item identity and do not shift the next image onto the wrong item.
            self.submit(next_item)

    def __exit__(self, *_exc):
        pending, self.pending = self.pending, None
        if pending is not None:
            pending.cancel()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        if pending is not None:
            _close_unused_image(pending)


def _close_unused_image(pending):
    if not pending.cancelled():
        try:
            image = pending.result()
            if image is not None:
                image.close()
        except BaseException:
            # Cleanup must preserve the owner's exception/cancellation.
            pass


def mark_cached_survivors(file_list):
    """The filtered list is authoritative; duplicate paths/objects are safe."""
    for item in file_list:
        item.is_latent_cached = True
