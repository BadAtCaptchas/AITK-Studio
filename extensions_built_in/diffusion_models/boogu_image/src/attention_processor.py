"""Backend-switchable processors for the standalone Boogu transformer.

Both implementations use the vendored projections and attention algorithms;
switching backend does not replace trainable parameters or checkpoint names.
"""

from functools import wraps

from .boogu.models.attention_processor import (
    BooguImageAttnProcessor as NativeAttention,
    BooguImageAttnProcessorFlash2Varlen as FlashAttention,
    BooguImageDoubleStreamSelfAttnProcessor as NativeDoubleStreamAttention,
    BooguImageDoubleStreamSelfAttnProcessorFlash2Varlen as FlashDoubleStreamAttention,
)
from .boogu.utils.import_utils import is_flash_attn_available

ATTENTION_BACKENDS = ("native", "flash")
_FLASH_ATTN_AVAILABLE = is_flash_attn_available()


def _check_backend(backend):
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(f"Unknown Boogu attention backend: {backend!r}")
    if backend == "flash" and not _FLASH_ATTN_AVAILABLE:
        raise ImportError("Boogu Flash attention requires the optional flash_attn package")


class BooguImageAttnProcessor(NativeAttention):
    attention_backend = "native"
    _upad_input = FlashAttention._upad_input

    @wraps(NativeAttention.__call__)
    def __call__(self, *args, **kwargs):
        _check_backend(self.attention_backend)
        implementation = FlashAttention if self.attention_backend == "flash" else NativeAttention
        return implementation.__call__(self, *args, **kwargs)


class BooguImageDoubleStreamSelfAttnProcessor(NativeDoubleStreamAttention):
    attention_backend = "native"
    _upad_input = FlashDoubleStreamAttention._upad_input

    @wraps(NativeDoubleStreamAttention.__call__)
    def __call__(self, *args, **kwargs):
        _check_backend(self.attention_backend)
        implementation = FlashDoubleStreamAttention if self.attention_backend == "flash" else NativeDoubleStreamAttention
        return implementation.__call__(self, *args, **kwargs)
