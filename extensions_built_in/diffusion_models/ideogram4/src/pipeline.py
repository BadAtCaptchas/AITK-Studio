"""Actionable contract for the retired standalone Ideogram integration.

The registered ``ideogram4_model.Ideogram4Model`` owns the maintained sampler.
The old holder expected a different transformer and text-feature interface;
silently substituting the current sampler would produce incompatible tensors.
"""

from ..ideogram4_model import patchify_latents, unpatchify_latents
from typing import NoReturn

__all__ = [
    "Ideogram4Pipeline", "get_qwen3_vl_features", "pad_text_features",
    "patchify_latents", "predict_velocity", "unpatchify_latents",
]


def _unsupported_legacy_pipeline() -> NoReturn:
    raise NotImplementedError(
        "The legacy ideogram4.ideogram4 holder is unsupported. Use the registered arch='ideogram4' "
        "implementation in extensions_built_in.diffusion_models.ideogram4.ideogram4_model instead."
    )


class Ideogram4Pipeline:
    def __init__(self, model):
        _unsupported_legacy_pipeline()

    def __call__(self, *args, **kwargs) -> NoReturn:
        _unsupported_legacy_pipeline()


def get_qwen3_vl_features(text_encoder, token_ids, attention_mask, position_ids) -> NoReturn:
    return _unsupported_legacy_pipeline()


def pad_text_features(features, device, dtype) -> NoReturn:
    return _unsupported_legacy_pipeline()


def predict_velocity(transformer, latents, timesteps, features, attention_mask) -> NoReturn:
    return _unsupported_legacy_pipeline()
