import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
    QwenImageDecoder3d,
    QwenImageEncoder3d,
)


_original_encoder_forward = QwenImageEncoder3d.forward
_original_decoder_forward = QwenImageDecoder3d.forward
_patched = False


def _checkpoint(module, value):
    return torch.utils.checkpoint.checkpoint(module, value, use_reentrant=False)


def _cache_is_fresh(feat_cache):
    return feat_cache is None or all(value is None for value in feat_cache)


def _encoder_forward(self, value, feat_cache=None, feat_idx=[0]):
    if not (
        self.gradient_checkpointing
        and torch.is_grad_enabled()
        and _cache_is_fresh(feat_cache)
    ):
        return _original_encoder_forward(self, value, feat_cache, feat_idx)

    value = self.conv_in(value)
    for layer in self.down_blocks:
        value = _checkpoint(layer, value)
    value = _checkpoint(self.mid_block, value)
    value = self.norm_out(value)
    value = self.nonlinearity(value)
    return self.conv_out(value)


def _decoder_forward(self, value, feat_cache=None, feat_idx=[0]):
    if not (
        self.gradient_checkpointing
        and torch.is_grad_enabled()
        and _cache_is_fresh(feat_cache)
    ):
        return _original_decoder_forward(self, value, feat_cache, feat_idx)

    value = self.conv_in(value)
    value = _checkpoint(self.mid_block, value)
    for layer in self.up_blocks:
        value = _checkpoint(layer, value)
    value = self.norm_out(value)
    value = self.nonlinearity(value)
    return self.conv_out(value)


def patch_qwen_vae_gradient_checkpointing():
    global _patched
    if _patched:
        return
    _patched = True

    AutoencoderKLQwenImage._supports_gradient_checkpointing = True
    QwenImageEncoder3d.forward = _encoder_forward
    QwenImageDecoder3d.forward = _decoder_forward
