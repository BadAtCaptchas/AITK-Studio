# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team.
# Copyright contributors to the vLLM-Omni project.
#
# Ming-specific sequence construction follows vLLM-Omni PR 8021, pinned to
# a62d2ec999ae8fa669e8c22194c67575c0f2d3dc. The execution uses differentiable
# diffusers blocks, rather than vLLM inference kernels. See UPSTREAM.md.
"""Native Ming DiT with direct VLM conditioning and independent image layers."""

from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_z_image import (
    ZImageTransformer2DModel as DiffusersZImageTransformer2DModel,
)

from toolkit.models.v2._mixin import OstrisModelMixin
from .quantization import register_ming_fp8

register_ming_fp8()


ALIGNMENT = 32


class MingImageTransformer2DModel(DiffusersZImageTransformer2DModel, OstrisModelMixin):
    """Keep the checkpoint's split q/k/v keys and exact two-stream conditioning.

    ``x`` contains noisy target frames (composite first for Design-Layer).
    ``ref_x`` is a clean reference appended after those frames. Returned
    predictions contain only target frames, never the reference.
    """

    aitk_subfolder = "transformer"
    aitk_config_repo = "inclusionAI/Ming-Image-0.1-Design"

    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=(32, 48, 48),
        axes_lens=(20480, 512, 512),
        alignment_padding_mode="zero_masked",
        multi_frame_output=False,
    ):
        if alignment_padding_mode not in ("zero_masked", "learned"):
            raise ValueError("Ming alignment_padding_mode must be 'zero_masked' or 'learned'")
        if not isinstance(multi_frame_output, bool):
            raise ValueError("Ming multi_frame_output must be a boolean")
        if (alignment_padding_mode == "learned") != multi_frame_output:
            raise ValueError("Ming Design uses zero_masked/single-frame; Design-Layer uses learned/multi-frame")
        super().__init__(
            all_patch_size=all_patch_size,
            all_f_patch_size=all_f_patch_size,
            in_channels=in_channels,
            dim=dim,
            n_layers=n_layers,
            n_refiner_layers=n_refiner_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            cap_feat_dim=cap_feat_dim,
            rope_theta=rope_theta,
            t_scale=t_scale,
            axes_dims=list(axes_dims),
            axes_lens=list(axes_lens),
        )
        self.alignment_padding_mode = alignment_padding_mode
        self.multi_frame_output = multi_frame_output
        if alignment_padding_mode == "zero_masked":
            # Design's checkpoint has no learned pad parameters.
            self.register_parameter("x_pad_token", None)
            self.register_parameter("cap_pad_token", None)

    @classmethod
    def get_transformer_block_names(cls) -> list[str]:
        return ["noise_refiner", "context_refiner", "layers"]

    @classmethod
    def get_quantization_exclude_modules(cls) -> list[str]:
        return ["t_embedder*", "cap_embedder*", "all_x_embedder*", "all_final_layer*"]

    def get_offload_ignore_modules(self) -> list[torch.Tensor]:
        return [token for token in (self.x_pad_token, self.cap_pad_token) if token is not None]

    @classmethod
    def aitk_from_pretrained(cls, path, subfolder=None, dtype=None, **kwargs):
        model, info = cls._local_first(
            cls.from_pretrained,
            path,
            subfolder=subfolder,
            torch_dtype=dtype,
            output_loading_info=True,
            **kwargs,
        )
        failures = {key: info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
        if failures:
            raise ValueError(f"Ming transformer checkpoint does not match its configuration: {failures}")
        return model

    def _prepare_ming_sequence(self, features, positions, padding, pad_token):
        """Pad samples while distinguishing alignment tokens from batch padding."""
        lengths = [len(item) for item in features]
        flat = torch.cat(features)
        inner_padding = torch.cat(padding)
        replacement = torch.zeros_like(flat) if pad_token is None else pad_token.expand_as(flat)
        flat = torch.where(inner_padding[:, None], replacement, flat)
        features = list(flat.split(lengths))
        frequencies = list(self.rope_embedder(torch.cat(positions)).split(lengths))
        mask = pad_sequence(
            [~item if self.alignment_padding_mode == "zero_masked" else torch.ones_like(item) for item in padding],
            batch_first=True,
            padding_value=False,
        )
        return (
            pad_sequence(features, batch_first=True),
            pad_sequence(frequencies, batch_first=True),
            mask,
            lengths,
        )

    def _run_blocks(self, blocks, features, mask, frequencies, modulation=None):
        for block in blocks:
            args = (features, mask, frequencies, modulation)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                features = self._gradient_checkpointing_func(block, *args)
            else:
                features = block(*args)
        return features

    def forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        cap_feats: list[torch.Tensor],
        patch_size: int = 2,
        f_patch_size: int = 1,
        ref_x: Optional[list[Optional[torch.Tensor]]] = None,
        cap_feats_2: Optional[list[torch.Tensor]] = None,
        return_dict: bool = True,
    ):
        if not x or len(x) != len(cap_feats):
            raise ValueError("Ming requires equally sized nonempty image and caption batches")
        if patch_size not in self.all_patch_size or f_patch_size not in self.all_f_patch_size:
            raise ValueError("Unsupported Ming spatial or frame patch size")
        batch_size = len(x)
        if ref_x is not None and len(ref_x) != batch_size:
            raise ValueError("Ming reference batch must match target batch")
        if cap_feats_2 is not None and len(cap_feats_2) != batch_size:
            raise ValueError("Ming direct condition batch must match target batch")
        refs = [None] * batch_size if ref_x is None else ref_x
        direct = [None] * batch_size if cap_feats_2 is None else cap_feats_2
        device = x[0].device
        time = t.to(device=device).reshape(-1)
        if time.numel() == 1:
            time = time.expand(batch_size)
        if time.numel() != batch_size:
            raise ValueError("Ming timestep batch must match target batch")

        images, captions, image_ids, caption_ids = [], [], [], []
        image_padding, caption_padding, sizes, target_frames = [], [], [], []
        for target, caption, extra, reference in zip(x, cap_feats, direct, refs):
            if target.ndim != 4 or target.shape[0] != self.in_channels:
                raise ValueError("Ming target must have shape [latent_channels, frames, height, width]")
            if not self.multi_frame_output and target.shape[1] != 1:
                raise ValueError("Ming Design accepts one target frame")
            if caption.ndim != 2 or not len(caption) or caption.shape[1] != self.config.cap_feat_dim:
                raise ValueError("Ming primary condition must have shape [tokens, cap_feat_dim]")
            if extra is not None and (extra.ndim != 2 or extra.shape[1] != self.dim):
                raise ValueError("Ming direct condition must have shape [tokens, dim]")
            target_frames.append(target.shape[1])
            if reference is not None:
                if reference.ndim != 4 or reference.shape != (target.shape[0], 1, *target.shape[-2:]):
                    raise ValueError("Ming reference must be one frame matching target channels and resolution")
                target = torch.cat((target, reference), dim=1)
            channels, frames, height, width = target.shape
            if frames % f_patch_size or height % patch_size or width % patch_size:
                raise ValueError("Ming latent geometry must be divisible by the patch size")
            if min(frames, height, width) <= 0:
                raise ValueError("Ming latent dimensions must be positive")
            sizes.append((frames, height, width))
            caption_length = len(caption) + (len(extra) if extra is not None else 0)
            caption_pad = (-caption_length) % ALIGNMENT
            caption_ids.append(self.create_coordinate_grid(
                (caption_length + caption_pad, 1, 1), start=(1, 0, 0), device=device,
            ).flatten(0, 2))
            caption_padding.append(torch.arange(caption_length + caption_pad, device=device) >= caption_length)
            # Direct features already have DiT width; never pass them through
            # cap_embedder, which accepts the query connector's 2560 channels.
            caption = self.cap_embedder(caption)
            if extra is not None:
                caption = torch.cat((caption, extra), dim=0)
            captions.append(torch.cat((caption, caption.new_zeros((caption_pad, self.dim)))))

            grid = (frames // f_patch_size, height // patch_size, width // patch_size)
            patches = target.reshape(channels, grid[0], f_patch_size, grid[1], patch_size, grid[2], patch_size)
            patches = patches.permute(1, 3, 5, 2, 4, 6, 0).reshape(-1, f_patch_size * patch_size * patch_size * channels)
            length = len(patches)
            padding = (-length) % ALIGNMENT
            patches = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](patches)
            images.append(torch.cat((patches, patches.new_zeros((padding, self.dim)))))
            image_padding.append(torch.arange(length + padding, device=device) >= length)
            ids = self.create_coordinate_grid(grid, start=(caption_length + caption_pad + 1, 0, 0), device=device).flatten(0, 2)
            image_ids.append(torch.cat((ids, ids.new_zeros((padding, 3)))))

        image, image_freqs, image_mask, image_lengths = self._prepare_ming_sequence(images, image_ids, image_padding, self.x_pad_token)
        caption, caption_freqs, caption_mask, caption_lengths = self._prepare_ming_sequence(captions, caption_ids, caption_padding, self.cap_pad_token)
        modulation = self.t_embedder(time * self.t_scale).to(dtype=image.dtype)
        image = self._run_blocks(self.noise_refiner, image, image_mask, image_freqs, modulation)
        caption = self._run_blocks(self.context_refiner, caption, caption_mask, caption_freqs)

        unified, frequencies, masks = [], [], []
        for index, (image_length, caption_length) in enumerate(zip(image_lengths, caption_lengths)):
            unified.append(torch.cat((image[index, :image_length], caption[index, :caption_length])))
            frequencies.append(torch.cat((image_freqs[index, :image_length], caption_freqs[index, :caption_length])))
            masks.append(torch.cat((image_mask[index, :image_length], caption_mask[index, :caption_length])))
        unified = self._run_blocks(
            self.layers,
            pad_sequence(unified, batch_first=True),
            pad_sequence(masks, batch_first=True, padding_value=False),
            pad_sequence(frequencies, batch_first=True),
            modulation,
        )
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, modulation)
        output = []
        for tokens, (frames, height, width), keep_frames in zip(unified, sizes, target_frames):
            grid = (frames // f_patch_size, height // patch_size, width // patch_size)
            image = tokens[:grid[0] * grid[1] * grid[2]].reshape(*grid, f_patch_size, patch_size, patch_size, self.out_channels)
            image = image.permute(6, 0, 3, 1, 4, 2, 5).reshape(self.out_channels, frames, height, width)
            output.append(image[:, :keep_frames])
        return Transformer2DModelOutput(sample=output) if return_dict else (output,)
