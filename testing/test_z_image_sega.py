import unittest

import torch

from extensions_built_in.diffusion_models.z_image.sega import (
    _radial_energy,
    apply_zimage_sega_rope_scale,
    build_zimage_sega_rope_scale,
)


class _FakeZImageTransformer:
    def __init__(self):
        self.x_pad_token = object()
        self.cap_pad_token = object()

    def _prepare_sequence(
        self,
        feats,
        pos_ids,
        inner_pad_mask,
        pad_token,
        noise_mask=None,
        device=None,
    ):
        batch_size = len(feats)
        seq_len = max(len(feat) for feat in feats)
        freqs_cis = torch.ones((batch_size, seq_len, 5), dtype=torch.complex64)
        return torch.zeros(batch_size, seq_len, 1), freqs_cis, None, [len(feat) for feat in feats], None


class ZImageSegaScaleTest(unittest.TestCase):
    def test_scale_builder_shape_identity_axes_and_base_resolution(self):
        latents = torch.randn(2, 8, 80, 64)
        axes_dims = [2, 4, 4]

        scale = build_zimage_sega_rope_scale(
            latents,
            axes_dims,
            base_resolution=32,
            strength=1.0,
            min_scale=0.5,
            max_scale=2.0,
            vae_scale_factor=8,
        )

        self.assertEqual(tuple(scale.shape), (2, 5))
        self.assertTrue(torch.isfinite(scale).all())
        self.assertGreaterEqual(float(scale.min()), 0.5)
        self.assertLessEqual(float(scale.max()), 2.0)
        self.assertTrue(torch.allclose(scale[:, :1], torch.ones_like(scale[:, :1])))

        identity = build_zimage_sega_rope_scale(
            latents,
            axes_dims,
            base_resolution=4096,
            strength=1.0,
            vae_scale_factor=8,
        )
        self.assertTrue(torch.allclose(identity, torch.ones_like(identity)))

    def test_scale_builder_boosts_low_energy_bands(self):
        torch.manual_seed(3)
        latents = torch.randn(1, 4, 64, 64)
        axes_dims = [2, 16, 16]

        energy = _radial_energy(latents, bins=8)[0]
        scale = build_zimage_sega_rope_scale(
            latents,
            axes_dims,
            base_resolution=256,
            strength=1.0,
            min_scale=0.1,
            max_scale=3.0,
            vae_scale_factor=8,
        )
        h_axis_pair_scale = scale[0, 1:9]

        self.assertLess(
            float(h_axis_pair_scale[energy.argmax()]),
            float(h_axis_pair_scale[energy.argmin()]),
        )

    def test_prepare_sequence_context_scales_only_image_rope(self):
        transformer = _FakeZImageTransformer()
        scale = torch.tensor(
            [
                [1.0, 1.1, 1.2, 1.3, 1.4],
                [1.0, 0.9, 0.8, 0.7, 0.6],
            ]
        )
        feats = [torch.zeros(3, 1), torch.zeros(3, 1)]
        masks = [torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)]
        pos_ids = [torch.zeros(3, 3, dtype=torch.int32), torch.zeros(3, 3, dtype=torch.int32)]

        with apply_zimage_sega_rope_scale(transformer, scale):
            _, image_freqs, _, _, _ = transformer._prepare_sequence(
                feats,
                pos_ids,
                masks,
                transformer.x_pad_token,
            )
            _, cap_freqs, _, _, _ = transformer._prepare_sequence(
                feats,
                pos_ids,
                masks,
                transformer.cap_pad_token,
            )

        _, restored_freqs, _, _, _ = transformer._prepare_sequence(
            feats,
            pos_ids,
            masks,
            transformer.x_pad_token,
        )

        self.assertTrue(torch.allclose(image_freqs.real, scale[:, None, :].expand_as(image_freqs.real)))
        self.assertTrue(torch.allclose(cap_freqs.real, torch.ones_like(cap_freqs.real)))
        self.assertTrue(torch.allclose(restored_freqs.real, torch.ones_like(restored_freqs.real)))


if __name__ == "__main__":
    unittest.main()
