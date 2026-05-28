import unittest

import torch

from extensions_built_in.diffusion_models.flux2.sega import _radial_energy, build_flux2_sega_rope_scale
from extensions_built_in.diffusion_models.flux2.src.model import Flux2, Flux2Params


class Flux2SegaScaleTest(unittest.TestCase):
    def test_scale_builder_shape_identity_and_clamp(self):
        latents = torch.randn(2, 8, 80, 64)
        axes_dim = [2, 4, 4, 2]

        scale = build_flux2_sega_rope_scale(
            latents,
            token_count=5,
            axes_dim=axes_dim,
            base_resolution=32,
            strength=1.0,
            min_scale=0.5,
            max_scale=2.0,
            vae_scale_factor=16,
        )

        self.assertEqual(tuple(scale.shape), (2, 5, sum(axes_dim)))
        self.assertTrue(torch.isfinite(scale).all())
        self.assertGreaterEqual(float(scale.min()), 0.5)
        self.assertLessEqual(float(scale.max()), 2.0)
        self.assertTrue(torch.allclose(scale[..., :2], torch.ones_like(scale[..., :2])))
        self.assertTrue(torch.allclose(scale[..., -2:], torch.ones_like(scale[..., -2:])))

        identity = build_flux2_sega_rope_scale(
            latents,
            token_count=5,
            axes_dim=axes_dim,
            base_resolution=4096,
            strength=1.0,
            vae_scale_factor=16,
        )
        self.assertTrue(torch.allclose(identity, torch.ones_like(identity)))

    def test_scale_builder_boosts_low_energy_bands(self):
        torch.manual_seed(2)
        latents = torch.randn(1, 4, 32, 32)
        axes_dim = [2, 16, 16, 2]

        energy = _radial_energy(latents, bins=8)[0]
        scale = build_flux2_sega_rope_scale(
            latents,
            token_count=1,
            axes_dim=axes_dim,
            base_resolution=480,
            strength=1.0,
            min_scale=0.1,
            max_scale=3.0,
            vae_scale_factor=16,
        )
        h_axis_pair_scale = scale[0, 0, 2:18:2]

        self.assertLess(
            float(h_axis_pair_scale[energy.argmax()]),
            float(h_axis_pair_scale[energy.argmin()]),
        )

    def test_flux2_forward_preserves_output_for_none_and_ones_scale(self):
        torch.manual_seed(1)
        params = Flux2Params(
            in_channels=4,
            context_in_dim=6,
            hidden_size=16,
            num_heads=2,
            depth=1,
            depth_single_blocks=1,
            axes_dim=[2, 2, 2, 2],
            mlp_ratio=1.0,
            use_guidance_embed=True,
        )
        model = Flux2(params).eval()
        x = torch.randn(1, 4, 4)
        x_ids = torch.tensor([[[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]]])
        ctx = torch.randn(1, 2, 6)
        ctx_ids = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 1]]])
        timesteps = torch.tensor([0.5])
        guidance = torch.tensor([1.0])

        with torch.no_grad():
            baseline = model(x, x_ids, timesteps, ctx, ctx_ids, guidance)
            ones = model(
                x,
                x_ids,
                timesteps,
                ctx,
                ctx_ids,
                guidance,
                sega_rope_scale=torch.ones(1, 4, 8),
            )
            scaled = model(
                x,
                x_ids,
                timesteps,
                ctx,
                ctx_ids,
                guidance,
                sega_rope_scale=torch.full((1, 4, 8), 1.2),
            )

        self.assertTrue(torch.allclose(baseline, ones, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.isfinite(scaled).all())
        self.assertFalse(torch.allclose(baseline, scaled))


if __name__ == "__main__":
    unittest.main()
