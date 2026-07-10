import pathlib
import unittest
from types import SimpleNamespace

import torch

from extensions_built_in.diffusion_models.wan22.wan22_14b_model import Wan2214bModel
from extensions_built_in.diffusion_models.wan22.wan22_5b_model import Wan225bModel
from toolkit.models.vae_tiling import temporary_vae_tiling
from toolkit.models.wan21.wan21 import Wan21
from toolkit.models.wan21.wan_utils import add_first_frame_conditioning_v22


ROOT = pathlib.Path(__file__).resolve().parents[1]


class FakeVAE:
    def __init__(self):
        self.use_tiling = False
        self.enable_calls = 0
        self.disable_calls = 0
        self.config = SimpleNamespace(
            scale_factor_spatial=1,
            latents_mean=[0.0, 0.0, 0.0],
            latents_std=[1.0, 1.0, 1.0],
        )

    def enable_tiling(self):
        self.enable_calls += 1
        self.use_tiling = True

    def disable_tiling(self):
        self.disable_calls += 1
        self.use_tiling = False

    def encode(self, value):
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: value))


class July10UpstreamPortsTest(unittest.TestCase):
    def test_temporary_vae_tiling_restores_state_after_success_and_failure(self):
        vae = FakeVAE()
        with temporary_vae_tiling(vae, True):
            self.assertTrue(vae.use_tiling)
        self.assertFalse(vae.use_tiling)
        self.assertEqual((vae.enable_calls, vae.disable_calls), (1, 1))

        vae.use_tiling = True
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with temporary_vae_tiling(vae, True):
                raise RuntimeError("boom")
        self.assertTrue(vae.use_tiling)
        self.assertEqual((vae.enable_calls, vae.disable_calls), (1, 1))

    def test_wan_vae_tiling_policy_supports_low_vram_and_explicit_kwarg(self):
        model = object.__new__(Wan21)
        model.model_config = SimpleNamespace(low_vram=False, model_kwargs={})
        self.assertFalse(model.use_vae_tiling)
        model.model_config.model_kwargs["vae_tiling"] = True
        self.assertTrue(model.use_vae_tiling)
        model.model_config.model_kwargs.clear()
        model.model_config.low_vram = True
        self.assertTrue(model.use_vae_tiling)

    def test_cached_first_frame_matches_raw_encoded_conditioning(self):
        vae = FakeVAE()
        latent = torch.randn(2, 3, 4, 8, 8)
        first_frame = torch.randn(2, 3, 8, 8)
        cached = first_frame.unsqueeze(2)

        raw_latent, raw_mask = add_first_frame_conditioning_v22(
            latent,
            vae=vae,
            first_frame=first_frame,
        )
        cached_latent, cached_mask = add_first_frame_conditioning_v22(
            latent,
            vae=vae,
            first_frame_latents=cached,
        )

        self.assertTrue(torch.equal(raw_latent, cached_latent))
        self.assertTrue(torch.equal(raw_mask, cached_mask))

    def test_cached_first_frame_validation_rejects_ambiguous_or_mismatched_input(self):
        vae = FakeVAE()
        latent = torch.zeros(2, 3, 4, 8, 8)
        frame = torch.zeros(2, 3, 8, 8)
        cached = frame.unsqueeze(2)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            add_first_frame_conditioning_v22(
                latent,
                vae=vae,
                first_frame=frame,
                first_frame_latents=cached,
            )
        with self.assertRaisesRegex(ValueError, "spatial"):
            add_first_frame_conditioning_v22(
                latent,
                vae=vae,
                first_frame_latents=torch.zeros(2, 3, 1, 4, 4),
            )

    def test_wan_sensitive_layers_are_excluded_from_quantization(self):
        expected = ["condition_embedder*", "proj_out*"]
        self.assertEqual(Wan225bModel.get_quantization_exclude_modules(object()), expected)
        self.assertEqual(Wan2214bModel.get_quantization_exclude_modules(object()), expected)

    def test_qwen_variants_use_exception_safe_low_vram_tiling(self):
        for relative in (
            "extensions_built_in/diffusion_models/qwen_image/qwen_image.py",
            "extensions_built_in/diffusion_models/qwen_image/qwen_image_edit.py",
            "extensions_built_in/diffusion_models/qwen_image/qwen_image_edit_plus.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(
                "with temporary_vae_tiling(pipeline.vae, self.model_config.low_vram):",
                source,
            )

    def test_wan_5b_prefers_cached_first_frame_latents(self):
        source = (
            ROOT
            / "extensions_built_in/diffusion_models/wan22/wan22_5b_model.py"
        ).read_text(encoding="utf-8")
        self.assertIn("if batch.first_frame_latents is not None:", source)
        self.assertIn("first_frame_latents=batch.first_frame_latents", source)

    def test_wan_5b_raw_image_batch_uses_first_frame_fallback(self):
        model = object.__new__(Wan225bModel)
        model.device_torch = torch.device("cpu")
        model.torch_dtype = torch.float32
        model.vae = FakeVAE()
        model.model = lambda **kwargs: (kwargs["hidden_states"],)
        batch = SimpleNamespace(
            dataset_config=SimpleNamespace(do_i2v=True),
            first_frame_latents=None,
            tensor=torch.randn(1, 3, 8, 8),
        )
        latent = torch.randn(1, 3, 4, 8, 8)

        output = model.get_noise_prediction(
            latent,
            torch.tensor([1.0]),
            SimpleNamespace(text_embeds=torch.empty(1)),
            batch,
        )

        self.assertTrue(torch.equal(output[:, :, :1], batch.tensor.unsqueeze(2)))
        self.assertIsNotNone(model._i2v_loss_mask)


if __name__ == "__main__":
    unittest.main()
