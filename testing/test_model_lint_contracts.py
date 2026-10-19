"""Real runtime contracts repaired while checking extension Pylint errors."""
import unittest
from unittest.mock import patch

import torch
from diffusers.models.attention_processor import Attention

import extensions_built_in.diffusion_models
from extensions_built_in.diffusion_models.boogu_image.src import attention_processor as boogu
from extensions_built_in.diffusion_models.boogu_image.src.transformer import BooguImageTransformer2DModel
from extensions_built_in.diffusion_models.example_model.example_model import ExampleModel
from extensions_built_in.diffusion_models.example_model.src.model import ExampleTransformer2DModel
from extensions_built_in.diffusion_models.example_model.src.pipeline import ExamplePipeline, pad_prompt_embeds
from extensions_built_in.diffusion_models.hidream.src.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model as LegacyIdeogram4Model
from extensions_built_in.diffusion_models.ideogram4.ideogram4_model import Ideogram4Model
from extensions_built_in.diffusion_models.ideogram4.src import pipeline as legacy_ideogram


class ModelLintContractsTest(unittest.TestCase):
    def test_boogu_attention_preserves_diffusers_keyword_contract(self):
        processor = boogu.BooguImageAttnProcessor()
        attention = Attention(query_dim=8, heads=2, kv_heads=1, dim_head=4, processor=processor)
        inputs = torch.randn(2, 3, 8)
        rotary = object()
        with patch.object(boogu.NativeAttention, "__call__", return_value=inputs) as call:
            self.assertIs(attention(inputs, encoder_hidden_states=inputs, image_rotary_emb=rotary), inputs)
            self.assertIs(call.call_args.kwargs["image_rotary_emb"], rotary)

    def test_boogu_standalone_processor_matches_vendored_native(self):
        torch.manual_seed(17)
        attention = Attention(query_dim=8, heads=2, kv_heads=1, dim_head=4)
        inputs = torch.randn(2, 3, 8)
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
        expected = boogu.NativeAttention()(attention, inputs, inputs, mask)
        actual = boogu.BooguImageAttnProcessor()(attention, inputs, inputs, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_boogu_double_stream_dispatch_keeps_projection_parameters(self):
        processor = boogu.BooguImageDoubleStreamSelfAttnProcessor(4, 2, 1)
        before = {name: id(value) for name, value in processor.named_parameters()}
        sentinel = torch.randn(1, 3, 8)
        for backend, implementation in (("native", boogu.NativeDoubleStreamAttention),
                                        ("flash", boogu.FlashDoubleStreamAttention)):
            processor.attention_backend = backend
            with patch.object(boogu, "_FLASH_ATTN_AVAILABLE", True), \
                    patch.object(implementation, "__call__", return_value=sentinel) as call:
                self.assertIs(processor("attention", "image", "instruction"), sentinel)
                self.assertEqual(call.call_count, 1)
        self.assertEqual(before, {name: id(value) for name, value in processor.named_parameters()})
        with patch.object(boogu, "_FLASH_ATTN_AVAILABLE", False):
            with self.assertRaisesRegex(ImportError, "flash_attn"):
                processor(None, None, None)

    def test_boogu_alternate_transformer_constructs_without_weights(self):
        model = BooguImageTransformer2DModel(
            hidden_size=24, num_attention_heads=2, num_kv_heads=1,
            num_layers=1, num_double_stream_layers=1, num_refiner_layers=0,
            in_channels=4, multiple_of=8, axes_dim_rope=(4, 4, 4),
            axes_lens=(16, 16, 16),
            instruction_feature_configs={"instruction_feat_dim": 8, "reduce_type": "mean",
                                         "num_instruction_feat_layers": 1},
        )
        model.set_attention_backend("native")
        self.assertEqual(model.attention_backend, "native")

    def test_example_template_helpers_and_sampler_contract(self):
        self.assertEqual(ExampleModel.arch, "example")
        features = [torch.randn(3, 4), torch.randn(1, 4)]
        padded, mask = pad_prompt_embeds(features, "cpu", torch.float32)
        self.assertEqual(tuple(padded.shape), (2, 3, 4))
        torch.testing.assert_close(padded[0], features[0])
        self.assertEqual(mask.tolist(), [[True, True, True], [True, False, False]])
        model = ExampleTransformer2DModel(in_channels=2, out_channels=2, hidden_size=8,
                                         num_heads=2, num_layers=1, text_dim=4)
        output = model(torch.randn(2, 2, 4, 4), torch.tensor([0.2, 0.8]), padded, mask)
        self.assertEqual(tuple(output.shape), (2, 2, 4, 4))
        self.assertTrue(torch.isfinite(output).all())
        with self.assertRaisesRegex(NotImplementedError, "integration template"):
            ExamplePipeline(None)(None)

    def test_legacy_ideogram_import_has_actionable_contract_and_shape_helpers(self):
        self.assertEqual(Ideogram4Model.arch, LegacyIdeogram4Model.arch)
        latents = torch.randn(2, 32, 6, 8)
        patched = legacy_ideogram.patchify_latents(latents, 2)
        self.assertEqual(tuple(patched.shape), (2, 128, 3, 4))
        torch.testing.assert_close(legacy_ideogram.unpatchify_latents(patched, 2), latents)
        with self.assertRaisesRegex(NotImplementedError, "ideogram4_model"):
            legacy_ideogram.Ideogram4Pipeline(None)
        with self.assertRaisesRegex(NotImplementedError, "ideogram4_model"):
            legacy_ideogram.predict_velocity(None, None, None, None, None)

    def test_unipc_all_orders_and_restarts_are_finite(self):
        for order in (1, 2, 3):
            scheduler = FlowUniPCMultistepScheduler(solver_order=order)
            torch.manual_seed(32)
            initial = torch.randn(1, 2, 3, 3)
            results = []
            for _ in range(2):
                scheduler.set_timesteps(5)
                sample = initial.clone()
                for timestep in scheduler.timesteps:
                    sample = scheduler.step(sample * 0.1, timestep, sample).prev_sample
                    self.assertTrue(torch.isfinite(sample).all())
                results.append(sample)
            torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
