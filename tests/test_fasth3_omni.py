"""CPU-only Omni/PDD regressions; no model weights are downloaded."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from extensions_built_in.diffusion_models.minimax_h3 import (
    MinimaxH3Model, MinimaxH3Ref2VAModel, MinimaxH3FastModel,
    MinimaxH3FastV2Model, MinimaxH3OmniModel,
)
from extensions_built_in.diffusion_models.minimax_h3.src import packing
from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (
    MiniMaxH3FinalLayer, MiniMaxH3Transformer, MiniMaxH3TransformerParams,
    _blend_pdd_heads,
)
from toolkit.config_modules import ModelConfig, TrainConfig, SaveConfig, NetworkConfig, DatasetConfig, validate_configs
from toolkit.model_registry import resolve_model
from toolkit.lora_special import LoRAModule
from testing.test_minimax_h3_smoke import _SmokeNetwork


class PDDTests(unittest.TestCase):
    def test_training_boundaries_and_eight_step_spans(self):
        self.assertEqual(packing.pdd_span(1, 32, 12), (0, 1))
        self.assertEqual(packing.pdd_span(0, 32, 12), (31, 32))
        for i in range(32):
            sigma = packing.shift_sigma(1 - (i + .5) / 32, 12)
            self.assertEqual(packing.pdd_span(sigma, 32, 12), (i, i + 1))
        ladder = [packing.shift_sigma(t / 1000, 12) for t in (999, 874, 750, 625, 500, 375, 250, 125, 0)]
        for i, (sigma, nxt) in enumerate(zip(ladder, ladder[1:])):
            self.assertEqual(packing.pdd_span(sigma, 32, 12, nxt), (i * 4, (i + 1) * 4))
        for sigma, nxt in ((-1, None), (float('nan'), None), (1.1, None), (.2, .3), (.2, -1)):
            with self.assertRaises(ValueError):
                packing.pdd_span(sigma, 32, 12, nxt)

    def test_weights_use_each_stream_shift(self):
        video, audio = packing.build_pdd_weights([1, .5], 32, 12, sigma_v_next=[.9, 0])
        for weights in (video, audio):
            self.assertEqual(weights.shape, (2, 32))
            self.assertTrue(torch.isfinite(weights).all())
            self.assertTrue((weights >= 0).all())
            self.assertTrue((weights[:, 0] == 1).all())
        self.assertFalse(torch.equal(video, audio))
        start, stop = packing.pdd_span(.5, 32, 12, 0)
        for weights, shift in ((video, 12), (audio, 3)):
            grid = torch.linspace(1, 0, 33, dtype=torch.float64)
            durations = (1 - packing.shift_sigma(grid, shift)).diff()[start:stop]
            torch.testing.assert_close(weights[1, start:stop].double(), durations / durations.sum())
        for args in (([], 32, 12), ([1], 0, 12), ([1], 32, 0)):
            with self.assertRaises(ValueError):
                packing.build_pdd_weights(*args)
        with self.assertRaises(ValueError):
            packing.build_pdd_weights([1], 32, 12, sigma_v_next=[])
        with self.assertRaises(ValueError):
            packing.pdd_head_weights(0, 33, 32, 12)

    def test_blending_shapes_empty_rows_and_gradients(self):
        out = torch.randn(2, 5, 12, requires_grad=True)
        weights = torch.tensor([[1., 0., 1.], [1., 1., 0.]])
        result = _blend_pdd_heads(out, weights, 3)
        expected = out.reshape(2, 5, 3, 4)
        torch.testing.assert_close(result[0], expected[0, :, 0] + expected[0, :, 2])
        result.sum().backward()
        self.assertTrue((out.grad[0, :, 4:8] == 0).all())
        self.assertEqual(_blend_pdd_heads(torch.empty(2, 0, 12), weights, 3).shape, (2, 0, 4))
        with self.assertRaises(ValueError):
            _blend_pdd_heads(out, torch.ones(1, 3), 3)

    def test_final_layer_single_head_parity_and_lora_hooks(self):
        for heads in (1, 32):
            params = MiniMaxH3TransformerParams(hidden_size=8, time_embed_dim=4, latents_dim=2,
                audio_latents_dim=2, pdd_steps=heads)
            layer = MiniMaxH3FinalLayer(params)
            x, temb = torch.randn(2, 6, 8), torch.randn(2, 4)
            indices = torch.tensor([[0] * 6, [1] * 6])
            vi, ai = torch.tensor([0, 3, 5]), torch.tensor([1, 4])
            weights = None if heads == 1 else packing.build_pdd_weights([1, .2], heads, 12)
            if heads == 1:
                shift, scale = layer.adaln_proj(temb)
                h = layer.norm(x) * (1 + scale[indices]) + shift[indices]
                expected = layer.video_out(h)[:, vi], layer.audio_out(h)[:, ai]
                actual = layer(x, temb, indices, vi, ai, weights)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b)
            else:
                with self.assertRaisesRegex(ValueError, 'pdd_weights'):
                    layer(x, temb, indices, vi, ai)
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
            network = _SmokeNetwork(torch.device('cpu'))
            lora = LoRAModule('video', layer.video_out, lora_dim=2, alpha=2, network=network)
            lora.apply_to()
            video, audio = layer(x, temb, indices, vi, ai, weights)
            self.assertEqual(video.shape, (2, 3, 8))
            self.assertEqual(audio.shape, (2, 2, 2))
            video.square().mean().backward()
            self.assertIsNotNone(lora.lora_up.weight.grad)
            self.assertGreater(lora.lora_up.weight.grad.abs().sum().item(), 0)

    def test_tiny_transformer_checkpoint_roundtrip_and_both_streams(self):
        for heads in (1, 32):
            params = MiniMaxH3TransformerParams(hidden_size=8, num_layers=1,
                token_refiner_num_layers=1, num_attention_heads=1, attention_head_dim=8,
                ffn_hidden_size=16, latents_dim=2, audio_latents_dim=2, text_dim=6,
                timestep_input_dim=4, time_embed_hidden_size=8, time_embed_dim=4,
                rope_inv_freq_len=1, pdd_steps=heads)
            transformer = MiniMaxH3Transformer(params)
            restored = MiniMaxH3Transformer(params)
            restored.load_state_dict(transformer.state_dict(), strict=True)
            inputs = dict(hidden_states=torch.randn(2, 3, 8), audio_hidden_states=torch.randn(2, 2, 2),
                encoder_hidden_states=torch.randn(2, 1, 6), row_timesteps=torch.full((2, 6), .4),
                token_tags=torch.tensor([[0, 1, 2, 0, 2, 0]] * 2), position_ids=torch.zeros(2, 6, 3),
                video_indices=torch.tensor([0, 3, 5]), audio_indices=torch.tensor([2, 4]), text_indices=torch.tensor([1]),
                pdd_weights=None if heads == 1 else packing.build_pdd_weights([.6, .6], heads, 12))
            output = transformer(**inputs)
            for actual, expected in zip(output, restored(**inputs)):
                torch.testing.assert_close(actual, expected)
            sum(x.square().mean() for x in output).backward()
            for layer in (transformer.final_layer.video_out, transformer.final_layer.audio_out):
                self.assertTrue(torch.isfinite(layer.weight.grad).all())
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_checkpoint_inference_rejects_mismatched_heads(self):
        for heads in (1, 32):
            state = {'final_layer.video_out.weight': torch.empty(heads * 96, 8),
                'final_layer.audio_out.weight': torch.empty(heads * 32, 8),
                'adaln_t_table': torch.empty(1000, 8)}
            params = MinimaxH3Model._transformer_params_for_state(state)
            self.assertEqual(params.pdd_steps, heads)
            self.assertEqual(params.time_embed_dim, 8)
            state['final_layer.audio_out.weight'] = torch.empty(3, 8)
            with self.assertRaisesRegex(ValueError, 'head counts'):
                MiniMaxH3Transformer.aitk_config_from_state_dict(state)
        with self.assertRaisesRegex(ValueError, 'video output'):
            MiniMaxH3Transformer.aitk_config_from_state_dict({'final_layer.video_out.weight': torch.empty(97, 8)})


class OmniIntegrationTests(unittest.TestCase):
    def model(self, cls=MinimaxH3OmniModel, **kwargs):
        return cls('cpu', ModelConfig(arch=cls.arch, name_or_path='test', model_kwargs=kwargs))

    def test_registry_defaults_reference_identity_and_legacy_models(self):
        omni, ref = self.model(), self.model(MinimaxH3Ref2VAModel)
        self.assertIs(resolve_model(omni.arch), MinimaxH3OmniModel)
        self.assertIsNone(omni.vsa_sparsity)
        omni.model = SimpleNamespace(params=SimpleNamespace(pdd_steps=32))
        weights = omni._pdd_weights(torch.tensor([1., .5]))
        self.assertEqual(weights[0].shape, (2, 32))
        omni.model.params.pdd_steps = 1
        self.assertIsNone(omni._pdd_weights(torch.tensor([1.])))
        noise, clean = torch.randn(1, 2), torch.randn(1, 2)
        torch.testing.assert_close(omni.get_loss_target(noise=noise, batch=SimpleNamespace(latents=clean)), noise - clean)
        self.assertTrue(omni.has_multiple_control_images)
        self.assertTrue(omni.supports_video_control_images)
        self.assertEqual(omni.text_embedding_space_version, ref.text_embedding_space_version)
        presented = self.model(image_refs_as_video=True, image_ref_video_frames=22)
        self.assertNotEqual(omni.text_embedding_space_version, presented.text_embedding_space_version)
        self.assertEqual(presented.text_embedding_space_version, self.model(MinimaxH3Ref2VAModel,
            image_refs_as_video=True, image_ref_video_frames=22).text_embedding_space_version)
        legacy, v2 = self.model(MinimaxH3FastModel), self.model(MinimaxH3FastV2Model)
        self.assertEqual([legacy.video_sigma_shift, v2.video_sigma_shift, omni.video_sigma_shift], [12, 10, 12])
        self.assertEqual(len({x._dit_component() for x in (omni, legacy, v2)}), 3)
        self.assertEqual([legacy.vsa_sparsity, v2.vsa_sparsity], [.9, .8])
        with patch.object(MinimaxH3Ref2VAModel, '_load_transformer', return_value=SimpleNamespace(params=SimpleNamespace(pdd_steps=1))):
            with self.assertRaisesRegex(ValueError, 'stacked'):
                omni._load_transformer()

    def test_config_guards_and_dopsd_adapter_exclusivity(self):
        def validate(train=None, model=None, network=None, datasets=None):
            validate_configs(TrainConfig(**(train or {})), ModelConfig(arch='minimax_h3_omni', name_or_path='test', **(model or {})),
                SaveConfig(), datasets or [], NetworkConfig(**(network or {'type': 'lora'})))
        validate()
        validate(model={'model_kwargs': {'dopsd': True}})
        for train, model, network in [({'train_text_encoder': True}, {}, {}), ({}, {}, {'type': 'lokr'}),
                ({}, {'model_kwargs': {'dopsd': True}, 'assistant_lora_path': 'adapter'}, {}),
                ({'do_guidance_loss': True}, {'model_kwargs': {'dopsd': True}}, {}),
                ({'diff_output_preservation': True}, {'model_kwargs': {'dopsd': True}}, {}),
                ({}, {'model_kwargs': {'image_ref_video_frames': 6}}, {})]:
            with self.assertRaises(ValueError):
                validate(train, model, network)
        with self.assertRaisesRegex(ValueError, 'reference'):
            validate(datasets=[DatasetConfig(num_frames=39, do_i2v=True)])
