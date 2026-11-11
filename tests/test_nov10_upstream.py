"""Weight-free CPU regressions for MiniMax-H3-Image and shared H3 contracts."""
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from extensions_built_in.diffusion_models import MinimaxH3ImageModel
from extensions_built_in.diffusion_models.minimax_h3 import MinimaxH3Model
from extensions_built_in.diffusion_models.minimax_h3.minimax_h3_image import H3_IMAGE_SAMPLE_SIGMAS
from extensions_built_in.diffusion_models.minimax_h3.src import packing
from extensions_built_in.diffusion_models.minimax_h3.src.pipeline import MiniMaxH3Pipeline
from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (
    MiniMaxH3FinalLayer, MiniMaxH3Transformer, MiniMaxH3TransformerParams,
)
from extensions_built_in.diffusion_models.minimax_h3.src.vae import MiniMaxH3ImageVideoVAE, ViTDecoder3d
from toolkit.config_modules import (
    DatasetConfig, ModelConfig, NetworkConfig, SaveConfig, TrainConfig, validate_configs,
)
from toolkit.dto import DTO
from toolkit.model_registry import resolve_model
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


def tiny_params(**overrides):
    values = dict(hidden_size=8, num_layers=1, token_refiner_num_layers=0,
        num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=16,
        latents_dim=2, audio_latents_dim=2, text_dim=8, timestep_input_dim=4,
        time_embed_hidden_size=8, time_embed_dim=4, rope_inv_freq_len=1,
        audio_enabled=False, pdd_steps=6)
    values.update(overrides)
    return MiniMaxH3TransformerParams(**values)


class ImageSigmaTests(unittest.TestCase):
    def test_fixed_training_grid_and_continuous_schedulers_are_independent(self):
        scheduler = MinimaxH3ImageModel.get_train_scheduler()
        for mode in ('shift', 'sigmoid', 'shifted_logit_normal', 'two_step'):
            result = scheduler.set_train_timesteps(1000, 'cpu', timestep_type=mode)
            torch.testing.assert_close(result, torch.tensor(H3_IMAGE_SAMPLE_SIGMAS[:-1]) * 1000)
            torch.testing.assert_close(scheduler.sigmas, torch.tensor(H3_IMAGE_SAMPLE_SIGMAS))
        ordinary = CustomFlowMatchEulerDiscreteScheduler()
        self.assertIsNone(ordinary.fixed_train_sigmas)
        ordinary.set_train_timesteps(1000, 'cpu', timestep_type='linear')
        self.assertEqual(len(ordinary.timesteps), 1000)
        self.assertIsNone(MinimaxH3Model.get_train_scheduler().fixed_train_sigmas)

    def test_fixed_training_grid_rejects_invalid_schedules(self):
        for grid in ([], [1, 0], [1, .5, .5], [.5, 1], [1, float('nan')],
                     [float('inf')], [1.1], [-.1], [[1, .5]]):
            with self.subTest(grid=grid):
                scheduler = MinimaxH3ImageModel.get_train_scheduler()
                scheduler.fixed_train_sigmas = grid
                with self.assertRaisesRegex(ValueError, 'fixed_train_sigmas'):
                    scheduler.set_train_timesteps(1000, 'cpu')

    def test_sampling_grid_and_single_head_training_weights(self):
        model = object.__new__(MinimaxH3ImageModel)
        torch.testing.assert_close(model.build_sample_sigmas(6), torch.tensor(H3_IMAGE_SAMPLE_SIGMAS))
        sampled = model.build_sample_sigmas(12)
        self.assertEqual(sampled.shape, (13,))
        torch.testing.assert_close(sampled[::2], torch.tensor(H3_IMAGE_SAMPLE_SIGMAS))
        for index, sigma in enumerate(H3_IMAGE_SAMPLE_SIGMAS[:-1]):
            weights, audio_weights = model._pdd_weights(torch.tensor([sigma]))
            expected = torch.zeros(1, 6)
            expected[0, 0] = 1
            expected[0, index] = 1
            torch.testing.assert_close(weights, expected)
            torch.testing.assert_close(audio_weights, expected)
        weights = packing.build_pdd_grid_weights([H3_IMAGE_SAMPLE_SIGMAS[1]], H3_IMAGE_SAMPLE_SIGMAS,
                                                 [H3_IMAGE_SAMPLE_SIGMAS[3]])
        span = H3_IMAGE_SAMPLE_SIGMAS[1] - H3_IMAGE_SAMPLE_SIGMAS[3]
        self.assertAlmostEqual(float(weights[0, 1]), (H3_IMAGE_SAMPLE_SIGMAS[1] - H3_IMAGE_SAMPLE_SIGMAS[2]) / span)
        self.assertAlmostEqual(float(weights[0, 2]), (H3_IMAGE_SAMPLE_SIGMAS[2] - H3_IMAGE_SAMPLE_SIGMAS[3]) / span)
        self.assertEqual(float(weights[0, 0]), 1.)

    def test_fixed_grid_ignores_legacy_weights_and_next_sample_stepping(self):
        scheduler = MinimaxH3ImageModel.get_train_scheduler()
        scheduler.set_train_timesteps(1000, 'cpu')
        for mode in ('linear', 'weighted', 'next_sample'):
            for v2 in (False, True):
                for x0_pred in (False, True):
                    result = scheduler.get_weights_for_timesteps(scheduler.timesteps,
                        v2=v2, timestep_type=mode, x0_pred=x0_pred)
                    torch.testing.assert_close(result, torch.ones(6))
        source = Path(__file__).resolve().parents[1] / 'extensions_built_in/sd_trainer/SDTrainer.py'
        module = ast.parse(source.read_text(encoding='utf-8'))
        matches = [node for node in ast.walk(module) if isinstance(node, ast.If)
                   and "self.train_config.timestep_type == 'next_sample'" in ast.unparse(node.test)]
        self.assertEqual(len(matches), 1)
        predicate = compile(ast.Expression(matches[0].test), str(source), 'eval')
        owner = SimpleNamespace(train_config=SimpleNamespace(timestep_type='next_sample'),
                                sd=SimpleNamespace(noise_scheduler=scheduler))
        self.assertFalse(eval(predicate, {'self': owner}))
        scheduler.fixed_train_sigmas = None
        self.assertTrue(eval(predicate, {'self': owner}))

    def test_grid_and_reference_bounds_reject_invalid_values(self):
        for grid in ([], [1], [1, 1], [0, 1], [1, float('nan'), 0], [float('inf'), 0]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                packing.resample_sigma_grid(grid, 6)
        for steps in (0, -1, 1.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                packing.resample_sigma_grid(H3_IMAGE_SAMPLE_SIGMAS, steps)
        for value in (float('nan'), float('inf'), -.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                packing.build_pdd_grid_weights([value], H3_IMAGE_SAMPLE_SIGMAS)
        for args in ((0, 32, 32, 64), (32, -1, 32, 64), (32, 32, 0, 64),
                     (32, 32, 64, 32), (32, 32, 33, 64)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                packing.bound_short_edge_size(*args)


class ImageArchitectureTests(unittest.TestCase):
    def test_no_audio_weights_and_final_head_backward(self):
        params = tiny_params()
        transformer = MiniMaxH3Transformer(params)
        self.assertIsNone(transformer.audio_patch_proj)
        self.assertIsNone(transformer.final_layer.audio_out)
        self.assertFalse(any('audio_patch_proj' in key or 'audio_out' in key for key in transformer.state_dict()))
        final = MiniMaxH3FinalLayer(params)
        x = torch.randn(1, 2, 8, requires_grad=True)
        weights = torch.tensor([[1., 0, 1, 0, 0, 0]])
        video, audio = final(x, torch.randn(1, 4), torch.zeros(1, 2, dtype=torch.long),
            torch.tensor([0, 1]), torch.empty(0, dtype=torch.long), (weights, weights))
        self.assertEqual(video.shape, (1, 2, 8))
        self.assertEqual(audio.shape, (1, 0, 2))
        video.sum().backward()
        self.assertIsNotNone(final.video_out.weight.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        detected = MiniMaxH3Transformer.aitk_config_from_state_dict({
            'final_layer.video_out.weight': torch.zeros(96 * 6, 8),
        })
        self.assertFalse(detected.audio_enabled)
        self.assertEqual(detected.pdd_steps, 6)
        with_audio = MiniMaxH3Transformer(tiny_params(audio_enabled=True))
        self.assertIsNotNone(with_audio.audio_patch_proj)
        self.assertIsNotNone(with_audio.final_layer.audio_out)

    def test_tiny_image_transformer_forward_backward_with_padded_text(self):
        transformer = MiniMaxH3Transformer(tiny_params())
        video = torch.randn(1, 2, 8, requires_grad=True)
        text = torch.randn(1, 2, 8, requires_grad=True)
        weights = torch.tensor([[1., 0, 1, 0, 0, 0]])
        result, audio = transformer(hidden_states=video,
            audio_hidden_states=torch.empty(1, 0, 2), encoder_hidden_states=text,
            row_timesteps=torch.full((1, 4), .5), token_tags=torch.tensor([[0, 1, 0, -1]]),
            position_ids=torch.zeros(1, 4, 3), video_indices=torch.tensor([0, 2]),
            audio_indices=torch.empty(0, dtype=torch.long), text_indices=torch.tensor([1, 3]),
            pdd_weights=(weights, weights))
        self.assertEqual(result.shape, video.shape)
        self.assertEqual(audio.shape, (1, 0, 2))
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(video.grad).all())
        self.assertTrue(torch.isfinite(text.grad).all())
        self.assertGreater(float(transformer.final_layer.video_out.weight.grad.abs().sum()), 0)

    def test_image_decoder_uses_two_axis_rope_and_one_frame(self):
        with torch.device('meta'):
            vae = MiniMaxH3ImageVideoVAE(decoder_num_layers=0)
        self.assertEqual(vae.decoder.rope_axes, 2)
        self.assertEqual(vae.decoder.patch_size_t, 1)
        self.assertIsNone(vae.decoder.fixed_time_coord)
        with self.assertRaisesRegex(ValueError, 'single-frame'):
            vae.decode(torch.zeros(1, 24, 2, 1, 1))
        decoder = ViTDecoder3d(2, 3, 2, 1, 1, 1, 8, rope_dim_ratio=.5, rope_axes=2)
        output = decoder(torch.randn(1, 2, 1, 2, 3))
        self.assertEqual(output.shape, (1, 3, 1, 4, 6))
        self.assertTrue(torch.isfinite(output).all())

    def test_reference_sizing_and_conditioning_identity(self):
        model = object.__new__(MinimaxH3ImageModel)
        model.reference_min_short_edge = 1024
        model.reference_max_short_edge = 2048
        self.assertEqual(model._reference_pixel_size(512, 256, 2048, 2048), (1024, 2048))
        self.assertEqual(model._reference_pixel_size(512, 256, 4096, 3072), (1024, 2048))
        self.assertEqual(model._reference_pixel_size(4096, 8192, 512, 512), (4096, 2048))
        self.assertEqual(model._present_image_control(Image.new('RGB', (256, 512))).size, (1024, 2048))
        self.assertEqual(model.text_embedding_space_version, 'minimax_h3_image')
        self.assertEqual(model._prompt_encoder_kwargs(), {'picture_label': 'Picture {}: '})
        self.assertEqual(model._image_ref_video_frames(), 0)
        self.assertEqual(model._num_audio_latents(1), 0)

    def test_prediction_has_no_audio_context_or_stale_target(self):
        model = object.__new__(MinimaxH3ImageModel)
        model.device_torch = torch.device('cpu')
        model.torch_dtype = torch.float32
        model.max_text_length = 512
        seen = []
        class Transformer(torch.nn.Module):
            device = torch.device('cpu')
            def forward(self, **kwargs):
                seen.append(kwargs)
                return torch.zeros_like(kwargs['hidden_states']), torch.zeros_like(kwargs['audio_hidden_states'])
        model.model = Transformer()
        model._build_condition = lambda *args: (None, None, (), ())
        video = torch.zeros(1, 24, 1, 2, 2)
        batch = SimpleNamespace(num_frames=1, dataset_config=SimpleNamespace(do_audio=False),
            audio_noise=None, audio_latents=torch.ones(1, 4, 32), latents=video)
        fields = ('audio_target', 'audio_pred', 'audio_pred_uncond', 'audio_pred_prior',
                  'audio_pred_preservation', 'audio_noisy', 'audio_sigma', 'audio_loss_mask')
        for field in fields:
            setattr(batch, field, torch.ones(1))
        embeds = SimpleNamespace(text_embeds=[torch.zeros(1, 8)], text_token_tags=[torch.zeros(1, dtype=torch.long)])
        result = model.get_noise_prediction(DTO(video, audio=torch.ones(1, 4, 32)), torch.tensor([750.]), embeds, batch)
        self.assertEqual(result.shape, video.shape)
        self.assertEqual(seen[0]['audio_hidden_states'].shape[1], 0)
        self.assertEqual(seen[0]['audio_indices'].numel(), 0)
        self.assertFalse((seen[0]['token_tags'] == 2).any())
        for field in fields:
            self.assertIsNone(getattr(batch, field))

    def test_real_pipeline_has_no_audio_rows_and_respects_sigma_override(self):
        model = object.__new__(MinimaxH3ImageModel)
        model.device_torch = torch.device('cpu')
        model.torch_dtype = torch.float32
        model.max_text_length = 512
        seen = []
        class Transformer(torch.nn.Module):
            def forward(self, **kwargs):
                seen.append(kwargs)
                return torch.zeros_like(kwargs['hidden_states']), torch.zeros_like(kwargs['audio_hidden_states'])
        model.model = Transformer()
        model._park_vae_for_denoise = Mock()
        model._stage_for_sampling = Mock()
        model._stage_for_decode = Mock()
        model.decode_latents = lambda value: torch.zeros(1, 3, 1, 32, 32)
        model.decode_audio_latents = Mock(side_effect=AssertionError('image must not decode audio'))
        embeds = SimpleNamespace(text_embeds=[torch.zeros(1, 8)], text_token_tags=[torch.zeros(1, dtype=torch.long)])
        pipeline = MiniMaxH3Pipeline(model)
        for override, count in ((None, 6), ([1., .5], 2)):
            seen.clear()
            result = pipeline(embeds, height=32, width=32, num_frames=1, num_inference_steps=6,
                latents=torch.zeros(1, 24, 1, 2, 2), sigmas=override)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].size, (32, 32))
            self.assertEqual(len(seen), count)
            for call in seen:
                self.assertEqual(call['audio_hidden_states'].shape, (1, 0, 32))
                self.assertEqual(call['audio_indices'].numel(), 0)
                self.assertFalse((call['token_tags'] == 2).any())
        model.decode_audio_latents.assert_not_called()

    def test_resolver_never_chooses_incompatible_precision(self):
        model = object.__new__(MinimaxH3ImageModel)
        model.model_config = SimpleNamespace(model_kwargs={}, name_or_path='Comfy-Org/MiniMax-H3-Image')
        model.print_and_status_update = Mock()
        prefix = 'extensions_built_in.diffusion_models.minimax_h3.minimax_h3_image'
        expected = {
            'dit': 'diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors',
            'text_encoder': 'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
            'video_vae': 'vae/minimax_h3_image_vae_fp16.safetensors',
        }
        for component, filename in expected.items():
            with self.subTest(component=component), patch(prefix + '.resolve_comfy_file', side_effect=[None, 'downloaded']) as resolve:
                self.assertEqual(model._resolve_comfy_file(component), 'downloaded')
                self.assertEqual([call.args[0] for call in resolve.call_args_list], [filename, filename])
                self.assertTrue(resolve.call_args_list[0].kwargs['local_only'])
            with patch(prefix + '.resolve_comfy_file', return_value='local') as resolve:
                self.assertEqual(model._resolve_comfy_file(component), 'local')
                resolve.assert_called_once()

    def test_lazy_public_class_and_validation_preserve_studio_safety(self):
        self.assertIs(resolve_model('minimax_h3_image'), MinimaxH3ImageModel)
        model = object.__new__(MinimaxH3ImageModel)
        model.model_config = SimpleNamespace(layer_offloading=False, quantize=True, qtype='convrot8',
            quantize_te=True, qtype_te='nvfp4', te_name_or_path=None)
        model._validate_prequantized_configuration()
        for field, value in (('layer_offloading', True), ('quantize', False), ('qtype', 'convrotw4a8'),
                             ('quantize_te', False), ('qtype_te', 'convrot8')):
            original = getattr(model.model_config, field)
            setattr(model.model_config, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                model._validate_prequantized_configuration()
            setattr(model.model_config, field, original)
        config = ModelConfig(name_or_path='Comfy-Org/MiniMax-H3-Image', arch='minimax_h3_image',
            quantize=True, qtype='convrot8', quantize_te=True, qtype_te='nvfp4')
        train = TrainConfig(noise_scheduler='flowmatch')
        save = SaveConfig()
        network = NetworkConfig(type='lora')
        validate_configs(train, config, save, [DatasetConfig(folder_path='unused', num_frames=1)], network)
        for extra in ({'num_frames': 5}, {'do_audio': True}, {'auto_frame_count': True}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_configs(train, config, save, [DatasetConfig(folder_path='unused', **extra)], network)
        for field, value in (('noise_scheduler', 'ddpm'), ('loss_type', 'mean_flow'), ('train_turbo', True), ('phases', [{'loss_type': 'mean_flow'}])):
            original = getattr(train, field)
            setattr(train, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_configs(train, config, save, [], network)
            setattr(train, field, original)
        train.train_text_encoder = True
        with self.assertRaisesRegex(ValueError, 'text-encoder'):
            validate_configs(train, config, save, [], network)


if __name__ == '__main__':
    unittest.main()
