"""Offline CPU regressions for the November H3, audio and sampling ports."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from toolkit.config_modules import GenerateImageConfig, SampleConfig
from toolkit.dto import DTO
from toolkit.models.base_model import BaseModel
from toolkit.sample_sigmas import normalize_sample_sigmas, install_sample_sigma_override
from extensions_built_in.diffusion_models.minimax_h3 import MinimaxH3Model, PH3ImageModel
from extensions_built_in.diffusion_models.minimax_h3.p_h3_image import PH3TextConnector, PH3_SAMPLE_SIGMAS
from extensions_built_in.diffusion_models.minimax_h3.src.vae import PH3ImageVideoVAE, PH3_IMAGE_FRAME_POSITION


class SigmaTests(unittest.TestCase):
    def test_normalization_and_config_inheritance(self):
        self.assertEqual(normalize_sample_sigmas([500, 1000, 0, 500]), [1., .5])
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaisesRegex(ValueError, 'finite'):
                normalize_sample_sigmas([1, value])
        config = GenerateImageConfig(output_path='/tmp/nov4-sample.png', sigmas=[1, .5, 0], num_inference_steps=40)
        self.assertEqual(config.num_inference_steps, 2)
        self.assertEqual(config.sigmas, [1, .5])
        config = SampleConfig(sigmas=[1, .5], samples=[{'prompt': 'one'}, {'prompt': 'two', 'sigmas': None}])
        self.assertEqual(config.samples[0].sigmas, [1, .5])
        self.assertIsNone(config.samples[1].sigmas)

    def test_base_sampling_exception_restores_scheduler_and_training_state(self):
        model = object.__new__(BaseModel)
        model.network = None
        model.model = torch.nn.Linear(1, 1)
        model.model_config = SimpleNamespace(low_vram=False, assistant_lora_path=None, inference_lora_path=None)
        model.device_torch = torch.device('cpu')
        model.torch_dtype = torch.float32
        model.adapter = None
        model.save_device_state = Mock()
        model.set_device_state_preset = Mock()
        model.restore_device_state = Mock()
        unhook = Mock()
        model._install_sample_step_hooks = lambda pipeline: unhook
        scheduler = FlowMatchEulerDiscreteScheduler()
        pipeline = SimpleNamespace(scheduler=scheduler)
        model.pipeline = pipeline
        model.noise_scheduler = scheduler

        class FailingConfig:
            sigmas = [1., .5]
            @property
            def network_multiplier(self):
                scheduler.set_timesteps(2)
                torch.rand(1)
                raise RuntimeError('sample setup failed')

        state = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, 'sample setup failed'):
            model.generate_images([FailingConfig()], pipeline=pipeline)
        self.assertNotIn('set_timesteps', scheduler.__dict__)
        self.assertNotIn('sample_sigmas', model.__dict__)
        torch.testing.assert_close(torch.get_rng_state(), state)
        unhook.assert_called_once()
        model.restore_device_state.assert_called_once()

    def test_native_pipeline_uses_exact_sigma_grid_and_step_count(self):
        from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
        from extensions_built_in.diffusion_models.fibo2.src.pipeline import Fibo2Pipeline
        seen = []
        class Transformer:
            params = SimpleNamespace(in_channels=3)
            def __call__(self, x, timestep, *args):
                seen.append(float(timestep[0]))
                return torch.ones_like(x)
        owner = SimpleNamespace(device_torch=torch.device('cpu'), torch_dtype=torch.float32,
            transformer=Transformer(), sample_step_hook=None, decode_latents=lambda x, **kwargs: x)
        embeds = AdvancedPromptEmbeds(text_embeds=[torch.zeros(1, 2)])
        Fibo2Pipeline(owner)(embeds, embeds, height=32, width=32, num_inference_steps=19,
            sigmas=[1., .75, .25], guidance_scale=1., latents=torch.zeros(1, 3, 2, 2))
        self.assertEqual(seen, [1000., 750., 250.])

    def test_instances_factories_mixed_defaults_and_exception_cleanup(self):
        scheduler = FlowMatchEulerDiscreteScheduler()
        unrelated = FlowMatchEulerDiscreteScheduler()
        original = type(scheduler).set_timesteps
        holder = SimpleNamespace(sample_sigmas=None, get_sample_scheduler=lambda: FlowMatchEulerDiscreteScheduler())
        factory = holder.get_sample_scheduler
        cleanup = install_sample_sigma_override(holder, SimpleNamespace(scheduler=scheduler), [SimpleNamespace(sigmas=[1, .5])])
        try:
            holder.sample_sigmas = [1, .5]
            scheduler.set_timesteps(2)
            torch.testing.assert_close(scheduler.sigmas, torch.tensor([1., .5, 0]))
            unrelated.set_timesteps(4)
            self.assertEqual(len(unrelated.timesteps), 4)
            created = holder.get_sample_scheduler()
            created.set_timesteps(2)
            torch.testing.assert_close(created.sigmas, scheduler.sigmas)
            holder.sample_sigmas = None
            scheduler.set_timesteps(4)
            self.assertEqual(len(scheduler.timesteps), 4)
            raise RuntimeError('sample failed')
        except RuntimeError:
            pass
        finally:
            cleanup()
            cleanup()
        self.assertIs(type(scheduler).set_timesteps, original)
        self.assertNotIn('set_timesteps', scheduler.__dict__)
        self.assertNotIn('set_timesteps', created.__dict__)
        self.assertIs(holder.get_sample_scheduler, factory)
        self.assertIsNone(holder.sample_sigmas)


class AudioStreamTests(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(BaseModel)
        self.video = torch.zeros(2, 3)
        self.silence = torch.zeros(2, 4, 3)
        self.batch = SimpleNamespace(latents=self.video, audio_noise=None)
        self.sigma = torch.tensor([1., .5])

    def test_dataset_reuses_noise_and_dto_carries_differentiable_prediction(self):
        raw = torch.ones_like(self.silence)
        first = self.model.audio_stream_input(self.video, self.batch, raw, self.sigma, self.silence)
        second = self.model.audio_stream_input(self.video, self.batch, raw, self.sigma, self.silence)
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])
        pred = torch.ones_like(raw, requires_grad=True)
        output = self.model.audio_stream_output(self.video, pred, first[1], first[0], self.sigma, False)
        output.audio.sum().backward()
        self.assertIsNotNone(pred.grad)
        self.assertIs(output.audio_noisy, first[0])

    def test_override_fresh_start_silence_and_shape_guard(self):
        override = torch.ones_like(self.silence)
        rows, target, _, stream = self.model.audio_stream_input(DTO(self.video, audio=override), self.batch, override * 8, self.sigma, self.silence)
        torch.testing.assert_close(rows, override)
        self.assertIsNone(target)
        self.assertTrue(stream)
        args = (DTO(self.video, audio=None), self.batch, override * 8, self.sigma, self.silence)
        first = self.model.audio_stream_input(*args)
        second = self.model.audio_stream_input(*args)
        torch.testing.assert_close(first[0], second[0])
        self.assertIsNone(first[1])
        with self.assertRaisesRegex(ValueError, 'shape'):
            self.model.audio_stream_input(DTO(self.video, audio=torch.zeros(1)), self.batch, None, self.sigma, self.silence)
        self.assertIs(self.model.audio_stream_output(self.video, override, None, None, self.sigma, False), self.video)

    def test_h3_silence_reuse_and_stale_field_cleanup_through_prediction(self):
        model = object.__new__(MinimaxH3Model)
        model.device_torch = torch.device('cpu')
        model.torch_dtype = torch.float32
        model.max_text_length = 512
        seen = []

        class Transformer(torch.nn.Module):
            device = torch.device('cpu')
            def forward(self, **kwargs):
                seen.append(kwargs['audio_hidden_states'].clone())
                return torch.zeros_like(kwargs['hidden_states']), torch.zeros_like(kwargs['audio_hidden_states'])

        model.model = Transformer()
        model._build_condition = lambda *args: (None, None, (), ())
        model._pdd_weights = lambda *args: None
        video = torch.zeros(1, 24, 1, 2, 2)
        batch = SimpleNamespace(num_frames=1, dataset_config=SimpleNamespace(do_audio=False), audio_noise=None,
            audio_latents=torch.ones(1, 4, 32), latents=video)
        fields = ('audio_target', 'audio_pred', 'audio_pred_uncond', 'audio_pred_prior',
                  'audio_pred_preservation', 'audio_noisy', 'audio_sigma', 'audio_loss_mask')
        for field in fields:
            setattr(batch, field, torch.ones(1))
        embeds = SimpleNamespace(text_embeds=[torch.zeros(1, 8)], text_token_tags=[torch.zeros(1, dtype=torch.long)])
        for _ in range(2):
            output = model.get_noise_prediction(DTO(video, audio=torch.ones(1, 4, 32)), torch.tensor([500.]), embeds, batch)
            self.assertNotIsInstance(output, DTO)
        torch.testing.assert_close(seen[0], seen[1])
        for field in fields:
            self.assertIsNone(getattr(batch, field))

    def test_ltx_stream_preserves_temporal_conditions_and_primary_slots(self):
        from testing.test_ltx_training_update import LTXTrainingUpdateTests
        fixture = LTXTrainingUpdateTests()
        strategy = {'audio': {'is_generated': True, 'conditions': [{'type': 'prefix', 'tokens': 2, 'probability': 1.0}]}}
        batch, _ = fixture._run_fake_ltx_prediction(ltx_strategy=strategy, do_audio=True,
            audio_latents=torch.tensor([[[10.], [20.], [30.]]]))
        target, primary, mask = batch.audio_target, batch.audio_pred, batch.audio_loss_mask.clone()
        model, transformer = fixture._fake_ltx_model()
        output = model.get_noise_prediction(
            DTO(torch.zeros(1, 1, 3, 1, 1), audio=torch.full((1, 3, 1), 99.)),
            torch.tensor([500.]), SimpleNamespace(text_embeds=torch.zeros(1, 1, 2), attention_mask=torch.ones(1, 1)), batch)
        self.assertEqual(transformer.last_call['audio_hidden_states'].flatten().tolist(), [10., 20., 99.])
        self.assertIsNone(output.audio_target)
        self.assertIs(batch.audio_target, target)
        self.assertIs(batch.audio_pred, primary)
        torch.testing.assert_close(batch.audio_loss_mask, mask)


class PH3Tests(unittest.TestCase):
    def test_validation_does_not_relax_video_h3(self):
        model = object.__new__(PH3ImageModel)
        model.model_config = SimpleNamespace(layer_offloading=False, quantize=True, qtype='convrot8', quantize_te=True, qtype_te='convrot8', te_name_or_path=None)
        model._validate_prequantized_configuration()
        h3 = object.__new__(MinimaxH3Model)
        h3.model_config = model.model_config
        with self.assertRaisesRegex(ValueError, 'NVFP4'):
            h3._validate_prequantized_configuration()
        model.model_config.quantize = model.model_config.quantize_te = False
        model._validate_prequantized_configuration()
        self.assertFalse(model.requires_packed_transformer)
        model.model_config.layer_offloading = True
        with self.assertRaisesRegex(ValueError, 'layer_offloading'):
            model._validate_prequantized_configuration()
        torch.testing.assert_close(model.build_sample_sigmas(8), torch.tensor(PH3_SAMPLE_SIGMAS))
        torch.testing.assert_close(model.audio_sigma(torch.tensor([.4])), torch.tensor([.4]))

    def test_bf16_transformer_loads_strictly_without_relaxing_packed_h3(self):
        from extensions_built_in.diffusion_models.minimax_h3.src.transformer import MiniMaxH3Transformer, MiniMaxH3TransformerParams
        params = MiniMaxH3TransformerParams(hidden_size=8, num_layers=1, token_refiner_num_layers=0,
            num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=16, latents_dim=2,
            audio_latents_dim=2, text_dim=8, timestep_input_dim=4, time_embed_hidden_size=8,
            time_embed_dim=4, rope_inv_freq_len=1)
        source = MiniMaxH3Transformer(params).to(torch.bfloat16)
        model = object.__new__(PH3ImageModel)
        model.model_config = SimpleNamespace(quantize=False)
        model.torch_dtype = torch.bfloat16
        model._resolve_comfy_file = lambda component: 'unused.safetensors'
        model._transformer_params_for_state = lambda state: params
        model.print_and_status_update = lambda message: None
        with patch('extensions_built_in.diffusion_models.minimax_h3.minimax_h3.load_file', return_value=source.state_dict()):
            restored = model._load_transformer()
            for key, tensor in source.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], tensor)
            model.model_config.quantize = True
            with self.assertRaisesRegex(ValueError, 'prequantized ConvRot8'):
                model._load_transformer()

    def test_connector_and_image_vae_layout(self):
        connector = PH3TextConnector(6, 4)
        hidden = torch.randn(2, 2, 3, 3)
        result = connector(hidden)
        self.assertEqual(result.shape, (2, 3, 4))
        torch.testing.assert_close(result[:, 0], connector.sink_embedding.expand(2, -1))
        result.sum().backward()
        self.assertIsNotNone(connector.proj.weight.grad)
        with torch.device('meta'):
            vae = PH3ImageVideoVAE(decoder_num_layers=0)
        self.assertEqual(vae.decoder.patch_size_t, 1)
        self.assertEqual(vae.decoder.fixed_time_coord, PH3_IMAGE_FRAME_POSITION)
        self.assertFalse(hasattr(vae.decoder, 'mask_token'))
        with self.assertRaisesRegex(ValueError, 'single-frame'):
            vae.decode(torch.empty(1, 24, 2, 1, 1))

    def test_sampling_uses_coordinator_phases(self):
        model = object.__new__(PH3ImageModel)
        coordinator = Mock()
        model._sample_memory_coordinator = coordinator
        self.assertEqual(model.sample_memory_generate_components(), ('vae',))
        model._park_vae_for_denoise()
        model._stage_for_sampling()
        coordinator.activate.assert_called_with(('unet', 'adapter'), phase_name='denoise')
        model._stage_for_decode()
        coordinator.activate.assert_called_with(('vae',), phase_name='decode')


if __name__ == '__main__':
    unittest.main()
