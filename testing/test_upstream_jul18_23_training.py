from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from PIL import Image
import torch

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.config_modules import (
    TrainConfig,
    ValidationConfig,
    ValidationItem,
)
from toolkit.exceptions import JobStopRequested
from toolkit.models.diffusion_feature_extraction import (
    DiffusionFeatureExtractor4,
    DiffusionFeatureExtractor6,
)
from toolkit.prompt_utils import PromptEmbeds


class _TrackingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.to_calls = []

    @property
    def device(self):
        return self.weight.device

    def to(self, *args, **kwargs):
        if args:
            self.to_calls.append(torch.device(args[0]))
        return super().to(*args, **kwargs)


class _SetupValidationSD:
    def __init__(self):
        self.text_encoder = _TrackingModule()
        self.vae = _TrackingModule()
        self.encoded_prompts = []
        self.encoded_shapes = []

    def get_bucket_divisibility(self):
        return 8

    def inject_trigger_into_prompt(
        self, prompt, trigger, add_if_not_present=False
    ):
        del add_if_not_present
        return prompt.replace('[trigger]', trigger)

    def text_encoder_to(self, device):
        self.text_encoder.to(device)

    def encode_prompt(self, prompt):
        self.encoded_prompts.append(prompt[0])
        return PromptEmbeds(torch.ones(1, 2, 3))

    def encode_images(self, image_list, device=None, dtype=None):
        image = image_list[0]
        self.encoded_shapes.append(tuple(image.shape))
        return torch.randn(
            1,
            4,
            image.shape[-2] // 8,
            image.shape[-1] // 8,
            device=device,
            dtype=dtype,
        )


class _ValidationNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.multiplier = 0.35
        self.is_active = True

    def __enter__(self):
        self.is_active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        self.is_active = False


class _ValidationSD:
    def __init__(self, fail=False):
        self.model = torch.nn.Linear(1, 1)
        self.model.train()
        self.is_flow_matching = True
        self.prediction_type = 'epsilon'
        self.fail = fail
        self.targets = []

    def get_model_to_train(self):
        return self.model

    def add_noise(self, latents, noise, timesteps):
        del timesteps
        self.targets.append(noise - latents)
        return latents + noise

    def predict_noise(self, **kwargs):
        del kwargs
        if self.fail:
            raise RuntimeError("validation failed")
        return self.targets[-1]


class _ProgressBar:
    def __init__(self):
        self.disable = False
        self.closed = False

    def close(self):
        self.closed = True


class _RecordingVAE:
    def __init__(self):
        self.device = torch.device('cpu')
        self.dtype = torch.bfloat16
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)
        self.decoded_latents = None

    def decode(self, latents):
        self.decoded_latents = latents.detach().clone()
        return latents


class UpstreamJuly1823TrainingTest(unittest.TestCase):
    def test_validation_config_defaults_and_nested_items(self):
        config = TrainConfig(
            validation_config={
                'validation_items': [
                    {'image_path': 'one.png', 'prompt': 'a [trigger]'}
                ]
            }
        )

        self.assertIsInstance(config.validation_config, ValidationConfig)
        self.assertIsInstance(
            config.validation_config.validation_items[0], ValidationItem
        )
        self.assertEqual(config.validation_config.resolution, 512)
        self.assertEqual(config.validation_config.validate_every_n_steps, 10)
        self.assertEqual(
            config.validation_config.validation_sigmas,
            [1.0, 0.75, 0.5, 0.25],
        )
        self.assertIsNone(TrainConfig().validation_config)

    def test_validation_cache_is_bucketed_triggered_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            landscape_path = Path(temp_dir) / 'landscape.png'
            portrait_path = Path(temp_dir) / 'portrait.png'
            Image.new('RGB', (640, 320), 'red').save(landscape_path)
            Image.new('RGB', (240, 480), 'blue').save(portrait_path)

            process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
            process.train_config = TrainConfig(
                dtype='fp32',
                validation_config={
                    'validation_items': [
                        {
                            'image_path': str(landscape_path),
                            'prompt': 'a [trigger] landscape',
                        },
                        {
                            'image_path': str(portrait_path),
                            'prompt': 'a [trigger] portrait',
                        },
                    ],
                    'resolution': 256,
                    'validation_sigmas': [0.5],
                },
            )
            process.accelerator = SimpleNamespace(is_main_process=True)
            process.device_torch = torch.device('cpu')
            process.trigger_word = 'subject'
            process.sd = _SetupValidationSD()
            process._validation_cache = None

            rng_state = torch.random.get_rng_state().clone()
            process.setup_validation()
            first_latents = [
                latent.clone()
                for latent in process._validation_cache['latents']
            ]
            first_noise = [
                noise.clone()
                for noise in process._validation_cache['noise']
            ]
            self.assertTrue(torch.equal(rng_state, torch.random.get_rng_state()))

            process.setup_validation()

            self.assertEqual(
                process.sd.encoded_prompts[:2],
                ['a subject landscape', 'a subject portrait'],
            )
            self.assertNotEqual(
                process.sd.encoded_shapes[0],
                process.sd.encoded_shapes[1],
            )
            for before, after in zip(
                first_latents, process._validation_cache['latents']
            ):
                self.assertTrue(torch.equal(before, after))
            for before, after in zip(
                first_noise, process._validation_cache['noise']
            ):
                self.assertTrue(torch.equal(before, after))

    def test_validation_cache_is_built_on_non_main_rank(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / 'validation.png'
            Image.new('RGB', (320, 160), 'green').save(image_path)

            process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
            process.train_config = TrainConfig(
                dtype='fp32',
                validation_config={
                    'validation_items': [
                        {
                            'image_path': str(image_path),
                            'prompt': 'validation prompt',
                        },
                    ],
                    'resolution': 128,
                    'validation_sigmas': [0.5],
                },
            )
            process.accelerator = SimpleNamespace(is_main_process=False)
            process.device_torch = torch.device('cpu')
            process.trigger_word = None
            process.sd = _SetupValidationSD()
            process._validation_cache = None

            process.setup_validation()

            self.assertIsNotNone(process._validation_cache)
            self.assertEqual(len(process._validation_cache['latents']), 1)
            self.assertEqual(
                process.sd.encoded_prompts,
                ['validation prompt'],
            )

    def _make_validation_process(self, fail=False):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.train_config = TrainConfig(
            dtype='fp32',
            cfg_scale=1.0,
            bypass_guidance_embedding=False,
            validation_config={
                'validation_items': [],
                'validation_sigmas': [0.75, 0.25],
            },
        )
        process.accelerator = SimpleNamespace(is_main_process=True)
        process.device_torch = torch.device('cpu')
        process.sd = _ValidationSD(fail=fail)
        process.network = _ValidationNetwork()
        process.additional_logs = OrderedDict()
        process._validation_cache = {
            'latents': [
                torch.zeros(1, 4, 8, 16),
                torch.zeros(1, 4, 16, 8),
            ],
            'noise': [
                torch.ones(1, 4, 8, 16),
                torch.ones(1, 4, 16, 8),
            ],
            'embeds': [
                PromptEmbeds(torch.zeros(1, 2, 3)),
                PromptEmbeds(torch.zeros(1, 2, 3)),
            ],
        }
        return process

    def test_validation_logs_loss_and_restores_training_state(self):
        process = self._make_validation_process()

        process.validate()

        self.assertEqual(process.additional_logs['val/loss'], 0.0)
        self.assertTrue(process.sd.model.training)
        self.assertEqual(process.network.multiplier, 0.35)
        self.assertTrue(process.network.is_active)
        self.assertTrue(process.network.training)
        self.assertEqual(len(process.sd.targets), 2)

    def test_validation_restores_state_when_prediction_fails(self):
        process = self._make_validation_process(fail=True)

        with self.assertRaisesRegex(RuntimeError, "validation failed"):
            process.validate()

        self.assertTrue(process.sd.model.training)
        self.assertEqual(process.network.multiplier, 0.35)
        self.assertTrue(process.network.is_active)
        self.assertTrue(process.network.training)
        self.assertNotIn('val/loss', process.additional_logs)

    def test_non_main_rank_executes_validation_without_external_log_write(self):
        process = self._make_validation_process()
        process.accelerator.is_main_process = False
        process.logger = SimpleNamespace(log=mock.Mock())

        process.validate()
        periodic_logs = process._publish_validation_logs(
            process.additional_logs
        )

        self.assertEqual(len(process.sd.targets), 2)
        self.assertEqual(process.additional_logs['val/loss'], 0.0)
        process.logger.log.assert_not_called()
        self.assertEqual(periodic_logs, OrderedDict())
        self.assertTrue(process.sd.model.training)
        self.assertEqual(process.network.multiplier, 0.35)
        self.assertTrue(process.network.is_active)
        self.assertTrue(process.network.training)

    def test_validation_cadence_always_includes_start_step(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.train_config = TrainConfig(
            validation_config={
                'validate_every_n_steps': 10,
                'validation_items': [],
            }
        )
        process._validation_cache = {'latents': []}
        process.start_step = 7

        process.step_num = 7
        self.assertTrue(process._should_validate_step())
        process.step_num = 10
        self.assertTrue(process._should_validate_step())
        process.step_num = 11
        self.assertFalse(process._should_validate_step())

    def test_validation_metric_publishes_at_start_outside_log_cadence(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.accelerator = SimpleNamespace(is_main_process=True)
        process.logger = SimpleNamespace(log=mock.Mock())
        process.start_step = 7
        process.step_num = 7
        process.logging_config = SimpleNamespace(log_every=100)
        additional_logs = OrderedDict(
            [
                ('val/loss', 0.25),
                ('monitor/custom', 1.5),
            ]
        )

        periodic_logs = process._publish_validation_logs(additional_logs)

        process.logger.log.assert_called_once_with(
            OrderedDict([('val/loss', 0.25)])
        )
        self.assertEqual(
            periodic_logs,
            OrderedDict([('monitor/custom', 1.5)]),
        )

    def test_cached_sample_prompts_replace_trigger_placeholder(self):
        process = SDTrainer.__new__(SDTrainer)
        process.train_config = SimpleNamespace(disable_sampling=False)
        process.sample_config = SimpleNamespace(
            prompts=['a [trigger] portrait'],
            samples=[
                SimpleNamespace(
                    neg='bad',
                    ctrl_img=None,
                    ctrl_img_1=None,
                    ctrl_img_2=None,
                    ctrl_img_3=None,
                )
            ],
        )
        process.save_root = '.'
        process.trigger_word = 'subject'
        encoded_prompts = []
        process.sd = SimpleNamespace(
            sample_prompts_cache=[],
            encode_control_in_text_embeddings=False,
            inject_trigger_into_prompt=lambda prompt, trigger, **kwargs: (
                prompt.replace('[trigger]', trigger)
            ),
            encode_prompt=lambda prompt: (
                encoded_prompts.append(prompt)
                or PromptEmbeds(torch.zeros(1, 2, 3))
            ),
        )

        process.cache_sample_prompts()

        self.assertEqual(encoded_prompts[0], 'a subject portrait')

    def test_ui_stop_silences_progress_without_overwriting_queue_status(self):
        process = DiffusionTrainer.__new__(DiffusionTrainer)
        process.is_ui_trainer = True
        process.is_stopping = True
        process.accelerator = SimpleNamespace(is_main_process=True)
        process.progress_bar = _ProgressBar()
        process.last_save_step = 12
        process.update_status = mock.Mock()
        process.update_db_key = mock.Mock()

        async def wait_for_all_async():
            return None

        process.wait_for_all_async = wait_for_all_async
        process.thread_pool = SimpleNamespace(shutdown=mock.Mock())
        process.ui_job_store = SimpleNamespace(close=mock.Mock())

        process.on_error(
            JobStopRequested("Job returning to queue", return_to_queue=True)
        )

        self.assertTrue(process.progress_bar.disable)
        self.assertTrue(process.progress_bar.closed)
        process.update_status.assert_not_called()

    def test_dfe4_uses_direct_x0_and_resets_each_logged_loss(self):
        extractor = DiffusionFeatureExtractor4.__new__(
            DiffusionFeatureExtractor4
        )
        torch.nn.Module.__init__(extractor)
        extractor.vae = _RecordingVAE()
        extractor.losses = {'older_loss': 2.0}
        extractor.log_every = 1
        extractor.step = 1

        extractor(
            noise=torch.zeros(1, 4, 2, 2),
            noise_pred=torch.ones(1, 4, 2, 2),
            noisy_latents=torch.full((1, 4, 2, 2), 10.0),
            timesteps=torch.tensor([500]),
            batch=SimpleNamespace(tensor=torch.zeros(1, 4, 2, 2)),
            scheduler=SimpleNamespace(),
            clip_weight=0.0,
            mse_weight=1.0,
        )

        self.assertTrue(
            torch.equal(
                extractor.vae.decoded_latents,
                torch.full((1, 4, 2, 2), 9.5, dtype=torch.bfloat16),
            )
        )
        self.assertEqual(extractor.losses['older_loss'], 0.0)
        self.assertEqual(extractor.losses['mse_loss'], 0.0)

    def test_dfe6_clamps_vae_overshoot_before_preprocessing(self):
        extractor = DiffusionFeatureExtractor6.__new__(
            DiffusionFeatureExtractor6
        )
        torch.nn.Module.__init__(extractor)
        extractor.processor = SimpleNamespace(
            do_rescale=False,
            do_normalize=False,
        )
        input_tensor = torch.zeros(1, 3, 16, 16)
        input_tensor[:, :, 0, 0] = -0.1
        input_tensor[:, :, 0, 1] = 1.1

        output = extractor.prepare_inputs(input_tensor)['pixel_values']

        self.assertEqual(float(output.min()), 0.0)
        self.assertEqual(float(output.max()), 1.0)


if __name__ == '__main__':
    unittest.main()
