"""Runtime contracts for training process hooks and watermark metadata."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest

from accelerate import Accelerator
import torch

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from testing.test_ming_integration import adapter
from toolkit.config_modules import TrainConfig
from toolkit.watermarking.authenlora import AuthenLoRAController


class TrainingProcessContractTests(unittest.TestCase):
    def test_base_training_hook_requires_named_loss_implementation(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        with self.assertRaisesRegex(NotImplementedError, 'loss dictionary'):
            process.hook_train_loop(None)

    def test_prepared_and_wrapped_watermark_metadata_remains_public(self):
        controller = AuthenLoRAController.__new__(AuthenLoRAController)
        torch.nn.Module.__init__(controller)
        controller.config = SimpleNamespace(msg_bits=4, mapper_rank=2, bake_on_save=False)
        controller.codec_hash = 'codec-digest'
        controller.current_secret = 'private watermark secret'
        preparation = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        preparation.authenlora = controller
        preparation.accelerator = Accelerator()
        preparation.sd = SimpleNamespace(vae=None, unet=None, text_encoder=None,
                                         refiner_unet=None, network=None)
        preparation.adapter = preparation.lr_scheduler = None
        preparation.modules_being_trained = []
        preparation.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        preparation.prepare_accelerator()
        prepared = preparation.authenlora
        self.assertIs(prepared, controller)
        self.assertEqual(preparation.modules_being_trained, [prepared])
        expected = controller.get_public_metadata()
        wrapper = torch.nn.Module()
        wrapper.module = prepared
        for module in (prepared, wrapper):
            with self.subTest(wrapped=module is wrapper):
                process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
                process.authenlora = module
                process.get_training_info = lambda: {'step': 1}
                process.sd = SimpleNamespace(get_base_model_version=lambda: 'test')
                process.model_config = SimpleNamespace(model_kwargs={})
                process.job = SimpleNamespace(name='test')
                process.trigger_word = None
                metadata = {}
                process.add_meta = metadata.update
                process.update_training_metadata()
                self.assertEqual(metadata['aitk_watermark'], expected)
                self.assertNotIn(controller.current_secret, str(metadata))

    def test_general_batch_records_optional_metrics_without_requiring_callbacks(self):
        for enabled in (False, True):
            with self.subTest(callbacks=enabled):
                holder = adapter(True)
                holder.model = torch.nn.Linear(1, 1)
                holder.noise_scheduler = holder.get_train_scheduler()
                process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
                process.sd = holder
                process.train_config = TrainConfig(dtype='float32', noise_scheduler='flowmatch',
                    timestep_type='linear', min_denoising_steps=400, max_denoising_steps=400)
                process.model_config = holder.model_config
                process.device_torch = torch.device('cpu')
                process.timer = lambda _: nullcontext()
                process.embedding = process.adapter = process.trigger_word = None
                metrics, stats = {}, {}
                if enabled:
                    process._record_monitor_metric = lambda key, value: metrics.update({key: float(value)})
                    process._record_tensor_stats = lambda key, value: stats.update({key: value.clone()})
                clean = torch.ones(1, 4, 3, 4, 4)
                batch = SimpleNamespace(latents=clean, tensor=None, file_items=[SimpleNamespace()],
                    unconditional_tensor=None, unconditional_latents=None, mask_tensor=None,
                    unaugmented_tensor=None, control_tensor=None,
                    get_caption_list=lambda: ['caption'], get_is_reg_list=lambda: [False])
                noisy, noise, times, captions, images = process.process_general_training_batch(batch)
                sigma = (times / 1000).reshape(-1, 1, 1, 1, 1)
                torch.testing.assert_close(noisy, clean * (1 - sigma) + noise * sigma)
                self.assertEqual(captions, ['caption'])
                self.assertIsNone(images)
                if enabled:
                    self.assertEqual(metrics['train/batch_size'], 1)
                    self.assertEqual(metrics['train/timestep_mean'], times.item())
                    torch.testing.assert_close(stats['train/latents'], clean)
                    torch.testing.assert_close(stats['train/noise'], noise)
                    torch.testing.assert_close(stats['train/noisy_latents'], noisy)


if __name__ == '__main__':
    unittest.main()
