"""CPU coverage of the real trainer's Ming cache, sampling and flow-loss paths."""
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from jobs.process.BaseSDTrainProcess import prepare_encrypted_dataset_cache_policy
from testing.test_ming_components import tiny_model
from testing.test_ming_integration import adapter, network
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import DatasetConfig, SampleConfig, TrainConfig


class MingTrainerTests(unittest.TestCase):
    def test_real_trainer_preserves_group_axes_and_uses_flow_target(self):
        holder = adapter(True)
        holder.model = tiny_model().requires_grad_(False)
        lora = network(holder)
        holder.network = lora
        holder.noise_scheduler = holder.get_train_scheduler()
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.sd = holder
        trainer.train_config = TrainConfig(dtype='float32', noise_scheduler='flowmatch',
            timestep_type='linear', min_denoising_steps=400, max_denoising_steps=400)
        trainer.model_config = holder.model_config
        trainer.device_torch = torch.device('cpu')
        trainer.timer = lambda _: nullcontext()
        trainer.embedding = trainer.adapter = trainer.trigger_word = trainer.dfe = None
        trainer._record_tensor_stats = lambda *args: None
        trainer._record_monitor_metric = lambda *args: None
        trainer.additional_logs = OrderedDict()
        clean = torch.randn(1, 4, 3, 4, 4)
        batch = SimpleNamespace(latents=clean, tensor=None, file_items=[SimpleNamespace()],
            unconditional_tensor=None, unconditional_latents=None, mask_tensor=None,
            unaugmented_tensor=None, control_tensor=torch.rand(1, 4, 32, 32),
            loss_multiplier_list=[1.0], audio_pred=None, audio_target=None,
            get_caption_list=lambda: ['Decompose the image into 2 layers.'],
            get_is_reg_list=lambda: [False])
        noisy, noise, times, captions, images = trainer.process_general_training_batch(batch)
        self.assertEqual(tuple(noisy.shape), (1, 4, 3, 4, 4))
        self.assertIsNone(images)
        self.assertEqual(captions, batch.get_caption_list())
        sigma = (times / 1000).reshape(-1, 1, 1, 1, 1)
        torch.testing.assert_close(noisy, clean * (1 - sigma) + noise * sigma)
        embeds = AdvancedPromptEmbeds(text_embeds=[torch.randn(4, 8)],
            direct_embeds=[torch.randn(3, 16)], reference_latents=[torch.randn(4, 4, 4)])
        with lora:
            prediction = holder.predict_noise(latents=noisy, text_embeddings=embeds,
                timestep=times, guidance_scale=1.0, batch=batch)
            loss = trainer.calculate_loss(prediction, noise, noisy, times, batch,
                mask_multiplier=torch.ones(1, 1, 1, 1))
            torch.testing.assert_close(loss, (prediction.float() - (noise-clean).float()).square().mean())
            loss.backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in lora.parameters()))

    def test_first_sample_cache_keeps_own_prompt_reference_geometry_and_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'reference.png'
            Image.new('RGBA', (16, 16), (255, 0, 0, 128)).save(reference)
            holder = adapter(True)
            recorded = []

            def condition(caption, reference=None):
                recorded.append((caption, reference.size))
                return torch.ones(2, 8), torch.ones(1, 16)

            holder._conditioner = SimpleNamespace(encode=condition)
            holder.encode_images = lambda images: torch.zeros(len(images), 16,
                images[0].shape[-2] // 8, images[0].shape[-1] // 8)
            trainer = SDTrainer.__new__(SDTrainer)
            trainer.sd = holder
            trainer.train_config = TrainConfig()
            trainer.sample_config = SampleConfig(format='png', width=64, height=64, num_layers=2,
                samples=[{'prompt': 'regular', 'ctrl_img': str(reference)}])
            trainer.first_sample_config = SampleConfig(format='png', width=128, height=64, num_layers=3,
                samples=[{'prompt': 'first', 'ctrl_img': str(reference)},
                         {'prompt': 'extra', 'ctrl_img': str(reference), 'num_layers': 4}])
            trainer.trigger_word = trainer.embedding = trainer.adapter = trainer.adapter_config = trainer.ema = None
            trainer.logger = None
            trainer.save_root = directory
            trainer.accelerator = SimpleNamespace(is_main_process=True)
            trainer.post_process_generate_image_config_list = lambda values: values
            trainer.cache_all_sample_prompts()
            self.assertEqual(recorded, [('Decompose the image into 2 layers.\nregular', (64, 64)),
                ('Decompose the image into 3 layers.\nfirst', (128, 64)),
                ('Decompose the image into 4 layers.\nextra', (128, 64))])
            regular_cache = holder.sample_prompts_cache
            captured = []

            def generate(configs, **kwargs):
                captured.append((configs, holder.sample_prompts_cache))

            trainer._generate_sample_images = generate
            with patch('jobs.process.BaseSDTrainProcess.flush'):
                trainer.sample(0, is_first=True)
                trainer.sample(1)
            first_configs, first_cache = captured[0]
            self.assertEqual([item.num_layers for item in first_configs], [3, 4])
            self.assertEqual(len(first_cache), 2)
            self.assertEqual(tuple(first_cache[0]['conditional'].reference_latents[0].shape), (16, 8, 16))
            self.assertIs(captured[1][1], regular_cache)
            self.assertIs(holder.sample_prompts_cache, regular_cache)
            trainer._generate_sample_images = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('cancelled'))
            with patch('jobs.process.BaseSDTrainProcess.flush'), self.assertRaisesRegex(RuntimeError, 'cancelled'):
                trainer.sample(0, is_first=True)
            self.assertIs(holder.sample_prompts_cache, regular_cache)

    def test_encrypted_policy_prevents_plaintext_cache_and_encoder_unload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encrypted = root / 'encrypted'
            encrypted.mkdir()
            marker = encrypted / '.aitk_encrypted_dataset.json'
            marker.write_text('{}')
            train = TrainConfig(cache_text_embeddings=True, unload_text_encoder=True)
            private = DatasetConfig(folder_path=str(encrypted), cache_text_embeddings=True,
                cache_latents=True, cache_latents_to_disk=True, cache_clip_vision_to_disk=True)
            plain = DatasetConfig(folder_path=str(root / 'plain'), cache_text_embeddings=True,
                cache_latents_to_disk=True)
            prepare_encrypted_dataset_cache_policy(train, [private, plain])
            self.assertTrue(private.encrypted)
            for setting in ('cache_text_embeddings', 'cache_latents', 'cache_latents_to_disk', 'cache_clip_vision_to_disk'):
                self.assertFalse(getattr(private, setting))
            self.assertFalse(plain.cache_text_embeddings)
            self.assertTrue(plain.cache_latents_to_disk)
            self.assertFalse(train.cache_text_embeddings)
            self.assertFalse(train.unload_text_encoder)
            self.assertEqual([path for path in root.rglob('*') if path.is_file()], [marker])
            grouped = DatasetConfig(type='layered_image', folder_path=str(encrypted))
            with self.assertRaisesRegex(ValueError, 'Grouped layered-image'):
                prepare_encrypted_dataset_cache_policy(train, [grouped])

    def test_design_validation_preserves_alpha_and_runs_after_encoder_unload(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'validation.png'
            Image.new('RGBA', (32, 32), (255, 0, 0, 64)).save(image)
            holder = adapter()
            holder.model = tiny_model().requires_grad_(False)
            holder.noise_scheduler = holder.get_train_scheduler()
            holder.vae = torch.nn.Linear(1, 1)
            holder.text_encoder = []
            pixels = []

            def encode_images(images, **kwargs):
                pixels.append(images[0].clone())
                return torch.nn.functional.interpolate(images[0].unsqueeze(0), size=(4, 4))

            holder.encode_images = encode_images
            holder.encode_prompt = lambda prompts: AdvancedPromptEmbeds(
                text_embeds=[torch.zeros(2, 8)], direct_embeds=[torch.zeros(1, 16)],
                reference_latents=[torch.empty(0)])
            trainer = SDTrainer.__new__(SDTrainer)
            trainer.sd = holder
            trainer.device_torch = torch.device('cpu')
            trainer.trigger_word = trainer.network = None
            trainer.additional_logs = OrderedDict()
            trainer.train_config = TrainConfig(dtype='float32', validation_config={
                'validation_items': [{'image_path': str(image), 'prompt': 'transparent red'}],
                'resolution': 32, 'validation_sigmas': [0.5, 0.25]})
            with patch('jobs.process.BaseSDTrainProcess.flush'):
                trainer.setup_validation()
            self.assertEqual(pixels[0].shape[0], 4)
            torch.testing.assert_close(pixels[0][3], torch.full_like(pixels[0][3], 64 / 127.5 - 1))
            holder.encode_prompt = holder.encode_images = lambda *args, **kwargs: self.fail('validation recached an unloaded component')
            trainer.validate()
            self.assertTrue(torch.isfinite(torch.tensor(trainer.additional_logs['val/loss'])))


if __name__ == '__main__':
    unittest.main()
