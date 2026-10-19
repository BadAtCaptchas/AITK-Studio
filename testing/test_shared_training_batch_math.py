"""Weight-free regressions for trainer batch targets, reductions and AMP."""
import os
import subprocess
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from diffusers import DDPMScheduler

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from testing.test_dataloader_batch_dto import make_file_item
from toolkit.config_modules import TrainConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.dto import DTO
from toolkit.prompt_utils import AdvancedPromptEmbeds, PromptEmbeds


def make_batch(size=2):
    rows = []
    for i in range(size):
        row = make_file_item()
        row.is_reg = row.prior_reg = False
        row.network_weight = 0.5 + i
        row.caption = f'caption {i}'
        row.caption_short = f'short {i}'
        row.loss_multiplier = 1.0 + i
        rows.append(row)
    batch = DataLoaderBatchDTO(file_items=rows)
    values = torch.arange(size, dtype=torch.float32)
    batch.latents = DTO(values[:, None, None, None].expand(-1, 4, 2, 2) * 10,
                        audio=values[:, None, None].expand(-1, 3, 2),
                        nested={'video_mask': values[:, None]}, sample_rate=16000)
    batch.test_noise = (values + 2)[:, None, None, None].expand(-1, 4, 2, 2)
    batch.mask_tensor = torch.ones(size, 1, 2, 2)
    if size > 1:
        batch.mask_tensor[1, :, :, 0] = 0
    batch.extra_values = values[:, None]
    batch.prompt_embeds = PromptEmbeds(values[:, None, None].expand(-1, 3, 2),
                                       attention_mask=torch.ones(size, 3))
    batch.control_tensor_list = [[torch.full((2, 2, 2), float(i))] for i in range(size)]
    batch.audio_data = [{'waveform': torch.full((size, 8), float(i)), 'sample_rate': 16000} for i in range(size)]
    batch.audio_loss_mask = torch.ones(size, 3)
    batch.video_loss_mask = values[:, None]
    return batch


class Timer:
    def __call__(self, *_):
        return nullcontext()

    def start(self, *_):
        pass

    def stop(self, *_):
        pass


class Network:
    multiplier = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def make_loss_trainer(prediction='epsilon', flow=False, **config):
    trainer = SDTrainer.__new__(SDTrainer)
    trainer.train_config = TrainConfig(dtype='fp32', timestep_type='linear', **config)
    trainer.device_torch = torch.device('cpu')
    trainer.sd = SimpleNamespace(prediction_type=prediction, is_flow_matching=flow, x0_pred=False,
                                 noise_scheduler=DDPMScheduler(prediction_type=prediction))
    trainer.dfe = trainer.adapter = None
    trainer.additional_logs = {}
    trainer._record_tensor_stats = trainer._record_monitor_metric = lambda *a, **k: None
    return trainer


def calculate(trainer, prediction, batch, prior=None):
    noise = batch.test_noise
    return trainer.calculate_loss(prediction, noise, noise,
                                  torch.full((noise.shape[0],), 500), batch,
                                  mask_multiplier=torch.ones(noise.shape[0], 1, 2, 2),
                                  prior_pred=prior)


def make_forward_trainer(single, *, cached=True, doubled=False, **config):
    trainer = make_loss_trainer(flow=True, single_item_batching=single,
                                unload_text_encoder=cached, do_cfg=True, **config)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    trainer.sd.vae = torch.nn.Linear(1, 1)
    trainer.sd.vae.dtype = torch.float32
    trainer.sd.text_encoder = torch.nn.Linear(1, 1)
    trainer.sd.text_encoder.dtype = torch.float32
    trainer.sd.vae_torch_dtype = trainer.sd.te_torch_dtype = torch.float32
    trainer.sd.do_masked_loss = True
    trainer.sd.is_xl = False
    trainer.sd.condition_noisy_latents = lambda latents, batch: latents
    trainer.model_config = SimpleNamespace(refiner_name_or_path=None)
    trainer.network = Network()
    trainer.timer = Timer()
    trainer.cuda_memory_phase = lambda *_: nullcontext()
    trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
    trainer.embedding = trainer.adapter_config = trainer.assistant_adapter = trainer.decorator = None
    trainer._clip_image_embeds_unconditional = trainer.negative_prompt_pool = None
    trainer.batch_negative_prompt = None
    trainer.is_caching_text_embeddings = False
    trainer.do_prior_prediction = trainer.do_guided_loss = trainer.do_long_prompts = False
    trainer.cached_trigger_embeds = None
    trainer.cached_blank_embeds = PromptEmbeds(torch.zeros(1, 3, 2))
    trainer._should_record_monitor_metrics = lambda: False
    trainer.get_batch_prompt_kwargs = lambda batch: {'source_ids': [row.caption for row in batch.file_items]}
    trainer.encoded = []
    trainer.seen = []

    def encode(prompts, *args, **kwargs):
        trainer.encoded.append((list(prompts), kwargs['source_ids']))
        return PromptEmbeds(torch.zeros(len(prompts), 3, 2))

    trainer.sd.encode_prompt = encode

    def process(batch):
        noise = batch.test_noise
        noisy = (batch.latents.tensor + noise) * 0.5
        timesteps = torch.full((noise.shape[0],), 500)
        prompts = batch.get_caption_list()
        imgs = batch.tensor
        if doubled:
            noisy, noise, timesteps = [torch.cat([value, value]) for value in (noisy, noise, timesteps)]
            prompts = prompts + batch.get_caption_short_list()
            batch.mask_tensor = torch.cat([batch.mask_tensor, batch.mask_tensor])
            imgs = torch.cat([imgs, imgs])
        return noisy, noise, timesteps, prompts, imgs

    trainer.process_general_training_batch = process

    def predict(*, noisy_latents, conditional_embeds, unconditional_embeds, batch, **kwargs):
        size = noisy_latents.shape[0]
        assert len(batch.file_items) == batch.latents.shape[0] == size
        assert conditional_embeds.text_embeds.shape[0] == unconditional_embeds.text_embeds.shape[0] == size
        assert batch.audio_latents.shape[0] == batch.audio_loss_mask.shape[0] == size
        assert batch.extra_values.shape[0] == len(batch.loss_multiplier_list) == size
        assert len(trainer.network.multiplier) == size
        trainer.seen.append(([row.caption for row in batch.file_items], list(trainer.network.multiplier)))
        scale = torch.tensor(trainer.network.multiplier)[:, None, None, None]
        batch.audio_target = batch.audio_latents
        batch.audio_pred = batch.audio_target + parameter * 0.25
        return batch.test_noise - batch.latents.tensor + parameter * scale

    trainer.predict_noise = predict
    return trainer, parameter


class BatchSelectionTests(unittest.TestCase):
    def test_selects_cached_streams_and_repeated_rows_without_mutating_source(self):
        batch = make_batch()
        batch.mask_tensor = torch.arange(4.)[:, None, None, None]
        batch._tensor = None
        batch._tensor_uint8 = torch.zeros(2, 3, 2, 2, dtype=torch.uint8)
        selected = batch.select_item(3, 4)
        self.assertIs(selected.file_items[0], batch.file_items[1])
        self.assertIsNone(batch._tensor)
        self.assertEqual(selected._tensor_uint8.shape[0], 1)
        self.assertEqual(selected.mask_tensor.item(), 3)
        self.assertEqual(selected.latents.shape[0], 1)
        self.assertEqual(selected.audio_latents.shape, (1, 3, 2))
        self.assertEqual(selected.latents.get('nested')['video_mask'].item(), 1)
        self.assertEqual(selected.latents.get('sample_rate'), 16000)
        self.assertEqual(selected.audio_data[0]['waveform'].shape, (2, 8))
        self.assertEqual(selected.control_tensor_list[0][0].shape, (2, 2, 2))
        self.assertEqual(selected.loss_multiplier_list, [2.0])
        self.assertEqual(selected.prompt_embeds.attention_mask.shape, (1, 3))
        selected.audio_pred = torch.ones(1, 3, 2)
        self.assertIsNone(batch.audio_pred)

    def test_preserves_prompt_layouts_subclass_fields_and_frozen_integer_tokens(self):
        from extensions_built_in.diffusion_models.anima import AnimaPromptEmbeds
        batch = make_batch()
        batch.prompt_embeds = AdvancedPromptEmbeds(text=[torch.zeros(2, 4), torch.ones(5, 4)],
                                                    ids=[torch.tensor([1]), torch.tensor([2])])
        batch.prompt_embeds.frozen_dtype_keys = ['ids']
        selected = batch.select_item(1)
        self.assertEqual(selected.prompt_embeds.text[0].shape, (5, 4))
        self.assertEqual(selected.prompt_embeds.frozen_dtype_keys, ['ids'])
        self.assertEqual(selected.prompt_embeds.ids[0].dtype, torch.int64)
        batch.prompt_embeds = AnimaPromptEmbeds(
            torch.arange(12.).reshape(2, 3, 2),
            t5_input_ids=torch.tensor([[1, 2], [3, 4]]),
            qwen_attention_mask=torch.ones(2, 3, dtype=torch.int64),
            t5_attention_mask=torch.ones(2, 2, dtype=torch.int64))
        selected = batch.select_item(1)
        self.assertIsInstance(selected.prompt_embeds, AnimaPromptEmbeds)
        torch.testing.assert_close(selected.prompt_embeds.t5_input_ids, torch.tensor([[3, 4]]))
        batch.prompt_embeds = PromptEmbeds(None)
        batch.prompt_embeds.text_embeds = [torch.zeros(2, 4), torch.ones(5, 4)]
        batch.prompt_embeds.attention_mask = [torch.ones(2), torch.ones(5)]
        selected = batch.select_item(1)
        self.assertEqual(selected.prompt_embeds.text_embeds[0].shape, (5, 4))
        self.assertEqual(selected.prompt_embeds.attention_mask[0].shape, (5,))
        batch.prompt_embeds.text_embeds = [torch.zeros(2, 3, 4), torch.ones(2, 5, 4)]
        batch.prompt_embeds.attention_mask = [torch.ones(2, 3), torch.ones(2, 5)]
        selected = batch.select_item(1)
        self.assertEqual([value.shape for value in selected.prompt_embeds.text_embeds], [(1, 3, 4), (1, 5, 4)])


class SharedLossTests(unittest.TestCase):
    def test_v_prediction_uses_clean_latents_for_cached_and_pixel_batches(self):
        trainer = make_loss_trainer(prediction='v_prediction')
        for cached in (True, False):
            with self.subTest(cached=cached):
                batch = make_batch()
                batch.tensor = None if cached else torch.zeros(2, 3, 16, 16)
                target = trainer.sd.noise_scheduler.get_velocity(batch.latents, batch.test_noise, torch.tensor([500, 500]))
                prediction = target.detach().requires_grad_(True)
                loss = calculate(trainer, prediction, batch)
                self.assertEqual(loss.item(), 0)
                loss.backward()
                torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_inverted_mask_uses_each_image_and_retains_valid_rows_for_empty_background(self):
        trainer = make_loss_trainer(inverted_mask_prior=True)
        for video in (False, True):
            for empty_background in (False, True):
                with self.subTest(video=video, empty_background=empty_background):
                    batch = make_batch()
                    batch.mask_tensor = torch.tensor([[[[1., 0.], [1., 0.]]], [[[0., 1.], [0., 1.]]]])
                    if empty_background:
                        batch.mask_tensor[0].fill_(1.)
                    if video:
                        batch.latents = batch.latents.unsqueeze(2).repeat(1, 1, 3, 1, 1)
                        batch.test_noise = batch.test_noise.unsqueeze(2).repeat(1, 1, 3, 1, 1)
                    prediction = batch.test_noise.clone().requires_grad_(True)
                    prior = prediction.detach().clone()
                    prior[1, ..., 0] += 2
                    loss = calculate(trainer, prediction, batch, prior)
                    self.assertEqual(loss.item(), 1.0)
                    loss.backward()
                    self.assertTrue(torch.isfinite(prediction.grad).all())
                    self.assertGreater(prediction.grad[1, ..., 0].abs().sum().item(), 0)

    def test_sparse_bfloat16_background_keeps_unit_mean_weight(self):
        trainer = make_loss_trainer(inverted_mask_prior=True)
        trainer.train_config.dtype = 'bf16'
        batch = make_batch(1)
        batch.latents = torch.zeros(1, 4, 32, 32, dtype=torch.bfloat16)
        batch.test_noise = torch.zeros_like(batch.latents)
        batch.mask_tensor = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)
        batch.mask_tensor[..., 0, 0] = 0
        prior = torch.zeros_like(batch.latents)
        prior[..., 0, 0] = 2
        loss = trainer.calculate_loss(
            batch.test_noise, batch.test_noise, batch.test_noise,
            torch.tensor([500]), batch, mask_multiplier=1.0, prior_pred=prior)
        self.assertEqual(loss.item(), 4.0 * trainer.train_config.inverted_mask_prior_multiplier)


class SingleItemForwardTests(unittest.TestCase):
    def test_real_training_forward_matches_batch_loss_and_gradients(self):
        for cached in (False, True):
            for size in (1, 2, 3):
                for regularization in (False, True):
                    with self.subTest(cached=cached, size=size, regularization=regularization):
                        outcomes = []
                        for single in (False, True):
                            trainer, parameter = make_forward_trainer(single, cached=cached, reg_weight=0.2)
                            batch = make_batch(size)
                            for row in batch.file_items:
                                row.is_reg = regularization
                            value = trainer.train_single_accumulation(batch)
                            outcomes.append((value, parameter.grad))
                            self.assertEqual(len(trainer.seen), size if single else 1)
                            if single:
                                self.assertEqual(trainer.seen, [([f'caption {i}'], [0.5 + i]) for i in range(size)])
                                if not cached:
                                    self.assertTrue(all(len(prompts) == len(sources) == 1 for prompts, sources in trainer.encoded))
                        torch.testing.assert_close(outcomes[0], outcomes[1])

    def test_doubled_captions_select_correct_original_rows_and_mean_reduction(self):
        trainer, parameter = make_forward_trainer(True, doubled=True)
        actual = trainer.train_single_accumulation(make_batch())
        reference, reference_parameter = make_forward_trainer(False)
        expected = reference.train_single_accumulation(make_batch())
        self.assertEqual(trainer.seen, [(['caption 0'], [0.5]), (['caption 1'], [1.5])] * 2)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)

    def test_internal_backward_guidance_and_mean_flow_match_batch_reduction(self):
        for mode in ('direct', 'mean_flow'):
            with self.subTest(mode=mode):
                outcomes = []
                for single in (False, True):
                    trainer, parameter = make_forward_trainer(single, loss_type='mean_flow' if mode == 'mean_flow' else 'mse', reg_weight=0.25)
                    batch = make_batch()
                    for row in batch.file_items:
                        row.is_reg = True
                    if mode == 'direct':
                        for row in batch.file_items:
                            row.dataset_config.guidance_type = 'direct'
                        batch.unconditional_latents = batch.latents + 1
                        trainer.sd.torch_dtype = torch.float32
                        trainer.sd.device_torch = torch.device('cpu')
                        trainer.sd.unet = torch.nn.Linear(1, 1)
                        trainer.sd.add_noise = lambda latents, noise, timesteps: latents * 0.5 + noise * 0.5
                        trainer.sd.predict_noise = lambda latents, **kwargs: parameter * latents
                    with mock.patch('extensions_built_in.sd_trainer.SDTrainer.random.randint', side_effect=[100, 700, 200, 800]):
                        value = trainer.train_single_accumulation(batch)
                    outcomes.append((value, parameter.grad))
                torch.testing.assert_close(outcomes[0], outcomes[1])


class AMPClippingTests(unittest.TestCase):
    def test_actual_accelerate_unscales_once_and_clips_global_multigroup_norm(self):
        # Accelerator state is global; isolate the FP16 simulation from other tests.
        script = '''
import json
import torch
from accelerate import Accelerator
accel = Accelerator(cpu=True)
from testing.test_upstream_sep4_10_training import make_trainer
p = torch.nn.Parameter(torch.tensor([1.]))
q = torch.nn.Parameter(torch.tensor([2.]))
groups = [{'params': [p]}, {'params': [q]}]
optimizer = torch.optim.SGD(groups, lr=.1)
accel.native_amp = True
accel.state._mixed_precision = 'fp16'
accel.scaler = torch.amp.GradScaler('cpu')
accel._optimizers = [optimizer]
trainer = make_trainer(p, optimizer, clip=1.0)
trainer.params = groups
trainer.accelerator = accel
def backward_batch(_):
    value = p.square().sum() + q.square().sum()
    accel.backward(value)
    return value.detach()
trainer.train_single_accumulation = backward_batch
trainer.hook_train_loop([None])
expected = torch.tensor([1., 2.]) - .1 * torch.tensor([2., 4.]) / torch.tensor([2., 4.]).norm()
torch.testing.assert_close(torch.cat([p, q]), expected)
print('AMP_GLOBAL_CLIP_OK')
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                                env={**os.environ, 'NO_ALBUMENTATIONS_UPDATE': '1'},
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('AMP_GLOBAL_CLIP_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
