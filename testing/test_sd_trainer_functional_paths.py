"""Real trainer regressions for FFT and watermark functional operations."""
import unittest
from types import SimpleNamespace

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from testing.test_shared_training_batch_math import calculate, make_batch, make_loss_trainer
from toolkit.prompt_utils import PromptEmbeds


class WatermarkNetwork:
    def __init__(self):
        self.is_active = True
        self.authenlora_rank_scale = torch.tensor([[0.75, 1.25]])

    def set_authenlora_rank_scale(self, scale):
        self.authenlora_rank_scale = scale

    def clear_authenlora_rank_scale(self):
        self.authenlora_rank_scale = None


class SDTrainerFunctionalPathsTest(unittest.TestCase):
    def test_fft_loss_matches_constant_image_analytic_loss_and_gradient(self):
        trainer = make_loss_trainer(flow=True, do_fft_loss=True)
        batch = make_batch(1)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        prediction = parameter.expand_as(batch.latents)

        # Clean latents are zero and noise/noisy latents are 2 at t=0.5.
        # Base loss is (p-2)^2; the 2x2 constant-image FFT term is
        # (2-0.5p)^2, giving 3.25 and derivative -3.5 at p=1.
        loss = calculate(trainer, prediction, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), 3.25)
        self.assertAlmostEqual(trainer.additional_logs['loss/fft'], 2.25)
        loss.backward()
        self.assertTrue(torch.isfinite(parameter.grad))
        self.assertAlmostEqual(parameter.grad.item(), -3.5)

    def test_authenlora_resizes_latent_delta_and_keeps_gradient(self):
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.device_torch = torch.device('cpu')
        trainer.watermark_config = SimpleNamespace(zero_message_probability=0.0)
        delta = torch.nn.Parameter(torch.tensor([[[[0., 2.], [4., 6.]]]]))
        bits = torch.ones(2, 2)
        rank_scale = torch.tensor([[0.75, 1.25], [0.75, 1.25]])
        encoded_sources = []

        def encode_latent_delta(latents, message):
            encoded_sources.append((latents.shape, message.clone()))
            return delta

        trainer.authenlora = SimpleNamespace(
            choose_bits=lambda size, force_zero: (bits, None),
            rank_scale_from_bits=lambda message: rank_scale,
            encode_latent_delta=encode_latent_delta,
        )
        network = WatermarkNetwork()
        batch = SimpleNamespace(latents=torch.ones(1, 1, 2, 2))
        noisy_latents = torch.ones(2, 1, 4, 4)

        output, selected_bits, force_zero = trainer._prepare_authenlora_batch(
            network, noisy_latents, batch)

        expected_delta = torch.tensor([
            [0., 0.5, 1.5, 2.],
            [1., 1.5, 2.5, 3.],
            [3., 3.5, 4.5, 5.],
            [4., 4.5, 5.5, 6.],
        ])[None, None].expand(2, -1, -1, -1)
        torch.testing.assert_close(output, noisy_latents + expected_delta)
        self.assertFalse(force_zero)
        self.assertIs(selected_bits, bits)
        self.assertIs(network.authenlora_rank_scale, rank_scale)
        self.assertEqual(encoded_sources[0][0], (1, 1, 2, 2))
        torch.testing.assert_close(encoded_sources[0][1], bits)
        output.sum().backward()
        torch.testing.assert_close(delta.grad, torch.full_like(delta, 8.0))

    def test_watermark_aux_loss_restores_clean_prediction_state_and_backpropagates(self):
        for initially_training in (True, False):
            with self.subTest(initially_training=initially_training):
                trainer = SDTrainer.__new__(SDTrainer)
                trainer.authenlora = SimpleNamespace(last_bit_accuracy=0.75,
                    last_zero_bit_accuracy=None, last_verification_pass=True)
                trainer.watermark_config = SimpleNamespace(watermark_loss_weight=0.25,
                    style_loss_weight=0.5, verify_every=0)
                trainer.additional_logs = {}
                trainer.sd = SimpleNamespace(unet=torch.nn.Linear(1, 1), vae=None)
                trainer.sd.unet.train(initially_training)
                network = WatermarkNetwork()
                previous_scale = network.authenlora_rank_scale
                clean_latents = torch.full((1, 1, 2, 2), 2.0, requires_grad=True)
                noise_pred = torch.nn.Parameter(torch.full_like(clean_latents, 3.0))
                style_parameter = torch.nn.Parameter(torch.tensor(2.0))
                clean_calls = []

                def predict_clean(**kwargs):
                    self.assertFalse(network.is_active)
                    self.assertIsNone(network.authenlora_rank_scale)
                    self.assertFalse(trainer.sd.unet.training)
                    self.assertFalse(torch.is_grad_enabled())
                    clean_calls.append(kwargs['noisy_latents'])
                    return kwargs['noisy_latents'] * 0.5

                trainer.predict_noise = predict_clean
                loss = trainer._add_authenlora_aux_loss(
                    loss=style_parameter.square(), network=network,
                    clean_latents=clean_latents, watermarked_latents=clean_latents + 1,
                    noise_pred=noise_pred, timesteps=torch.tensor([500]),
                    conditional_embeds=PromptEmbeds(torch.zeros(1, 2, 3)),
                    unconditional_embeds=None, batch=SimpleNamespace(), pred_kwargs={},
                    bits=torch.ones(1, 2), zero_message=False,
                )

                self.assertEqual(len(clean_calls), 1)
                self.assertTrue(network.is_active)
                self.assertIs(network.authenlora_rank_scale, previous_scale)
                self.assertEqual(trainer.sd.unet.training, initially_training)
                self.assertAlmostEqual(loss.item(), 3.0)
                self.assertEqual(trainer.additional_logs['watermark/loss'], 1.0)
                self.assertEqual(trainer.additional_logs['watermark/bit_acc'], 0.75)
                loss.backward()
                torch.testing.assert_close(noise_pred.grad, torch.full_like(noise_pred, 0.25))
                self.assertAlmostEqual(style_parameter.grad.item(), 2.0)
                self.assertIsNone(clean_latents.grad)


if __name__ == '__main__':
    unittest.main()
