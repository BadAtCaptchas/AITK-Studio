"""Real-tensor failures must never look like successful training updates."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.guidance import _backward_guidance_loss
from toolkit.training_numerics import TrainingNumericsError, require_finite_gradients
from testing.test_shared_training_batch_math import make_batch, make_forward_trainer
from testing.test_upstream_sep4_10_training import make_trainer


class InvalidBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.square()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float('inf'))


class TrainingNumericsTests(unittest.TestCase):
    def test_invalid_fully_weighted_loss_never_calls_backward(self):
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.step_num = 7
        trainer.accelerator = SimpleNamespace(backward=mock.Mock())
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(TrainingNumericsError, 'step 7'):
                trainer._checked_backward(torch.tensor(value, requires_grad=True))
        trainer.accelerator.backward.assert_not_called()

    def test_guidance_checks_after_backward_scale(self):
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.accelerator = SimpleNamespace(backward=mock.Mock())
        with self.assertRaises(TrainingNumericsError):
            _backward_guidance_loss(torch.tensor(1., requires_grad=True), {
                'backward_fn': trainer._checked_backward, 'backward_scale': float('inf'),
            })
        trainer.accelerator.backward.assert_not_called()

    def test_llm_checks_after_loss_multiplier(self):
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.network = None
        trainer.device_torch = torch.device('cpu')
        trainer.timer = make_trainer(torch.nn.Parameter(torch.ones(1)), None).timer
        trainer.sd = SimpleNamespace(get_llm_loss=lambda _: torch.tensor(1., requires_grad=True))
        trainer.additional_logs = {}
        trainer.accelerator = SimpleNamespace(backward=mock.Mock())
        batch = SimpleNamespace(get_network_weight_list=lambda: [1.], loss_multiplier_list=[float('inf')])
        with self.assertRaises(TrainingNumericsError):
            trainer.train_llm_accumulation(batch)
        trainer.accelerator.backward.assert_not_called()

    def test_diffusion_and_mean_flow_reject_nonfinite_predictions(self):
        for loss_type in ('mse', 'mean_flow'):
            with self.subTest(loss_type=loss_type):
                trainer, parameter = make_forward_trainer(False, loss_type=loss_type)
                parameter.data.fill_(float('inf'))
                with self.assertRaises(TrainingNumericsError):
                    trainer.train_single_accumulation(make_batch())
                self.assertIsNone(parameter.grad)

    def test_invalid_gradient_does_not_update_optimizer_ema_scheduler_or_counts(self):
        for accumulating in (False, True):
            for optimizer_name in ('adamw', 'adafactor'):
                with self.subTest(accumulating=accumulating, optimizer=optimizer_name):
                    parameter = torch.nn.Parameter(torch.tensor([1.]))
                    optimizer = torch.optim.AdamW([parameter], lr=.1)
                    trainer = make_trainer(parameter, optimizer)
                    trainer.is_grad_accumulation_step = accumulating
                    trainer.train_config.optimizer = optimizer_name
                    trainer.lr_scheduler = SimpleNamespace(step=mock.Mock())
                    trainer.ema = SimpleNamespace(update=mock.Mock())

                    def batch(_):
                        loss = InvalidBackward.apply(parameter).mean()
                        loss.backward()
                        return loss.detach()

                    trainer.train_single_accumulation = batch
                    with self.assertRaisesRegex(TrainingNumericsError, 'gradients'):
                        trainer.hook_train_loop(None)
                    torch.testing.assert_close(parameter, torch.tensor([1.]))
                    self.assertEqual(len(optimizer.state), 0)
                    self.assertEqual(trainer._accumulated_microbatches, 0)
                    self.assertEqual(trainer.steps_this_boundary, 0)
                    trainer.lr_scheduler.step.assert_not_called()
                    trainer.ema.update.assert_not_called()

    def test_private_accumulation_and_sparse_gradients_are_checked(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        parameter._accum_grad = torch.tensor([1., float('nan')])
        with self.assertRaises(TrainingNumericsError):
            require_finite_gradients(optimizer)
        del parameter._accum_grad
        parameter.grad = torch.sparse_coo_tensor(torch.tensor([[1]]), torch.tensor([float('inf')]), (2,))
        with self.assertRaises(TrainingNumericsError):
            require_finite_gradients(optimizer)

    def test_overflow_in_merged_private_buffers_is_rejected(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        trainer = make_trainer(parameter, optimizer)
        trainer.lr_scheduler = SimpleNamespace(step=mock.Mock())

        def batch(_):
            parameter.grad = torch.tensor([3e38])
            parameter._accum_grad = torch.tensor([3e38])
            return torch.tensor(1.)

        trainer.train_single_accumulation = batch
        with self.assertRaises(TrainingNumericsError):
            trainer.hook_train_loop(None)
        torch.testing.assert_close(parameter, torch.ones(1))
        trainer.lr_scheduler.step.assert_not_called()

    def test_distributed_failure_is_reported_on_other_ranks(self):
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.accelerator = SimpleNamespace(
            num_processes=2, device=torch.device('cpu'),
            reduce=lambda flag, reduction: torch.ones_like(flag), backward=mock.Mock(),
        )
        with self.assertRaises(TrainingNumericsError):
            trainer._checked_backward(torch.tensor(1., requires_grad=True))
        trainer.accelerator.backward.assert_not_called()


if __name__ == '__main__':
    unittest.main()
