"""Tiny real optimizer, checkpoint and manual-phase lifecycle regressions."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import load_file, save_file

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from testing.test_upstream_sep4_10_training import make_trainer
from toolkit.config_modules import TrainConfig
from toolkit.optimizer_checkpoint import OptimizerCheckpointValidationError, load_optimizer_checkpoint_for_resume
from toolkit.prompt_utils import PromptEmbeds
from toolkit.training_phases import TrainingPhaseManager


def make_process(folder, *, value=1.0, use_ema=True, phases=None):
    parameter = torch.nn.Parameter(torch.tensor([value]))
    trainer = make_trainer(parameter, torch.optim.AdamW([parameter], lr=0.1))
    trainer.train_config = TrainConfig(dtype='fp32', optimizer='adamw', lr=0.1,
        optimizer_params={'weight_decay': 0.0}, steps=6, phases=phases,
        ema_config={'use_ema': use_ema, 'ema_decay': 0.9})
    trainer.params = [{'params': [parameter], 'lr': 0.1}]
    trainer.phase_manager = TrainingPhaseManager(trainer.train_config)
    trainer.phase_manager.apply(trainer.train_config, 0)
    trainer.device_torch = torch.device('cpu')
    trainer.accelerator = SimpleNamespace(is_main_process=True,
        prepare=lambda component: component, clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
    trainer.modules_being_trained = []
    trainer.logger = SimpleNamespace(start=lambda: None)
    trainer.is_caching_text_embeddings = False
    trainer.do_long_prompts = False
    trainer.sd = SimpleNamespace(is_multistage=False, vae=None, unet=None,
        text_encoder=None, refiner_unet=None, network=None, noise_scheduler=None,
        torch_dtype=torch.float32,
        save=lambda path, meta, dtype: save_file({'weight': parameter.detach().to(dtype)}, path))
    trainer.encode_static_prompt = lambda *args, **kwargs: PromptEmbeds(torch.zeros(1, 2, 3))
    trainer.save_root = str(folder)
    trainer.network = None
    trainer._resume_checkpoint_step = trainer._resume_checkpoint_path = None
    trainer._pending_ema_resume_state = None
    trainer.step_num = 0
    trainer.job = SimpleNamespace(name='tiny')
    trainer.meta = {}
    trainer.is_fine_tuning = True
    trainer.named_lora = False
    trainer.decorator = trainer.snr_gos = None
    trainer.save_config = SimpleNamespace(dtype='float32', save_format='safetensors')
    trainer.update_training_metadata = lambda: None
    trainer.clean_up_saves = lambda: None
    trainer.setup_optimizer(load_existing_state=False)
    trainer.setup_lr_scheduler()
    return trainer, parameter


def update(trainer):
    trainer.hook_train_loop(torch.tensor([1.0]))
    trainer.step_num += 1


class EMALifecycleTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'Device staging requires CUDA')
    def test_shadow_follows_live_parameter_device_during_staging(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer, parameter = make_process(folder)
            trainer.hook_before_train_loop()
            parameter.data = parameter.data.to('cuda')
            parameter.data.fill_(2.0)
            trainer.ema.update()
            self.assertEqual(trainer.ema.shadow_params[0].device, parameter.device)
            torch.testing.assert_close(trainer.ema.shadow_params[0].cpu(), torch.tensor([1.1]))
            parameter.data = parameter.data.cpu()
            parameter.data.fill_(3.0)
            trainer.ema.update()
            self.assertEqual(trainer.ema.shadow_params[0].device, parameter.device)
            torch.testing.assert_close(trainer.ema.shadow_params[0], torch.tensor([1.29]))

    def test_actual_pretraining_hook_initializes_ema_and_only_updates_at_optimizer_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer, parameter = make_process(folder)
            trainer.hook_before_train_loop()
            self.assertIsNotNone(trainer.ema)
            trainer.is_grad_accumulation_step = True
            update(trainer)
            torch.testing.assert_close(trainer.ema.shadow_params[0], torch.ones(1))
            trainer.is_grad_accumulation_step = False
            update(trainer)
            torch.testing.assert_close(trainer.ema.shadow_params[0], 0.9 + 0.1 * parameter)
            before = parameter.detach().clone()
            trainer.ema.eval()
            torch.testing.assert_close(parameter, trainer.ema.shadow_params[0])
            trainer.ema.train()
            torch.testing.assert_close(parameter, before)

    def test_real_save_resume_preserves_live_parameters_moments_and_ema_history(self):
        for resumed_use_ema in (True, False):
            with self.subTest(resumed_use_ema=resumed_use_ema), tempfile.TemporaryDirectory() as folder:
                trainer, parameter = make_process(folder)
                trainer.hook_before_train_loop()
                update(trainer)
                update(trainer)
                live = parameter.detach().clone()
                shadow = trainer.ema.shadow_params[0].clone()
                with mock.patch('jobs.process.BaseSDTrainProcess.flush'):
                    trainer.save(step=2)
                model_path = Path(folder) / 'tiny_000000002.safetensors'
                exported = load_file(str(model_path))['weight']
                torch.testing.assert_close(exported, shadow)
                torch.testing.assert_close(parameter, live)
                resumed, resumed_parameter = make_process(folder, value=exported.item(), use_ema=resumed_use_ema)
                resumed._resume_checkpoint_step = resumed.step_num = 2
                resumed._resume_checkpoint_path = str(model_path)
                with mock.patch('jobs.process.BaseSDTrainProcess.flush'):
                    resumed.load_optimizer_state()
                resumed.hook_before_train_loop()
                torch.testing.assert_close(resumed_parameter, live)
                if resumed_use_ema:
                    torch.testing.assert_close(resumed.ema.shadow_params[0], shadow)
                else:
                    self.assertIsNone(resumed.ema)
                self.assertTrue(resumed.optimizer.state)
                update(trainer)
                update(resumed)
                torch.testing.assert_close(resumed_parameter, parameter)
                if resumed_use_ema:
                    torch.testing.assert_close(resumed.ema.shadow_params[0], trainer.ema.shadow_params[0])
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(resumed.optimizer.state[resumed_parameter][key], trainer.optimizer.state[parameter][key])

    def test_phase_optimizer_restart_keeps_ema_history(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer, parameter = make_process(folder, phases=[{'steps': 2}, {'steps': 4, 'lr': 0.02}])
            trainer.hook_before_train_loop()
            update(trainer)
            update(trainer)
            ema = trainer.ema
            shadow = ema.shadow_params[0].clone()
            self.assertTrue(trainer.phase_manager.maybe_advance_after_step(1).changed)
            trainer.restart_training_phase_optimizer()
            self.assertIs(trainer.ema, ema)
            torch.testing.assert_close(ema.shadow_params[0], shadow)
            self.assertEqual(trainer.optimizer.param_groups[0]['lr'], 0.02)
            update(trainer)
            torch.testing.assert_close(ema.shadow_params[0], 0.9 * shadow + 0.1 * parameter)

    def test_ema_resume_rejects_wrong_model_and_bad_tensor_shapes(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer, _ = make_process(folder)
            trainer.hook_before_train_loop()
            update(trainer)
            with mock.patch('jobs.process.BaseSDTrainProcess.flush'):
                trainer.save(step=1)
            checkpoint = Path(folder) / 'tiny_000000001.safetensors'
            optimizer_path = Path(folder) / 'optimizer.pt'
            other = Path(folder) / 'different.safetensors'
            save_file({'weight': torch.tensor([42.0])}, str(other))
            with self.assertRaisesRegex(OptimizerCheckpointValidationError, 'does not match'):
                load_optimizer_checkpoint_for_resume(str(optimizer_path), 1, str(other), include_ema_state=True)
            payload = torch.load(optimizer_path, weights_only=True)
            payload['ema_resume_state']['parameters'][0] = torch.zeros(2)
            torch.save(payload, optimizer_path)
            resumed, parameter = make_process(folder, value=0.5)
            resumed._resume_checkpoint_step = 1
            resumed._resume_checkpoint_path = str(checkpoint)
            resumed.load_optimizer_state()
            torch.testing.assert_close(parameter, torch.tensor([0.5]))
            self.assertFalse(resumed.optimizer.state)
            self.assertIsNone(resumed._pending_ema_resume_state)

    def test_ema_checkpoint_supports_diffusers_directory_exports(self):
        from toolkit.optimizer_checkpoint import atomic_save_optimizer_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            trainer, _ = make_process(folder)
            trainer.hook_before_train_loop()
            update(trainer)
            checkpoint = Path(folder) / 'model'
            checkpoint.mkdir()
            (checkpoint / 'config.json').write_text('{}', encoding='utf-8')
            save_file({'weight': torch.ones(1)}, str(checkpoint / 'model.safetensors'))
            path = Path(folder) / 'optimizer.pt'
            atomic_save_optimizer_checkpoint(str(path), trainer.optimizer.state_dict(), 1, str(checkpoint),
                                             ema_resume_state=trainer.get_ema_resume_state())
            _, _, state = load_optimizer_checkpoint_for_resume(str(path), 1, str(checkpoint), include_ema_state=True)
            self.assertEqual(state['format_version'], 1)
            (checkpoint / 'config.json').write_text('{"changed":true}', encoding='utf-8')
            with self.assertRaisesRegex(OptimizerCheckpointValidationError, 'does not match'):
                load_optimizer_checkpoint_for_resume(str(path), 1, str(checkpoint), include_ema_state=True)


class StepSchedulerTests(unittest.TestCase):
    def test_actual_scheduler_setup_decay_and_resume_with_and_without_phases(self):
        for phases in (None, [{'steps': 3}, {'steps': 3}]):
            with self.subTest(phases=phases), tempfile.TemporaryDirectory() as folder:
                trainer, parameter = make_process(folder, use_ema=False, phases=phases)
                trainer.train_config.lr_scheduler = 'step'
                trainer.train_config.lr_scheduler_params = {'step_size': 2, 'gamma': 0.5}
                trainer.setup_lr_scheduler()
                lrs = []
                for _ in range(4):
                    lrs.append(trainer.optimizer.param_groups[0]['lr'])
                    parameter.grad = torch.ones_like(parameter)
                    trainer.optimizer.step()
                    trainer.lr_scheduler.step()
                self.assertEqual(lrs, [0.1, 0.1, 0.05, 0.05])
                resumed, _ = make_process(folder, use_ema=False, phases=phases)
                resumed.train_config.lr_scheduler = 'step'
                resumed.train_config.lr_scheduler_params = {'step_size': 2, 'gamma': 0.5}
                resumed.setup_lr_scheduler()
                resumed.lr_scheduler.step(4)
                self.assertEqual(resumed.optimizer.param_groups[0]['lr'], trainer.optimizer.param_groups[0]['lr'])


if __name__ == '__main__':
    unittest.main()
