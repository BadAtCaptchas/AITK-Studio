import os
import sys
import unittest
from types import SimpleNamespace


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.training_phases import TrainingPhaseManager


def make_train_config(**overrides):
    lr = overrides.pop("lr", 1e-4)
    values = {
        "steps": 10,
        "phases": None,
        "save_on_phase_change": True,
        "lr": lr,
        "unet_lr": lr,
        "text_encoder_lr": lr,
        "refiner_lr": lr,
        "embedding_lr": lr,
        "adapter_lr": lr,
        "optimizer": "adamw",
        "optimizer_params": {"weight_decay": 0.01, "eps": 1e-8},
        "lr_scheduler": "constant",
        "lr_scheduler_params": {"warmup_steps": 5},
        "timestep_type": "sigmoid",
        "content_or_style": "balanced",
        "content_or_style_reg": "balanced",
        "loss_type": "mse",
        "min_denoising_steps": 0,
        "max_denoising_steps": 999,
        "min_snr_gamma": None,
        "snr_gamma": None,
        "prompt_dropout_prob": 0.0,
        "noise_offset": 0.0,
        "noise_multiplier": 1.0,
        "target_noise_multiplier": 1.0,
        "random_noise_multiplier": 0.0,
        "random_noise_shift": 0.0,
        "img_multiplier": 1.0,
        "noisy_latent_multiplier": 1.0,
        "latent_multiplier": 1.0,
        "pred_scaler": 1.0,
        "reg_weight": 1.0,
        "max_grad_norm": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TrainingPhaseManagerTest(unittest.TestCase):
    def test_retired_automatic_modes_fail_early_and_false_flags_are_ignored(self):
        from toolkit.config_modules import TrainConfig
        for settings in (
            {'auto_train': True}, {'auto_learn': True},
            {'auto_train': False, 'auto_learn': True},
            {'auto_advance': None},
            {'steps': 10, 'phases': [{'steps': 10, 'auto_advance': {}}]},
            {'steps': 10, 'phases': [{'steps': 10, 'auto_advance': None}]},
        ):
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(ValueError, 'Auto Learn is no longer supported'):
                    TrainConfig(**settings)
                with self.assertRaisesRegex(ValueError, 'Auto Learn is no longer supported'):
                    TrainingPhaseManager(make_train_config(**settings))
        config = TrainConfig(auto_train=False, auto_learn=False, steps=5, phases=[{'steps': 5}])
        self.assertNotIn('auto_train', vars(config))
        self.assertNotIn('auto_learn', vars(config))
        self.assertTrue(TrainingPhaseManager(config).enabled)

    def test_training_and_phase_steps_must_be_positive_finite_integers(self):
        from toolkit.config_modules import TrainConfig
        for value in (None, 0, -1, True, 1.5, float('inf'), float('nan'), '10'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'positive finite integer'):
                    TrainConfig(steps=value)
                with self.assertRaisesRegex(ValueError, 'positive finite integer'):
                    TrainConfig(phases=[{'steps': value}])
        with self.assertRaisesRegex(ValueError, 'positive finite integer'):
            TrainConfig(phases=[{'name': 'missing steps'}])

    def test_final_manual_phase_stops_after_its_finite_steps(self):
        manager = TrainingPhaseManager(make_train_config(steps=2, phases=[{'steps': 2}]))
        self.assertFalse(manager.maybe_advance_after_step(0).should_stop)
        self.assertTrue(manager.maybe_advance_after_step(1).should_stop)
        self.assertFalse(manager.maybe_advance_after_step(2).changed)

    def test_legacy_config_without_phases_is_disabled(self):
        config = make_train_config(steps=25, lr=3e-4)
        manager = TrainingPhaseManager(config)

        self.assertFalse(manager.enabled)
        manager.apply(config, 12)
        self.assertEqual(config.lr, 3e-4)
        self.assertEqual(manager.metrics_for_step(12), {})

    def test_validation_rejects_unsupported_fields_and_step_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Unsupported train.phases\\[0\\] fields: network"):
            TrainingPhaseManager(make_train_config(steps=10, phases=[{"name": "bad", "steps": 10, "network": {}}]))

        with self.assertRaisesRegex(ValueError, "train.steps must equal"):
            TrainingPhaseManager(make_train_config(steps=10, phases=[{"name": "short", "steps": 9}]))

    def test_phase_overlay_inherits_and_shallow_merges_params(self):
        config = make_train_config(
            steps=10,
            lr=1e-4,
            phases=[
                {
                    "name": "detail",
                    "steps": 10,
                    "lr": 5e-5,
                    "optimizer_params": {"weight_decay": 0.2},
                    "lr_scheduler_params": {"total_iters": 10},
                }
            ],
        )
        manager = TrainingPhaseManager(config)

        manager.apply(config, 0)

        self.assertEqual(config.lr, 5e-5)
        self.assertEqual(config.unet_lr, 5e-5)
        self.assertEqual(config.text_encoder_lr, 5e-5)
        self.assertEqual(config.optimizer_params, {"weight_decay": 0.2, "eps": 1e-8})
        self.assertEqual(config.lr_scheduler_params, {"warmup_steps": 5, "total_iters": 10})

    def test_manual_phase_switch_uses_global_step_boundary(self):
        config = make_train_config(
            steps=7,
            phases=[
                {"name": "shape", "steps": 3, "lr": 3e-5},
                {"name": "detail", "steps": 4, "lr": 5e-6},
            ],
        )
        manager = TrainingPhaseManager(config)

        manager.apply(config, 0)
        self.assertEqual(config.lr, 3e-5)
        self.assertFalse(manager.maybe_advance_after_step(1).changed)

        result = manager.maybe_advance_after_step(2)
        manager.apply(config, 3)

        self.assertTrue(result.changed)
        self.assertEqual(result.reason, "steps")
        self.assertEqual(manager.current_index, 1)
        self.assertEqual(manager.get_phase_local_step(3), 0)
        self.assertEqual(config.lr, 5e-6)

    def test_resume_without_phase_metadata_resolves_phase_from_step(self):
        config = make_train_config(
            steps=7,
            phases=[
                {"name": "shape", "steps": 3, "lr": 3e-5},
                {"name": "detail", "steps": 4, "lr": 5e-6},
            ],
        )
        manager = TrainingPhaseManager(config)

        manager.restore_from_training_info({"step": 5})
        manager.apply(config, 5)

        self.assertEqual(manager.current_index, 1)
        self.assertEqual(manager.get_phase_local_step(5), 2)
        self.assertEqual(config.lr, 5e-6)

    def test_resume_with_phase_metadata_preserves_deferred_update_boundary(self):
        config = make_train_config(
            steps=10,
            phases=[
                {"name": "shape", "steps": 6, "lr": 3e-5},
                {"name": "detail", "steps": 4, "lr": 5e-6},
            ],
        )
        manager = TrainingPhaseManager(config)

        manager.restore_from_training_info({"step": 8, "phase_index": 1, "phase_start_step": 7})
        manager.apply(config, 8)

        self.assertEqual(manager.current_index, 1)
        self.assertEqual(manager.get_phase_local_step(8), 1)
        self.assertEqual(config.lr, 5e-6)





if __name__ == "__main__":
    unittest.main()
