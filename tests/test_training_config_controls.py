"""Real configuration parsing regressions for training UI controls."""

import unittest

from toolkit.config_modules import TrainConfig
from toolkit.training_phases import TrainingPhaseManager


class TrainingConfigControlsTests(unittest.TestCase):
    def test_explicit_regularization_bias_is_independent_of_main_bias(self):
        for main in ("balanced", "style", "content"):
            for regularization in ("balanced", "style", "content"):
                with self.subTest(main=main, regularization=regularization):
                    config = TrainConfig(content_or_style=main, content_or_style_reg=regularization)
                    self.assertEqual(config.content_or_style, main)
                    self.assertEqual(config.content_or_style_reg, regularization)

    def test_omitted_regularization_bias_preserves_legacy_main_bias_default(self):
        self.assertEqual(TrainConfig().content_or_style_reg, "balanced")
        for main in ("balanced", "style", "content"):
            self.assertEqual(TrainConfig(content_or_style=main).content_or_style_reg, main)

    def test_phase_bias_override_restores_the_independent_base_bias(self):
        config = TrainConfig(content_or_style="style", content_or_style_reg="content", steps=4,
                             phases=[{"name": "first", "steps": 2, "content_or_style_reg": "balanced"},
                                     {"name": "second", "steps": 2}])
        manager = TrainingPhaseManager(config)
        manager.apply(config, 0)
        self.assertEqual(config.content_or_style_reg, "balanced")
        manager.apply(config, 2)
        self.assertEqual(config.content_or_style_reg, "content")
        self.assertEqual(config.content_or_style, "style")

    def test_legacy_auto_learning_alias_and_explicit_canonical_override(self):
        for raw, expected in (({"auto_learn": True}, True), ({"auto_learn": False}, False),
                              ({"auto_learn": True, "auto_train": False}, False),
                              ({"auto_learn": False, "auto_train": True}, True)):
            with self.subTest(raw=raw):
                self.assertEqual(TrainConfig(**raw).auto_train, expected)

    def test_automatic_phase_step_cap_is_an_effective_training_limit(self):
        config = TrainConfig(auto_train=True, steps=3000,
                             phases=[{"name": "capped", "steps": 2,
                                      "auto_advance": {"min_steps": 100, "window": 10}}])
        manager = TrainingPhaseManager(config)
        result = manager.maybe_advance_after_step(1)
        self.assertTrue(result.should_stop)
        self.assertEqual(result.reason, "steps")


if __name__ == "__main__":
    unittest.main()
