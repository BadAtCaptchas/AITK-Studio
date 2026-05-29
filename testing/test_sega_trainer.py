import types
import unittest
from collections import OrderedDict

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class _Embeds:
    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self


class _FakeUnet:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class _FakeSD:
    def __init__(self):
        self.unet = _FakeUnet()
        self.calls = []

    def predict_noise(self, **kwargs):
        self.calls.append(kwargs)
        return torch.ones_like(kwargs["latents"])


class _FakeNetwork:
    def __init__(self):
        self.is_active = True


class SegaTeacherPredictionTest(unittest.TestCase):
    def test_teacher_disables_and_restores_network(self):
        trainer = object.__new__(SDTrainer)
        trainer.sd = _FakeSD()
        trainer.network = _FakeNetwork()
        trainer.device_torch = torch.device("cpu")
        trainer.train_config = types.SimpleNamespace(
            dtype="fp32",
            cfg_scale=1.0,
            cfg_rescale=None,
            bypass_guidance_embedding=True,
            sega_distill_base_resolution=1024,
            sega_distill_strength=1.0,
            sega_distill_min_scale=0.5,
            sega_distill_max_scale=2.0,
        )
        trainer._monitor_metrics = OrderedDict()
        trainer._should_record_monitor_metrics = lambda: True

        pred = trainer.get_sega_teacher_prediction(
            noisy_latents=torch.zeros(1, 4, 8, 8),
            conditional_embeds=_Embeds(),
            timesteps=torch.tensor([500]),
            pred_kwargs={"adapter_kwarg": "kept"},
            batch=object(),
            unconditional_embeds=None,
        )

        self.assertTrue(trainer.network.is_active)
        self.assertTrue(trainer.sd.unet.training)
        self.assertTrue(torch.equal(pred, torch.ones(1, 4, 8, 8)))
        self.assertEqual(len(trainer.sd.calls), 1)
        call = trainer.sd.calls[0]
        self.assertEqual(call["adapter_kwarg"], "kept")
        self.assertTrue(call["bypass_guidance_embedding"])
        self.assertTrue(call["sega_config"]["enabled"])

    def test_distillation_is_auxiliary_to_supervised_loss(self):
        trainer = object.__new__(SDTrainer)
        trainer.train_config = types.SimpleNamespace(sega_distill_weight=0.25)
        trainer._monitor_metrics = OrderedDict()
        trainer._should_record_monitor_metrics = lambda: True

        loss = torch.tensor(2.0)
        noise_pred = torch.zeros(1, 1, 2, 2)
        teacher_pred = torch.ones(1, 1, 2, 2)

        final_loss = trainer._add_sega_distill_aux_loss(loss, noise_pred, teacher_pred)

        self.assertAlmostEqual(float(final_loss), 2.25)
        self.assertAlmostEqual(trainer._monitor_metrics["train/sega_supervised_loss"], 2.0)
        self.assertAlmostEqual(trainer._monitor_metrics["train/sega_distill_loss"], 1.0)
        self.assertAlmostEqual(trainer._monitor_metrics["train/sega_distill_weighted_loss"], 0.25)


if __name__ == "__main__":
    unittest.main()
