import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.models.base_model import BaseModel


class FakeModule:
    def __init__(self, device="cpu", training=True):
        self.device = torch.device(device)
        self.training = training
        self.requires_grad = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self.device = torch.device(device)
        return self

    def requires_grad_(self, requires_grad):
        self.requires_grad = requires_grad
        return self


def make_base_model_with_fake_modules(text_encoder):
    model = BaseModel.__new__(BaseModel)
    model.vae = FakeModule("cuda:0")
    model.unet = FakeModule("cuda:0")
    model.text_encoder = text_encoder
    model.adapter = None
    model.refiner_unet = None
    model.device_torch = torch.device("cuda:0")
    model.vae_device_torch = torch.device("cuda:0")
    model.te_device_torch = torch.device("cuda:0")
    model.device_state = None
    model.get_model_has_grad = lambda: True
    model.get_te_has_grad = lambda: True
    return model


class BaseModelDeviceStatePresetTest(unittest.TestCase):
    def test_cache_text_encoder_uses_text_encoder_device_for_list_encoders(self):
        text_encoders = [FakeModule("cpu"), FakeModule("cpu")]
        model = make_base_model_with_fake_modules(text_encoders)

        model.set_device_state_preset("cache_text_encoder")

        self.assertEqual(model.vae.device, torch.device("cpu"))
        self.assertEqual(model.unet.device, torch.device("cpu"))
        self.assertTrue(all(encoder.device == torch.device("cuda:0") for encoder in text_encoders))
        self.assertFalse(model.vae.training)
        self.assertFalse(model.unet.training)
        self.assertTrue(all(not encoder.training for encoder in text_encoders))

    def test_cache_text_encoder_uses_text_encoder_device_for_single_encoder(self):
        text_encoder = FakeModule("cpu")
        model = make_base_model_with_fake_modules(text_encoder)

        model.set_device_state_preset("cache_text_encoder")

        self.assertEqual(model.vae.device, torch.device("cpu"))
        self.assertEqual(model.unet.device, torch.device("cpu"))
        self.assertEqual(text_encoder.device, torch.device("cuda:0"))

    def test_unknown_device_state_preset_raises(self):
        model = BaseModel.__new__(BaseModel)

        with self.assertRaises(ValueError):
            model.set_device_state_preset("cache_text_embedings")


if __name__ == "__main__":
    unittest.main()
