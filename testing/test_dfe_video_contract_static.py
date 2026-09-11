"""Behavioral replacement for old assertions that folded video before VAE decode."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from toolkit.models import diffusion_feature_extraction as features


class VideoModel:
    def __init__(self, x0_pred=False):
        self.x0_pred = x0_pred
        self.noise_scale = 2.0 if x0_pred else 1.0
        self.decoded_inputs = []
        self.vae = SimpleNamespace(
            device=torch.device('cpu'), dtype=torch.float32,
            config=SimpleNamespace(scaling_factor=1.0, shift_factor=0.0),
            decode=self.decode_latents,
        )

    def decode_latents(self, latents):
        self.decoded_inputs.append(latents.detach().clone())
        if latents.ndim != 5:
            raise AssertionError('Video VAE must receive full clips')
        return F.interpolate(latents, size=(5, 2, 2), mode='nearest')


class TinyDino(nn.Module):
    def forward(self, pixel_values):
        patches = pixel_values.flatten(2).transpose(1, 2)
        cls = patches.mean(dim=1, keepdim=True)
        register = torch.full_like(cls, 10000)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, register, patches], dim=1))


def make_dfe(version, model, partial=False):
    cls = getattr(features, f'DiffusionFeatureExtractor{version}')
    dfe = cls.__new__(cls)
    nn.Module.__init__(dfe)
    dfe.sd_ref = lambda: model
    dfe.vae = model.vae
    dfe.model = SimpleNamespace(device=torch.device('cpu'), dtype=torch.float32)
    dfe.losses = {}
    dfe.step = 0
    dfe.log_every = 100
    dfe.do_partial_step = partial
    if version == 4:
        dfe.get_siglip_features = lambda x: x
    elif version == 6:
        dfe.model = TinyDino()
        dfe.num_prefix_tokens = 2
        dfe.cls_weight = 0.1
        dfe.prepare_inputs = lambda x: {'pixel_values': x}
    elif version == 7:
        dfe.get_pred = lambda x: SimpleNamespace(head=x, depth=x, normals=x, segmentation=x)
    elif version == 9:
        dfe.get_pred = lambda x: x
    elif version == 10:
        dfe.get_lpips_features = lambda x: [x, x * 2]
    return dfe


class DFEVideoContractTests(unittest.TestCase):
    def test_video_decode_keeps_clips_and_backpropagates_to_every_latent_frame(self):
        for version in (4, 6, 7, 9, 10):
            for x0 in (False, True):
                with self.subTest(version=version, x0=x0):
                    torch.manual_seed(4)
                    model = VideoModel(x0)
                    dfe = make_dfe(version, model)
                    pred = torch.randn(2, 3, 2, 2, 2, requires_grad=True)
                    loss = dfe(
                        noise=torch.ones_like(pred), noise_pred=pred,
                        noisy_latents=torch.ones_like(pred), timesteps=torch.tensor([100., 700.]),
                        batch=SimpleNamespace(tensor=torch.zeros(2, 5, 3, 2, 2), latents=torch.zeros_like(pred)),
                        scheduler=None,
                    ).mean()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(tuple(x.shape) == (2, 3, 2, 2, 2) for x in model.decoded_inputs))
                    if x0:
                        torch.testing.assert_close(model.decoded_inputs[-1], pred.detach())
                    loss.backward()
                    self.assertTrue((pred.grad.abs().sum(dim=(1, 3, 4)) > 0).all())

    def test_partial_x0_steps_reconstruct_the_same_target_with_scaled_noise(self):
        for version in (7, 9, 10):
            with self.subTest(version=version):
                model = VideoModel(True)
                dfe = make_dfe(version, model, partial=True)
                clean = torch.full((2, 3, 2, 2, 2), 0.25)
                noise = torch.ones_like(clean)
                tv = torch.tensor([0.5, 0.75]).view(2, 1, 1, 1, 1)
                noisy = (1 - tv) * clean + tv * noise * model.noise_scale
                dfe(noise=noise, noise_pred=clean, noisy_latents=noisy,
                    timesteps=tv.flatten() * 1000,
                    batch=SimpleNamespace(tensor=torch.zeros(2, 5, 3, 2, 2), latents=clean), scheduler=None)
                self.assertEqual(len(model.decoded_inputs), 2)
                torch.testing.assert_close(model.decoded_inputs[0], model.decoded_inputs[1])

    def test_dino_excludes_register_tokens(self):
        dfe = make_dfe(6, VideoModel(True))
        pixels = torch.ones(2, 3, 2, 2)
        cls, patches = dfe._dino_features({'pixel_values': pixels})
        self.assertEqual(tuple(patches.shape), (2, 4, 3))
        self.assertTrue(torch.equal(patches, torch.ones_like(patches)))
        self.assertTrue(torch.equal(cls, torch.ones_like(cls)))


if __name__ == '__main__':
    unittest.main()
