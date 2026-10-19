"""Custom VAEs must never consume another VAE's cached posterior samples."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from diffusers import AutoencoderKL
from PIL import Image

from toolkit.config_modules import DatasetConfig, ModelConfig
from toolkit.data_loader import AiToolkitDataset
from toolkit.models.base_model import BaseModel
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.vae_cache_identity import with_vae_cache_identity


class VaeCacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.targets = self.root / 'targets'
        self.targets.mkdir()
        Image.new('RGB', (64, 64), (96, 128, 160)).save(self.targets / 'sample.png')
        (self.targets / 'sample.txt').write_text('A small image', encoding='utf-8')
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)

    def save_vae(self, name, seed):
        torch.manual_seed(seed)
        vae = AutoencoderKL(
            in_channels=3, out_channels=3,
            down_block_types=('DownEncoderBlock2D',) * 4,
            up_block_types=('UpDecoderBlock2D',) * 4,
            block_out_channels=(4, 8, 8, 8), layers_per_block=1,
            latent_channels=4, norm_num_groups=2, sample_size=64,
            scaling_factor=0.18215, shift_factor=0.0,
        )
        path = self.root / name
        vae.save_pretrained(path)
        return path

    def holder(self, path, **overrides):
        config = ModelConfig(
            arch='sd1', name_or_path='tiny-local-base', vae_path=str(path),
            dtype='float32', vae_dtype='float32', te_dtype='float32', **overrides,
        )
        model = StableDiffusion(device='cpu', model_config=config, dtype='float32')
        model.vae = AutoencoderKL.from_pretrained(path, local_files_only=True).eval()
        model.unet = None
        model.text_encoder = None
        calls = []
        model.vae.encoder.register_forward_hook(lambda *_: calls.append(1))
        return model, calls

    def dataset(self, model, cache=True):
        config = DatasetConfig(
            folder_path=str(self.targets), resolution=64, caption_dropout_rate=0,
            cache_latents_to_disk=cache, random_crop=False, random_scale=False,
        )
        with mock.patch.object(model, 'set_device_state_preset'), mock.patch.object(model, 'restore_device_state'):
            torch.manual_seed(777)
            return AiToolkitDataset(config, sd=model)

    def test_switching_vae_reencodes_and_reuses_correct_latents(self):
        first, first_calls = self.holder(self.save_vae('a', 101))
        second, second_calls = self.holder(self.save_vae('b', 202))
        a = self.dataset(first).file_list[0]
        b = self.dataset(second).file_list[0]
        self.assertEqual((len(first_calls), len(second_calls)), (1, 1))
        self.assertNotEqual(a.get_latent_path(), b.get_latent_path())
        pixels = self.dataset(second, cache=False)._get_single_item(0).tensor.unsqueeze(0)
        torch.manual_seed(777)
        expected = second.encode_images(pixels).squeeze(0)
        torch.testing.assert_close(b.get_latent(), expected)
        self.assertFalse(torch.equal(a.get_latent(), b.get_latent()))
        reloaded, calls = self.holder(self.root / 'b')
        reused = self.dataset(reloaded).file_list[0]
        self.assertEqual(len(calls), 0)
        torch.testing.assert_close(reused.get_latent(), expected)
        self.assertEqual(BaseModel.get_latent_space_version(second), second.get_latent_space_version())

    def test_same_path_replacement_detected_when_size_and_mtime_are_unchanged(self):
        path = self.save_vae('a', 101)
        first, _ = self.holder(path)
        original_key = first.get_latent_space_version()
        weights = path / 'diffusion_pytorch_model.safetensors'
        before = weights.stat()
        self.save_vae('a', 202)
        os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(weights.stat().st_size, before.st_size)
        second, _ = self.holder(path)
        self.assertNotEqual(original_key, second.get_latent_space_version())
        # A still-loaded VAE keeps the identity of the weights it actually loaded.
        self.assertEqual(first.get_latent_space_version(), original_key)

    def test_precision_and_explicit_namespace(self):
        model, _ = self.holder(self.save_vae('a', 101))
        key = model.get_latent_space_version()
        model.vae_torch_dtype = torch.bfloat16
        self.assertNotEqual(key, model.get_latent_space_version())
        model.model_config.latent_space_version = 'user-managed'
        self.assertEqual(model.get_latent_space_version(), 'user-managed')

    def test_shards_and_cached_hub_revision_are_resolved_offline(self):
        repo = self.root / 'snapshot' / 'vae'
        repo.mkdir(parents=True)
        (repo / 'config.json').write_text('{}', encoding='utf-8')
        shard = repo / 'model-00002-of-00002.safetensors'
        shard.write_bytes(b'first')
        config = SimpleNamespace(vae_path='example/repository', vae_dtype='float32', latent_space_version=None)
        with mock.patch('toolkit.vae_cache_identity.try_to_load_from_cache', side_effect=[None, str(repo / 'config.json')]):
            first = with_vae_cache_identity(SimpleNamespace(model_config=config, vae=None), 'sd1')
        shard.write_bytes(b'other')
        with mock.patch('toolkit.vae_cache_identity.try_to_load_from_cache', return_value=str(repo / 'config.json')):
            second = with_vae_cache_identity(SimpleNamespace(model_config=config, vae=None), 'sd1')
        self.assertNotEqual(first, second)

    def test_single_file_references_use_resolved_weight_content(self):
        weights = self.root / 'vae.safetensors'
        weights.write_bytes(b'first')
        config = SimpleNamespace(vae_path='org/repo/vae/model.safetensors', vae_dtype='float32', latent_space_version=None)
        model = SimpleNamespace(model_config=config, vae=SimpleNamespace(_aitk_vae_source=str(weights)))
        first = with_vae_cache_identity(model, 'moss_image')
        weights.write_bytes(b'other')
        model.vae = SimpleNamespace(_aitk_vae_source=str(weights))
        self.assertNotEqual(first, with_vae_cache_identity(model, 'moss_image'))
        # Standard Hub cache users resolve the same file without a network call.
        model.vae = SimpleNamespace()
        with mock.patch('toolkit.vae_cache_identity.try_to_load_from_cache', return_value=str(weights)) as resolve:
            with_vae_cache_identity(model, 'moss_image')
        resolve.assert_called_once_with('org/repo', 'vae/model.safetensors')


if __name__ == '__main__':
    unittest.main()
