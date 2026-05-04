import os
import shutil
import sys
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset
import toolkit.dataloader_mixins as dataloader_mixins


class FakeSD:
    def __init__(self):
        self.adapter = None
        self.device = "cpu"
        self.device_torch = torch.device("cpu")
        self.encode_control_in_text_embeddings = False
        self.is_audio_model = False
        self.is_auraflow = False
        self.is_flux = False
        self.is_v3 = False
        self.is_xl = False
        self.latent_space_version = None
        self.model_config = SimpleNamespace(
            arch="sd1",
            is_pixart_sigma=False,
            latent_space_version=None,
        )
        self.sample_rate = 48000
        self.te_padding_side = "right"
        self.torch_dtype = torch.float32
        self.unet = SimpleNamespace()
        self.use_raw_control_images = False
        self.vae = SimpleNamespace()
        self.encode_calls = 0
        self.device_state_presets = []
        self.restore_calls = 0

    def get_bucket_divisibility(self):
        return 32

    def set_device_state_preset(self, preset):
        self.device_state_presets.append(preset)

    def restore_device_state(self):
        self.restore_calls += 1

    def encode_images(self, imgs):
        self.encode_calls += 1
        batch_size = imgs.shape[0]
        height = max(1, imgs.shape[-2] // 8)
        width = max(1, imgs.shape[-1] // 8)
        return torch.full((batch_size, 4, height, width), float(self.encode_calls))


def _tmp_root():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp")
    os.makedirs(root, exist_ok=True)
    return root


def _write_image(path, size=(96, 80), color=(128, 96, 64)):
    Image.new("RGB", size, color=color).save(path)


class LatentCacheReuseTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = os.path.join(_tmp_root(), f"latent_cache_reuse_{uuid.uuid4().hex}")
        os.makedirs(self.temp_dir, exist_ok=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _dataset_config(self, **overrides):
        kwargs = {
            "dataset_path": self.temp_dir,
            "resolution": 64,
            "buckets": True,
            "cache_latents_to_disk": True,
        }
        kwargs.update(overrides)
        return DatasetConfig(**kwargs)

    def test_resume_reuses_existing_disk_latents_without_device_cache_setup(self):
        _write_image(os.path.join(self.temp_dir, "a.png"))
        _write_image(os.path.join(self.temp_dir, "b.png"), color=(64, 128, 96))

        first_sd = FakeSD()
        AiToolkitDataset(self._dataset_config(), batch_size=1, sd=first_sd)

        self.assertEqual(first_sd.encode_calls, 2)
        self.assertEqual(first_sd.device_state_presets, ["cache_latents"])
        self.assertEqual(first_sd.restore_calls, 1)

        resumed_sd = FakeSD()
        resumed_dataset = AiToolkitDataset(self._dataset_config(), batch_size=1, sd=resumed_sd)

        self.assertEqual(resumed_sd.encode_calls, 0)
        self.assertEqual(resumed_sd.device_state_presets, [])
        self.assertEqual(resumed_sd.restore_calls, 0)
        self.assertTrue(all(file_item.is_latent_cached for file_item in resumed_dataset.file_list))

    def test_num_repeats_encodes_each_unique_latent_once(self):
        _write_image(os.path.join(self.temp_dir, "a.png"))

        sd = FakeSD()
        dataset = AiToolkitDataset(self._dataset_config(num_repeats=3), batch_size=1, sd=sd)
        latent_paths = {file_item.get_latent_path() for file_item in dataset.file_list}

        self.assertEqual(len(dataset.file_list), 3)
        self.assertEqual(len(latent_paths), 1)
        self.assertEqual(sd.encode_calls, 1)

    def test_memory_cache_loads_each_unique_existing_latent_once(self):
        _write_image(os.path.join(self.temp_dir, "a.png"))

        first_sd = FakeSD()
        AiToolkitDataset(self._dataset_config(num_repeats=3), batch_size=1, sd=first_sd)

        resumed_sd = FakeSD()
        with mock.patch.object(dataloader_mixins, "load_file", wraps=dataloader_mixins.load_file) as load_mock:
            resumed_dataset = AiToolkitDataset(
                self._dataset_config(num_repeats=3, cache_latents=True),
                batch_size=1,
                sd=resumed_sd,
            )

        self.assertEqual(resumed_sd.encode_calls, 0)
        self.assertEqual(load_mock.call_count, 1)
        self.assertTrue(all(file_item._encoded_latent is not None for file_item in resumed_dataset.file_list))

    def test_cached_random_crop_uses_stable_latent_path_on_resume(self):
        _write_image(os.path.join(self.temp_dir, "wide.png"), size=(180, 96))

        first_sd = FakeSD()
        first_dataset = AiToolkitDataset(
            self._dataset_config(random_crop=True),
            batch_size=1,
            sd=first_sd,
        )
        first_paths = [file_item.get_latent_path() for file_item in first_dataset.file_list]
        first_crops = [(file_item.crop_x, file_item.crop_y) for file_item in first_dataset.file_list]

        resumed_sd = FakeSD()
        resumed_dataset = AiToolkitDataset(
            self._dataset_config(random_crop=True),
            batch_size=1,
            sd=resumed_sd,
        )
        resumed_paths = [file_item.get_latent_path() for file_item in resumed_dataset.file_list]
        resumed_crops = [(file_item.crop_x, file_item.crop_y) for file_item in resumed_dataset.file_list]

        self.assertEqual(first_paths, resumed_paths)
        self.assertEqual(first_crops, resumed_crops)
        self.assertEqual(resumed_sd.encode_calls, 0)
        self.assertEqual(resumed_sd.device_state_presets, [])


if __name__ == "__main__":
    unittest.main()
