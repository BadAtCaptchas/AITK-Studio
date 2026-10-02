import os
import shutil
import sys
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
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
        self.text_embedding_space_version = "sd1"
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

    def get_latent_space_version(self):
        return self.latent_space_version or 'sd1'

    def get_text_embedding_space_version(self):
        return self.text_embedding_space_version

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

    def test_text_embedding_cache_path_uses_model_space_version(self):
        _write_image(os.path.join(self.temp_dir, "a.png"))

        first_sd = FakeSD()
        first_sd.text_embedding_space_version = "sd1_te_v1"
        first_dataset = AiToolkitDataset(
            self._dataset_config(cache_latents_to_disk=False),
            batch_size=1,
            sd=first_sd,
        )
        first_path = first_dataset.file_list[0].get_text_embedding_path(recalculate=True)

        second_sd = FakeSD()
        second_sd.text_embedding_space_version = "sd1_te_v2"
        second_dataset = AiToolkitDataset(
            self._dataset_config(cache_latents_to_disk=False),
            batch_size=1,
            sd=second_sd,
        )
        second_path = second_dataset.file_list[0].get_text_embedding_path(recalculate=True)

        self.assertNotEqual(first_path, second_path)

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

    def test_same_size_media_replaced_within_one_second_rebuilds_latents(self):
        image_path = os.path.join(self.temp_dir, 'a.png')
        images = []
        for color in ('red', 'black'):
            buffer = BytesIO()
            Image.new('RGB', (96, 80), color=color).save(buffer, format='PNG')
            images.append(buffer.getvalue())
        padded_size = max(map(len, images))
        first_path = None
        for index, contents in enumerate(images):
            with open(image_path, 'wb') as image_file:
                image_file.write(contents.ljust(padded_size, b'\0'))
            timestamp = 1_700_000_000_000_000_000 + (index + 1) * 100_000_000
            os.utime(image_path, ns=(timestamp, timestamp))
            sd = FakeSD()
            dataset = AiToolkitDataset(self._dataset_config(), batch_size=1, sd=sd)
            path = dataset.file_list[0].get_latent_path()
            self.assertEqual(sd.encode_calls, 1)
            if first_path is None:
                first_path = path
            else:
                self.assertNotEqual(first_path, path)
                self.assertTrue(os.path.exists(first_path))

    def test_changed_model_identity_rebuilds_latents(self):
        _write_image(os.path.join(self.temp_dir, 'a.png'))
        first = AiToolkitDataset(self._dataset_config(), batch_size=1, sd=FakeSD())
        sd = FakeSD()
        sd.latent_space_version = 'sd1_new_vae'
        second = AiToolkitDataset(self._dataset_config(), batch_size=1, sd=sd)
        self.assertNotEqual(first.file_list[0].get_latent_path(), second.file_list[0].get_latent_path())
        self.assertEqual(sd.encode_calls, 1)

    def test_failed_samples_remap_buckets_without_changing_cached_geometry(self):
        original_encode = dataloader_mixins.LatentCachingMixin._encode_latent_for_file_item
        for failed_name in ('a.png', 'b.png', 'c.png'):
            with self.subTest(failed_name=failed_name), tempfile.TemporaryDirectory(dir=self.temp_dir) as folder:
                for name, size in [('a.png', (180, 96)), ('b.png', (180, 96)), ('c.png', (96, 180))]:
                    _write_image(os.path.join(folder, name), size=size)
                encoded_geometry = {}

                def encode(dataset, item, latent_path, to_disk):
                    if os.path.basename(item.path) == failed_name:
                        raise ValueError('Unusable training sample')
                    encoded_geometry[item.path] = (
                        item.crop_x, item.crop_y, item.crop_width, item.crop_height, latent_path,
                    )
                    return original_encode(dataset, item, latent_path, to_disk)

                with mock.patch.object(dataloader_mixins.LatentCachingMixin, '_encode_latent_for_file_item', encode):
                    dataset = AiToolkitDataset(
                        self._dataset_config(dataset_path=folder, num_repeats=2, random_crop=True, resolution=256),
                        batch_size=3, sd=FakeSD(),
                    )
                self.assertEqual(len(dataset.file_list), 4)
                self.assertEqual(len(dataset.buckets), 1 if failed_name == 'c.png' else 2)
                self.assertTrue(all(bucket.file_list_idx for bucket in dataset.buckets.values()))
                for item in dataset.file_list:
                    self.assertEqual(
                        (item.crop_x, item.crop_y, item.crop_width, item.crop_height, item.get_latent_path()),
                        encoded_geometry[item.path],
                    )
                for index in range(len(dataset)):
                    batch = dataset[index]
                    self.assertEqual(len(batch), 3)
                    self.assertEqual(len({tuple(item.get_latent().shape) for item in batch}), 1)
                    self.assertTrue(all(os.path.basename(item.path) != failed_name for item in batch))

    def test_all_failed_samples_report_clear_error(self):
        _write_image(os.path.join(self.temp_dir, 'a.png'))
        with mock.patch.object(FakeSD, 'encode_images', side_effect=ValueError('Unusable sample')):
            with self.assertRaisesRegex(ValueError, 'No usable samples remain after caching latents'):
                AiToolkitDataset(self._dataset_config(), batch_size=1, sd=FakeSD())

    def test_corrupt_or_incomplete_caches_are_rebuilt_for_disk_and_memory(self):
        for to_memory in (False, True):
            for corruption in ('truncated', 'missing_latent'):
                with self.subTest(to_memory=to_memory, corruption=corruption), tempfile.TemporaryDirectory(dir=self.temp_dir) as folder:
                    _write_image(os.path.join(folder, 'a.png'))
                    config = self._dataset_config(dataset_path=folder, cache_latents=to_memory)
                    first = AiToolkitDataset(config, batch_size=1, sd=FakeSD())
                    path = first.file_list[0].get_latent_path()
                    if corruption == 'truncated':
                        with open(path, 'wb') as cache_file:
                            cache_file.write(b'partial')
                    else:
                        dataloader_mixins.save_file({'unrelated': torch.ones(1)}, path)
                    sd = FakeSD()
                    recovered = AiToolkitDataset(config, batch_size=1, sd=sd)
                    self.assertEqual(sd.encode_calls, 1)
                    self.assertTrue(recovered.file_list[0].is_latent_cached)
                    self.assertTrue(torch.equal(recovered.file_list[0].get_latent(), torch.ones(4, 8, 8)))

    def test_failed_write_preserves_existing_cache_and_removes_temporary_file(self):
        _write_image(os.path.join(self.temp_dir, 'a.png'))
        dataset = AiToolkitDataset(self._dataset_config(cache_latents_to_disk=False), batch_size=1, sd=FakeSD())
        item = dataset.file_list[0]
        path = item.get_latent_path()
        os.makedirs(os.path.dirname(path))
        original_latent = torch.full((1,), -7.0)
        dataloader_mixins.save_file({'latent': original_latent}, path)

        def interrupted_save(state, temporary_path, **kwargs):
            self.assertNotEqual(temporary_path, path)
            with open(temporary_path, 'wb') as temporary_file:
                temporary_file.write(b'partial')
            raise OSError('Interrupted cache write')

        with mock.patch.object(dataloader_mixins, 'save_file', side_effect=interrupted_save):
            with self.assertRaisesRegex(OSError, 'Interrupted cache write'):
                dataset._encode_latent_for_file_item(item, path, True)
        self.assertTrue(torch.equal(dataloader_mixins.load_file(path)['latent'], original_latent))
        self.assertEqual(os.listdir(os.path.dirname(path)), [os.path.basename(path)])

    def test_concurrent_writers_publish_only_complete_cache_files(self):
        _write_image(os.path.join(self.temp_dir, 'a.png'))
        datasets = [AiToolkitDataset(self._dataset_config(cache_latents_to_disk=False), batch_size=1, sd=FakeSD()) for _ in range(2)]
        path = datasets[0].file_list[0].get_latent_path()
        save_file = dataloader_mixins.save_file
        barrier = threading.Barrier(2, timeout=10)
        temporary_paths = []

        def overlapping_save(state, temporary_path, **kwargs):
            temporary_paths.append(temporary_path)
            barrier.wait()
            save_file(state, temporary_path, **kwargs)
            self.assertFalse(os.path.exists(path))
            barrier.wait()

        with mock.patch.object(dataloader_mixins, 'save_file', side_effect=overlapping_save):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(dataset._encode_latent_for_file_item, dataset.file_list[0], path, True) for dataset in datasets]
                for future in futures:
                    future.result(timeout=15)
        self.assertEqual(len(set(temporary_paths)), 2)
        self.assertIn('latent', dataloader_mixins.load_file(path))
        self.assertEqual(os.listdir(os.path.dirname(path)), [os.path.basename(path)])

    def test_losing_publisher_reuses_winner_for_memory_and_disk(self):
        _write_image(os.path.join(self.temp_dir, 'a.png'))
        dataset = AiToolkitDataset(self._dataset_config(cache_latents_to_disk=False), batch_size=1, sd=FakeSD())
        item = dataset.file_list[0]
        path = item.get_latent_path()
        winner = torch.full((4, 8, 8), 7.0)

        def competing_replace(temporary_path, target_path):
            dataloader_mixins.save_file({'latent': winner}, target_path)
            raise PermissionError('Concurrent reader holds destination open')

        with mock.patch.object(dataloader_mixins.os, 'replace', side_effect=competing_replace):
            state = dataset._encode_latent_for_file_item(item, path, True)
        self.assertTrue(torch.equal(state['latent'], winner))
        self.assertTrue(torch.equal(dataloader_mixins.load_file(path)['latent'], winner))
        self.assertEqual(os.listdir(os.path.dirname(path)), [os.path.basename(path)])


if __name__ == "__main__":
    unittest.main()
