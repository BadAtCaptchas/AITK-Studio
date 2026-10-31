import random
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset
from toolkit.latent_cache_preparation import (
    MAX_STAGED_PIXELS, LatentImageStager, can_stage_latent_image,
    decode_latent_image, mark_cached_survivors,
)
from testing.test_latent_cache_reuse import FakeSD


class LatentPreparationTest(unittest.TestCase):
    def test_finalization_uses_survivor_identity_not_equality_or_paths(self):
        class Item:
            def __eq__(self, _other):
                raise AssertionError('Survivors must never be compared by equality')
        survivor, failed = Item(), Item()
        survivor.path = failed.path = 'duplicate.png'
        failed.is_latent_cached = False
        mark_cached_survivors([survivor, survivor])
        self.assertTrue(survivor.is_latent_cached)
        self.assertFalse(failed.is_latent_cached)
        mark_cached_survivors([])

    def test_specialized_inputs_and_disabled_staging_stay_serial(self):
        for flag in ('is_video', 'is_audio_model', 'is_layered', 'is_encrypted',
                     'has_control_image', 'has_inpaint_image', 'has_clip_image',
                     'has_mask_image', 'has_unconditional', 'is_text_embedding_cached'):
            self.assertFalse(can_stage_latent_image(SimpleNamespace(**{flag: True})))
        # D-OPSD self references do not set has_control_image. Also reject
        # explicitly attached paths even if an adapter has not set that flag.
        for reference in ({'dopsd_self_ref': True}, {'control_path': 'ref.png'},
                          {'control_path': ['ref.png']}, {'control_video_paths': ['ref.mp4']}):
            self.assertFalse(can_stage_latent_image(SimpleNamespace(has_control_image=False, **reference)))
        self.assertTrue(can_stage_latent_image(SimpleNamespace(
            has_control_image=False, dopsd_self_ref=False, control_path=None, control_video_paths=[])))
        with mock.patch('toolkit.latent_cache_preparation.decode_latent_image') as decode:
            with LatentImageStager(False) as stager:
                stager.submit(SimpleNamespace())
                self.assertIsNone(stager.take_and_stage(None))
            decode.assert_not_called()

    def test_bound_uses_opened_dimensions_and_never_decodes_large_source(self):
        image = mock.MagicMock()
        image.__enter__.return_value = image
        image.width, image.height, image.mode = MAX_STAGED_PIXELS + 1, 1, 'RGB'
        with mock.patch('toolkit.latent_cache_preparation.Image.open', return_value=image), \
             mock.patch('toolkit.latent_cache_preparation._finalize_static_image') as finalize:
            item = SimpleNamespace(path='image.png', width=1, height=1)
            self.assertIsNone(decode_latent_image(item))
            finalize.assert_not_called()

    def test_one_decode_ahead_and_exit_closes_unused_image(self):
        owner = threading.get_ident()
        decoded_threads = []
        image = mock.Mock()
        def decode(_item):
            decoded_threads.append(threading.get_ident())
            return image
        with mock.patch('toolkit.latent_cache_preparation.decode_latent_image', side_effect=decode):
            with self.assertRaisesRegex(RuntimeError, 'cancel'):
                with LatentImageStager() as stager:
                    stager.submit(SimpleNamespace())
                    stager.pending.result()
                    raise RuntimeError('cancel')
        self.assertEqual(len(decoded_threads), 1)
        self.assertNotEqual(decoded_threads[0], owner)
        image.close.assert_called_once()

    def test_failed_decode_advances_next_item_without_shifting_identity(self):
        good = mock.Mock()
        with mock.patch('toolkit.latent_cache_preparation.decode_latent_image', side_effect=[ValueError('bad'), good]):
            with LatentImageStager() as stager:
                stager.submit(SimpleNamespace())
                with self.assertRaisesRegex(ValueError, 'bad'):
                    stager.take_and_stage(SimpleNamespace())
                self.assertIs(stager.take_and_stage(None), good)
            good.close()  # consumed images belong to the owner

    def test_speculative_failure_retries_serial_decode_without_filtering_repaired_items(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in range(2):
                Image.new('RGB', (96, 96)).save(Path(directory) / f'{index}.png')
            encoder = FakeSD()
            with mock.patch('toolkit.latent_cache_preparation.decode_latent_image', side_effect=OSError('transient speculative read')):
                dataset = AiToolkitDataset(DatasetConfig(
                    folder_path=directory, resolution=64, buckets=True,
                    cache_latents_to_disk=True, num_repeats=2,
                ), batch_size=1, sd=encoder)
            self.assertEqual(encoder.encode_calls, 2)
            self.assertEqual(len(dataset.file_list), 4)
            self.assertTrue(all(item.is_latent_cached for item in dataset.file_list))

    def test_concurrent_publication_wins_even_when_speculative_decode_fails(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            for index in range(2):
                Image.new('RGB', (96, 96)).save(Path(directory) / f'{index}.png')
            def publish_then_fail(item):
                path = Path(item.get_latent_path())
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file({'latent': torch.full((4, 8, 8), 123.0)}, str(path))
                raise ValueError('source changed after another writer published')
            encoder = FakeSD()
            with mock.patch('toolkit.latent_cache_preparation.decode_latent_image', side_effect=publish_then_fail):
                dataset = AiToolkitDataset(DatasetConfig(
                    folder_path=directory, resolution=64, buckets=True,
                    cache_latents_to_disk=True, num_repeats=2,
                ), batch_size=1, sd=encoder)
            self.assertEqual(encoder.encode_calls, 0)
            self.assertEqual(len(dataset.file_list), 4)
            for item in dataset.file_list:
                self.assertTrue(item.is_latent_cached)
                torch.testing.assert_close(item.get_latent(), torch.full((4, 8, 8), 123.0))

    def test_malformed_concurrent_winner_is_reencoded_not_accepted_or_skipped(self):
        from safetensors.torch import save_file, load_file
        with tempfile.TemporaryDirectory() as directory:
            for index in range(2):
                Image.new('RGB', (96, 96)).save(Path(directory) / f'{index}.png')
            def publish_malformed(item):
                path = Path(item.get_latent_path())
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file({'latent': torch.full((4, 8, 8), 123.0),
                           'waveform_sample_rate': torch.tensor([1, 2])}, str(path))
                return None
            encoder = FakeSD()
            with mock.patch('toolkit.latent_cache_preparation.decode_latent_image', side_effect=publish_malformed):
                dataset = AiToolkitDataset(DatasetConfig(
                    folder_path=directory, resolution=64, buckets=True,
                    cache_latents_to_disk=True,
                ), batch_size=1, sd=encoder)
            self.assertEqual(encoder.encode_calls, 2)
            self.assertEqual(len(dataset.file_list), 2)
            for item in dataset.file_list:
                self.assertTrue(item.is_latent_cached)
                self.assertNotIn('waveform_sample_rate', load_file(item.get_latent_path()))

    def test_real_decode_crop_and_rng_match_serial_and_encoder_stays_on_owner(self):
        owner = threading.get_ident()
        class CaptureSD(FakeSD):
            def __init__(self):
                super().__init__()
                self.inputs = []
                self.random_values = []
            def encode_images(self, imgs):
                self.assert_owner = threading.get_ident() == owner
                self.inputs.append(imgs.clone())
                self.random_values.append((random.random(), np.random.rand(), torch.rand(()).item()))
                return super().encode_images(imgs)
        with tempfile.TemporaryDirectory() as directory:
            rng = np.random.default_rng(42)
            for index in range(4):
                Image.fromarray(rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)).save(Path(directory) / f'{index}.png')
            runs = []
            for staged in (False, True):
                random.seed(17)
                np.random.seed(17)
                torch.manual_seed(17)
                encoder = CaptureSD()
                dataset = AiToolkitDataset(DatasetConfig(
                    folder_path=directory, resolution=64, buckets=True,
                    random_crop=True, cache_latents=True, num_repeats=2,
                    latent_decode_prefetch=staged,
                ), batch_size=1, sd=encoder)
                self.assertTrue(encoder.assert_owner)
                self.assertTrue(all(item.is_latent_cached for item in dataset.file_list))
                self.assertTrue(all(not hasattr(item, '_latent_predecoded_image') for item in dataset.file_list))
                runs.append((encoder.inputs, encoder.random_values, [item.get_latent_path() for item in dataset.file_list]))
            self.assertEqual(runs[0][1:], runs[1][1:])
            self.assertEqual(len(runs[0][0]), len(runs[1][0]))
            for serial, staged in zip(runs[0][0], runs[1][0]):
                torch.testing.assert_close(serial, staged, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
