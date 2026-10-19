"""Exercise cache invalidation through real dataset loading and disk reuse."""
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
from PIL import Image

from testing.test_latent_cache_reuse import FakeSD
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset, get_dataloader_from_datasets
from toolkit.prompt_utils import PromptEmbeds


class PixelEncoder(FakeSD):
    def __init__(self, control=False):
        super().__init__()
        self.is_vega = False
        self.is_ssd = False
        self.encode_control_in_text_embeddings = control
        self.has_multiple_control_images = False
        self.text_embedding_space_version = 'qwen_image_edit' if control else 'sd1'
        self.model_config.arch = self.text_embedding_space_version
        self.text_calls = 0

    def encode_prompt(self, caption, control_images=None, **kwargs):
        self.text_calls += 1
        value = float(control_images.mean()) if control_images is not None else len(caption)
        return PromptEmbeds(torch.full((1, 2, 4), value))

    def encode_images(self, images):
        self.encode_calls += 1
        return torch.full(
            (images.shape[0], 4, images.shape[-2] // 8, images.shape[-1] // 8),
            float(images.mean()),
        )


class DatasetCacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.target = self.root / 'targets' / 'sample.png'
        self.reference = self.root / 'references' / 'sample.png'
        self.write_image(self.target, (96, 128, 160))
        self.target.with_suffix('.txt').write_text('A reference image', encoding='utf-8')

    def write_image(self, path, color):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Fixed-size encoding lets replacement retain the size and timestamp.
        Image.new('RGB', (64, 64), color).save(path, compress_level=0)

    def reference_loader(self):
        encoder = PixelEncoder(control=True)
        config = DatasetConfig(
            folder_path=str(self.target.parent), control_path=str(self.reference.parent),
            resolution=64, cache_text_embeddings=True, caption_dropout_rate=0,
        )
        loader = get_dataloader_from_datasets([config], sd=encoder)
        return encoder, loader, loader.dataset.datasets[0].file_list[0]

    def latent_dataset(self, standardize, cache=True):
        encoder = PixelEncoder()
        config = DatasetConfig(
            folder_path=str(self.target.parent), resolution=64, caption_dropout_rate=0,
            cache_latents_to_disk=cache, standardize_images=standardize,
        )
        return encoder, AiToolkitDataset(config, sd=encoder)

    def test_reference_replacement_reencodes_even_with_identical_file_stats(self):
        self.write_image(self.reference, (0, 0, 0))
        first_encoder, first, first_item = self.reference_loader()
        before = next(iter(first)).prompt_embeds.text_embeds.clone()
        first_path = first_item.get_text_embedding_path()
        dropout_path = first_item._build_text_embedding_path(text_only=True)
        self.assertEqual(first_encoder.text_calls, 1)

        reused_encoder, reused, _ = self.reference_loader()
        self.assertEqual(reused_encoder.text_calls, 0)
        torch.testing.assert_close(next(iter(reused)).prompt_embeds.text_embeds, before)

        original = self.reference.stat()
        self.write_image(self.reference, (255, 255, 255))
        os.utime(self.reference, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.assertEqual(self.reference.stat().st_size, original.st_size)
        self.assertEqual(self.reference.stat().st_mtime_ns, original.st_mtime_ns)
        second_encoder, second, second_item = self.reference_loader()
        after = next(iter(second)).prompt_embeds.text_embeds.clone()
        self.assertEqual(second_encoder.text_calls, 1)
        self.assertNotEqual(first_path, second_item.get_text_embedding_path())
        self.assertEqual(dropout_path, second_item._build_text_embedding_path(text_only=True))
        torch.testing.assert_close(before, torch.zeros_like(before))
        torch.testing.assert_close(after, torch.ones_like(after))
        self.assertTrue(Path(first_path).is_file())

    def test_multiple_references_track_each_file_and_their_order(self):
        self.write_image(self.reference, (0, 0, 0))
        _, _, item = self.reference_loader()
        other = self.reference.with_name('other.png')
        self.write_image(other, (255, 255, 255))
        item.control_path = [str(self.reference), str(other)]
        first = item.get_text_embedding_info_dict()
        item.control_path.reverse()
        reordered = item.get_text_embedding_info_dict()
        self.assertEqual(first['control_reference_contents'], list(reversed(reordered['control_reference_contents'])))
        self.assertNotEqual(first, reordered)
        self.write_image(other, (128, 128, 128))
        replaced = item.get_text_embedding_info_dict()
        self.assertNotEqual(reordered['control_reference_contents'][0], replaced['control_reference_contents'][0])
        self.assertEqual(reordered['control_reference_contents'][1], replaced['control_reference_contents'][1])
        self.assertNotIn('control_reference_contents', item.get_text_embedding_info_dict(text_only=True))

    def test_standardization_changes_latents_and_reuses_the_correct_variant(self):
        plain_encoder, plain = self.latent_dataset(False)
        plain_item = plain.file_list[0]
        plain_value = plain_item.get_latent().clone()
        standardized_encoder, standardized = self.latent_dataset(True)
        standardized_item = standardized.file_list[0]
        standardized_value = standardized_item.get_latent().clone()
        fresh_encoder, fresh = self.latent_dataset(True, cache=False)
        expected = fresh_encoder.encode_images(fresh._get_single_item(0).tensor.unsqueeze(0))

        self.assertEqual(plain_encoder.encode_calls, 1)
        self.assertEqual(standardized_encoder.encode_calls, 1)
        self.assertNotEqual(plain_item.get_latent_path(), standardized_item.get_latent_path())
        self.assertFalse(torch.allclose(plain_value, standardized_value))
        torch.testing.assert_close(standardized_value, expected.squeeze(0))
        for flag, value in ((False, plain_value), (True, standardized_value)):
            encoder, dataset = self.latent_dataset(flag)
            self.assertEqual(encoder.encode_calls, 0)
            torch.testing.assert_close(dataset.file_list[0].get_latent(), value)

    def test_ambiguous_legacy_latent_cache_is_not_reused(self):
        _, standardized = self.latent_dataset(True)
        item = standardized.file_list[0]
        # An old cache key did not disclose which pixel transform wrote it.
        info = item.get_latent_info_dict()
        del info['standardize_images']
        digest = hashlib.md5(json.dumps(info, sort_keys=True).encode('utf-8')).digest()
        suffix = base64.urlsafe_b64encode(digest).decode('ascii').replace('=', '')
        current_path = Path(item.get_latent_path())
        legacy_path = current_path.with_name(f'sample_{suffix}.safetensors')
        current_path.rename(legacy_path)
        for flag in (False, True):
            encoder, dataset = self.latent_dataset(flag)
            self.assertEqual(encoder.encode_calls, 1)
            self.assertNotEqual(str(legacy_path), dataset.file_list[0].get_latent_path())
        self.assertTrue(legacy_path.is_file())


if __name__ == '__main__':
    unittest.main()
