"""Caption and text-cache regressions using real dataset objects and temporary media."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from PIL import Image

from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import get_dataloader_from_datasets
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.prompt_utils import PromptEmbeds


class DatasetCaptionLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory(prefix="aitk-caption-regression-")
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "sample.png"
        Image.new("RGB", (64, 64)).save(self.path)
        self.caption = "a red flower"
        self.path.with_suffix(".txt").write_text(self.caption, encoding="utf-8")

    def config(self, **overrides):
        return DatasetConfig(
            folder_path=self.temp_dir.name,
            resolution=64,
            num_workers=0,
            **overrides,
        )

    def item(self, **overrides):
        return FileItemDTO(path=str(self.path), dataset_config=self.config(**overrides))

    def encoder(self):
        # Only model inference is synthetic; dataset construction, caption
        # processing, safetensors persistence, and batch loading are real.
        return SimpleNamespace(
            adapter=None,
            device="cpu",
            device_torch=torch.device("cpu"),
            torch_dtype=torch.float32,
            model_config=SimpleNamespace(arch="krea2", latent_space_version="issue87"),
            text_embedding_space_version="issue87",
            encode_control_in_text_embeddings=False,
            use_raw_control_images=False,
            te_padding_side="right",
            unet=SimpleNamespace(),
            vae=SimpleNamespace(),
            get_bucket_divisibility=lambda: 32,
            encode_prompt=Mock(side_effect=lambda caption: PromptEmbeds(
                torch.full((1, 2, 4), float(len(caption)))
            )),
            set_device_state_preset=Mock(),
            restore_device_state=Mock(),
        )

    def test_ordinary_caption_loads_with_and_without_embedding_cache(self):
        for cache in (False, True):
            with self.subTest(cache_text_embeddings=cache):
                item = self.item(cache_text_embeddings=cache)
                # This is the exact caption-loading entry point in issue #87.
                path = item.get_text_embedding_path(recalculate=True)
                self.assertEqual(item.caption, self.caption)
                self.assertIsNone(item.caption_dop)
                self.assertEqual(Path(path).parent, self.path.parent / "_t_e_cache")

    def test_preservation_caption_uses_configured_class(self):
        item = self.item(
            trigger_word="flower",
            diff_output_preservation=True,
            diff_output_preservation_class="plant",
        )
        item.load_caption()
        self.assertEqual(item.caption, self.caption)
        self.assertEqual(item.caption_dop, "a red plant")
        self.assertNotEqual(item.get_text_embedding_path(), item.get_dop_text_embedding_path())

    def test_empty_caption_override_has_its_own_cache_key(self):
        item = self.item(cache_text_embeddings=True)
        normal_path = item.get_text_embedding_path()
        self.assertNotEqual(normal_path, item.get_blank_text_embedding_path())
        self.assertEqual(item.get_text_embedding_info_dict(caption_override="")["caption"], "")
        self.assertEqual(
            item._build_text_embedding_path(caption_override=self.caption), normal_path
        )
        self.assertEqual(item.caption, self.caption)

    def test_dataset_caches_reloads_and_reuses_ordinary_embeddings(self):
        config = self.config(cache_text_embeddings=True)
        encoder = self.encoder()
        loader = get_dataloader_from_datasets([config], batch_size=1, sd=encoder)
        batch = next(iter(loader))
        encoder.encode_prompt.assert_called_once_with(self.caption)
        self.assertTrue(torch.equal(
            batch.prompt_embeds.text_embeds, torch.full((1, 2, 4), float(len(self.caption)))
        ))
        self.assertIsNone(batch.dop_prompt_embeds)

        encoder.encode_prompt.reset_mock()
        resumed = get_dataloader_from_datasets([config], batch_size=1, sd=encoder)
        resumed_batch = next(iter(resumed))
        encoder.encode_prompt.assert_not_called()
        self.assertTrue(torch.equal(
            resumed_batch.prompt_embeds.text_embeds, batch.prompt_embeds.text_embeds
        ))

    def test_cached_caption_dropout_loads_blank_embeddings(self):
        encoder = self.encoder()
        config = self.config(cache_text_embeddings=True, caption_dropout_rate=1.0)
        loader = get_dataloader_from_datasets([config], batch_size=1, sd=encoder)
        batch = next(iter(loader))
        self.assertEqual(
            [call.args[0] for call in encoder.encode_prompt.call_args_list],
            [self.caption, ""],
        )
        self.assertTrue(torch.equal(batch.prompt_embeds.text_embeds, torch.zeros(1, 2, 4)))

    def test_enabling_dropout_fills_missing_cache_without_reencoding_caption(self):
        encoder = self.encoder()
        config = self.config(cache_text_embeddings=True)
        loader = get_dataloader_from_datasets([config], batch_size=1, sd=encoder)
        next(iter(loader))
        encoder.encode_prompt.reset_mock()

        config.caption_dropout_rate = 1.0
        resumed = get_dataloader_from_datasets([config], batch_size=1, sd=encoder)
        batch = next(iter(resumed))
        encoder.encode_prompt.assert_called_once_with("")
        self.assertTrue(torch.equal(batch.prompt_embeds.text_embeds, torch.zeros(1, 2, 4)))


if __name__ == "__main__":
    unittest.main()
