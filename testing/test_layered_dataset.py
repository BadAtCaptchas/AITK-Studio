"""Real PSD/ORA conversion and grouped loader contracts; no model weights needed."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from toolkit.layered_dataset import (
    LayeredDatasetError, discover_layered_samples, format_layered_caption,
    load_layered_sample, load_layered_tensor,
)
from toolkit.layered_import import import_layered_document


def png(color, size=(2, 2)):
    output = io.BytesIO()
    Image.new("RGBA", size, color).save(output, format="PNG")
    return output.getvalue()


def ora(path, xml=None, extra=None):
    xml = xml or '''<image version="0.0.6" w="8" h="8"><stack>
      <layer name="hidden" src="data/blue.png" visibility="hidden"/>
      <stack name="foreground" opacity="0.5"><layer name="red" src="data/red.png" x="2" y="1"/></stack>
      <layer name="background" src="data/blue.png" x="-1" y="0"/>
    </stack></image>'''
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "image/openraster", compress_type=zipfile.ZIP_STORED)
        archive.writestr("stack.xml", xml)
        archive.writestr("data/red.png", png((255, 0, 0, 128)))
        archive.writestr("data/blue.png", png((0, 0, 255, 255)))
        for name, contents in (extra or {}).items():
            archive.writestr(name, contents)
    return path


class LayeredTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.dataset = self.base / "dataset"
        self.dataset.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def import_ora(self, **kwargs):
        source = ora(self.base / "sample.ora", **kwargs)
        result = import_layered_document(source, self.dataset)
        return result, load_layered_sample(self.dataset, self.dataset / result["manifest"])


class LayeredImportTests(LayeredTestCase):
    def test_ora_order_groups_offsets_opacity_and_composite(self):
        result, sample = self.import_ora()
        self.assertEqual(result["layer_count"], 2)
        self.assertEqual([row["name"] for row in sample.manifest["layers"]], ["background", "foreground"])
        self.assertEqual(len(result["warnings"]), 1)
        self.assertEqual([row["caption"] for row in sample.manifest["layers"]], ["", ""])
        composite, bottom, top = [Image.open(path).copy() for path in sample.image_paths()]
        self.assertEqual(bottom.size, (8, 8))
        self.assertEqual(bottom.getpixel((0, 0)), (0, 0, 255, 255))
        self.assertEqual(bottom.getpixel((1, 0)), (0, 0, 0, 0))
        self.assertEqual(top.getpixel((2, 1)), (255, 0, 0, 64))
        np.testing.assert_array_equal(np.array(composite), np.array(Image.alpha_composite(bottom, top)))
        self.assertEqual(sample.manifest["source"]["filename"], "sample.ora")
        self.assertEqual(len(discover_layered_samples(self.dataset)), 1)

    def test_psd_raster_groups_mask_opacity_and_order(self):
        from psd_tools import PSDImage
        from psd_tools.constants import BlendMode

        document = PSDImage.new(mode="RGB", size=(8, 8), depth=8)
        document.create_pixel_layer(Image.new("RGBA", (8, 8), (0, 0, 255, 255)), name="background")
        group = document.create_group(name="foreground", blend_mode=BlendMode.NORMAL)
        pixel = document.create_pixel_layer(Image.new("RGB", (2, 2), "red"), name="red", top=1, left=2)
        group.append(pixel)
        pixel.create_mask(Image.new("L", (2, 2), 128), top=1, left=2)
        group.opacity = 128
        invisible = document.create_pixel_layer(Image.new("RGBA", (1, 1), (0, 255, 0, 255)), name="hidden")
        invisible.visible = False
        source = self.base / "document.psd"
        document.save(source)
        result = import_layered_document(source, self.dataset)
        sample = load_layered_sample(self.dataset, self.dataset / result["manifest"])
        self.assertEqual([row["name"] for row in sample.manifest["layers"]], ["background", "foreground"])
        with Image.open(sample.image_paths()[2]) as foreground:
            self.assertAlmostEqual(foreground.getpixel((2, 1))[3], 64, delta=1)
            self.assertEqual(foreground.getpixel((0, 0))[3], 0)

    def test_rejects_psd_blend_and_clipping(self):
        from psd_tools import PSDImage
        from psd_tools.constants import BlendMode
        for feature in ("blend", "clipping"):
            with self.subTest(feature=feature):
                document = PSDImage.new(mode="RGB", size=(8, 8), depth=8)
                layer = document.create_pixel_layer(Image.new("RGBA", (2, 2), "red"), name="unsupported")
                if feature == "blend":
                    layer.blend_mode = BlendMode.MULTIPLY
                else:
                    layer.clipping = True
                source = self.base / f"{feature}.psd"
                document.save(source)
                with self.assertRaisesRegex(LayeredDatasetError, "Unsupported"):
                    import_layered_document(source, self.dataset)
        self.assertEqual(list(self.dataset.iterdir()), [])

    def test_rejects_traversal_entities_blends_and_nonfinite_opacity(self):
        bad_xml = [
            '<image w="8" h="8"><stack><layer src="../outside.png"/></stack></image>',
            '<!DOCTYPE image [<!ENTITY x "expanded">]><image w="8" h="8"><stack name="&x;"/></image>',
            '<image w="8" h="8"><stack><layer src="data/red.png" composite-op="svg:multiply"/></stack></image>',
            '<image w="8" h="8"><stack><layer src="data/red.png" opacity="nan"/></stack></image>',
            '<image w="999999999" h="8"><stack/></image>',
            '<image w="8" h="8"><stack><stack x="12"><layer src="data/red.png"/></stack></stack></image>',
            '<image w="8" h="8"><stack><layer src="data/red.png" mask="unhandled.png"/></stack></image>',
        ]
        for xml in bad_xml:
            with self.subTest(xml=xml), self.assertRaises(Exception):
                import_layered_document(ora(self.base / "bad.ora", xml=xml), self.dataset)
        with self.assertRaisesRegex(LayeredDatasetError, "Unsafe"):
            import_layered_document(ora(self.base / "bad.ora", extra={"../escape": b"x"}), self.dataset)
        self.assertEqual(list(self.dataset.iterdir()), [])

    def test_layer_and_decode_budgets_fail_without_publishing(self):
        xml = '<image w="8" h="8"><stack>' + '<layer src="data/red.png"/>' * 33 + '</stack></image>'
        with self.assertRaisesRegex(LayeredDatasetError, "32 exported"):
            import_layered_document(ora(self.base / "many.ora", xml), self.dataset)
        with patch("toolkit.layered_import.MAX_DECODED_BYTES", 16):
            with self.assertRaisesRegex(LayeredDatasetError, "pixel|decoded"):
                self.import_ora()
        self.assertEqual(list(self.dataset.iterdir()), [])

    def test_encrypted_target_and_invalid_manifest_are_rejected(self):
        (self.dataset / ".aitk_encrypted_dataset.json").write_text("{}")
        with self.assertRaisesRegex(LayeredDatasetError, "encrypted"):
            self.import_ora()
        (self.dataset / ".aitk_encrypted_dataset.json").unlink()
        _, sample = self.import_ora()
        bad = copy.deepcopy(sample.manifest)
        bad["layers"][0]["path"] = "../outside.png"
        sample.manifest_path.write_text(json.dumps(bad))
        with self.assertRaises(LayeredDatasetError):
            load_layered_sample(self.dataset, sample.manifest_path)

    def test_atomic_failure_cleans_owned_artifacts_and_collision_preserves_user_file(self):
        source = ora(self.base / "sample.ora")
        fixed = SimpleNamespace(hex="12345678" + "0" * 24)
        existing = self.dataset / "sample_12345678.png"
        existing.write_bytes(b"user-owned")
        with patch("toolkit.layered_import.uuid.uuid4", return_value=fixed):
            with self.assertRaises(FileExistsError):
                import_layered_document(source, self.dataset)
        self.assertEqual(existing.read_bytes(), b"user-owned")
        self.assertFalse((self.dataset / "sample_12345678.txt").exists())
        self.assertEqual(list((self.dataset / ".layers").iterdir()), [])
        self.assertFalse(any(path.name.startswith(".layered-import-") for path in self.dataset.iterdir()))

    def test_manual_manifest_allows_omitted_source_and_layer_captions(self):
        _, sample = self.import_ora()
        manual = copy.deepcopy(sample.manifest)
        del manual['source']
        for layer in manual['layers']:
            del layer['caption']
        sample.manifest_path.write_text(json.dumps(manual))
        loaded = load_layered_sample(self.dataset, sample.manifest_path)
        self.assertEqual([layer['caption'] for layer in loaded.manifest['layers']], ['', ''])
        self.assertEqual(format_layered_caption(loaded, 'Poster'), 'Decompose the image into 2 layers.\nPoster')
        for source in (None, {}, {'format': 'ora', 'filename': 'manual.ora', 'sha256': 'A' * 64}):
            with self.subTest(source=source):
                manual['source'] = source
                sample.manifest_path.write_text(json.dumps(manual))
                with self.assertRaisesRegex(LayeredDatasetError, 'source metadata'):
                    load_layered_sample(self.dataset, sample.manifest_path)

    def test_manifest_limits_match_ui_and_import_preserves_names_or_rejects(self):
        from toolkit.layered_dataset import contained_path, validate_canvas
        _, sample = self.import_ora()
        for field, value in (('name', 'x' * 513), ('name', '\U0001f600' * 257), ('caption', 'x' * 65_537), ('caption', None)):
            with self.subTest(field=field, size=len(value) if isinstance(value, str) else None):
                invalid = copy.deepcopy(sample.manifest)
                invalid['layers'][0][field] = value
                sample.manifest_path.write_text(json.dumps(invalid))
                with self.assertRaisesRegex(LayeredDatasetError, 'target metadata'):
                    load_layered_sample(self.dataset, sample.manifest_path)
        with self.assertRaisesRegex(LayeredDatasetError, '32 megapixel'):
            validate_canvas(8000, 4001)
        for relative in ('x' * 1024, 'bad\x01.png'):
            with self.assertRaisesRegex(LayeredDatasetError, 'Invalid layered sample path'):
                contained_path(self.dataset, relative, must_exist=False)
        target = self.base / 'long-names'
        target.mkdir()
        source = ora(self.base / 'long.ora', '<image w="8" h="8"><stack><layer src="data/red.png" name="' + 'x' * 513 + '"/></stack></image>')
        with self.assertRaisesRegex(LayeredDatasetError, 'Layer name exceeds'):
            import_layered_document(source, target)
        self.assertEqual(list(target.iterdir()), [])

    def test_layer_change_invalidates_signature_and_caption_uses_actual_count(self):
        _, sample = self.import_ora()
        before = sample.signature()
        sample.manifest["layers"][0]["caption"] = "Blue background"
        self.assertIn("2 layers", format_layered_caption(sample, "Poster"))
        self.assertIn("Layer 1: Blue background", format_layered_caption(sample, "Poster"))
        self.assertNotEqual(before, sample.signature())
        before = sample.signature()
        Image.new("RGBA", (8, 8), "green").save(sample.image_paths()[1])
        self.assertNotEqual(before, sample.signature())

    def test_aligned_rgba_geometry_and_control_range(self):
        import torch
        _, sample = self.import_ora()
        item = SimpleNamespace(layered_sample=sample, flip_x=True, flip_y=True,
            dataset_config=SimpleNamespace(buckets=True), scale_to_width=16, scale_to_height=16,
            crop_x=2, crop_y=4, crop_width=8, crop_height=8)
        result = load_layered_tensor(item)
        self.assertEqual(tuple(result.shape), (3, 4, 8, 8))
        torch.testing.assert_close(item.control_tensor, (result[0] + 1) / 2)
        full_control = item.control_tensor.clone()
        load_layered_tensor(item, composite_only=True)
        torch.testing.assert_close(item.control_tensor, full_control)


class LayeredLoaderTests(LayeredTestCase):
    def model(self):
        return SimpleNamespace(supports_layered_images=True, preserve_image_alpha=True,
            encode_control_in_text_embeddings=True, te_padding_side="right", use_raw_control_images=False,
            vae=SimpleNamespace(config=SimpleNamespace()), unet=SimpleNamespace(config=SimpleNamespace()),
            get_bucket_divisibility=lambda: 8, get_latent_space_version=lambda: "test-layers",
            get_text_embedding_space_version=lambda: "test-caption")

    def test_real_dataset_buckets_by_count_and_excludes_assets_for_ordinary_images(self):
        from toolkit.config_modules import DatasetConfig
        from toolkit.data_loader import AiToolkitDataset
        from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
        result, _ = self.import_ora()
        source = ora(self.base / "one.ora", '<image w="8" h="8"><stack><layer src="data/red.png"/></stack></image>')
        import_layered_document(source, self.dataset)
        config = DatasetConfig(type="layered_image", folder_path=str(self.dataset), resolution=8, buckets=True)
        dataset = AiToolkitDataset(config, batch_size=2, sd=self.model())
        self.assertEqual(sorted(item.num_frames for item in dataset.file_list), [2, 3])
        self.assertEqual(len(dataset.buckets), 2)
        item = dataset._get_single_item(next(i for i, x in enumerate(dataset.file_list) if Path(x.path).name == result["composite"]))
        batch = DataLoaderBatchDTO(file_items=[item])
        self.assertEqual(tuple(batch.tensor.shape), (1, 3, 4, 8, 8))
        self.assertEqual(tuple(batch.control_tensor.shape), (1, 4, 8, 8))
        self.assertIn("2 layers", item.caption)
        self.assertIn("layered_signature", item.get_latent_info_dict())
        self.assertIn("layered_geometry", item.get_text_embedding_info_dict())
        ordinary = AiToolkitDataset(DatasetConfig(folder_path=str(self.dataset), resolution=8), sd=self.model())
        self.assertEqual(len(ordinary.file_list), 2)

    def test_cached_conditioning_rejects_random_geometry(self):
        from toolkit.config_modules import DatasetConfig
        from toolkit.data_loader import AiToolkitDataset
        self.import_ora()
        config = DatasetConfig(type="layered_image", folder_path=str(self.dataset), resolution=8,
                               random_crop=True, cache_text_embeddings=True)
        with self.assertRaisesRegex(ValueError, "fixed geometry"):
            AiToolkitDataset(config, sd=self.model())

    def test_real_latent_and_condition_cache_reuse_and_layer_invalidation(self):
        import torch
        from toolkit.config_modules import DatasetConfig
        from toolkit.data_loader import AiToolkitDataset
        from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
        from toolkit.prompt_utils import PromptEmbeds

        _, sample = self.import_ora()
        encoded_images, encoded_controls = [], []
        model = self.model()
        model.device = 'cpu'
        model.device_torch = torch.device('cpu')
        model.torch_dtype = torch.float32
        model.has_multiple_control_images = False
        model.set_device_state_preset = lambda preset: None
        model.restore_device_state = lambda: None

        def encode_images(images):
            encoded_images.append(images.clone())
            # Real loader/caching with a cheap encoder, preserving native C,F axes.
            return images.permute(0, 2, 1, 3, 4).repeat(1, 4, 1, 1, 1).contiguous()

        def encode_prompt(caption, control_images=None):
            encoded_controls.append(control_images.clone())
            return PromptEmbeds(torch.zeros(1, 2, 4))

        model.encode_images = encode_images
        model.encode_prompt = encode_prompt

        def create():
            return AiToolkitDataset(DatasetConfig(type='layered_image', folder_path=str(self.dataset),
                resolution=8, cache_latents_to_disk=True, cache_text_embeddings=True), sd=model)

        dataset = create()
        self.assertEqual(len(encoded_images), 1)
        self.assertEqual(tuple(encoded_images[0].shape), (1, 3, 4, 8, 8))
        torch.testing.assert_close(encoded_controls[0][0], (encoded_images[0][0, 0] + 1) / 2)
        batch = DataLoaderBatchDTO(file_items=[dataset._get_single_item(0)])
        self.assertEqual(tuple(batch.latents.shape), (1, 16, 3, 8, 8))
        torch.testing.assert_close(batch.control_tensor, encoded_controls[0])
        cached = create()
        self.assertEqual(len(encoded_images), 1)
        self.assertEqual(len(encoded_controls), 1)
        cached_batch = DataLoaderBatchDTO(file_items=[cached._get_single_item(0)])
        torch.testing.assert_close(cached_batch.latents, batch.latents)
        Image.new('RGBA', (8, 8), 'green').save(sample.image_paths()[1])
        create()
        self.assertEqual(len(encoded_images), 2)
        self.assertEqual(len(encoded_controls), 2)


if __name__ == "__main__":
    unittest.main()
