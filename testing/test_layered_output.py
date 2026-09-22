"""Transactional sample publication keeps composites and layer assets together."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from toolkit.config_modules import GenerateImageConfig
from toolkit.layered_output import LayeredImageOutput


class LayeredOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = GenerateImageConfig(prompt='A poster', width=16, height=16,
                                         output_path=str(self.root / 'sample.png'))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def output(color='red'):
        return LayeredImageOutput(Image.new('RGBA', (16, 16), color),
                                  [Image.new('RGBA', (16, 16), (255, 0, 0, 128))])

    def fail_composite_publication(self):
        original = os.replace

        def replace(source, destination):
            if Path(destination) == self.root / 'sample.png':
                raise OSError('injected composite publication failure')
            return original(source, destination)

        return patch('toolkit.layered_output.os.replace', side_effect=replace)

    def test_success_publishes_full_canvas_alpha_and_relative_sidecar(self):
        self.config.save_image_atomic(self.output())
        sidecar = json.loads((self.root / 'sample.png.layers.json').read_text())
        self.assertEqual(sidecar['order'], 'bottom-to-top')
        self.assertEqual(len(sidecar['layers']), 1)
        with Image.open(self.root / sidecar['layers'][0]) as layer:
            self.assertEqual(layer.mode, 'RGBA')
            self.assertEqual(layer.getpixel((0, 0))[3], 128)
        self.assertTrue((self.root / '.thumbs' / 'sample.png.png').is_file())
        self.assertFalse(list((self.root / '.tmp').iterdir()))

    def test_failed_new_sample_removes_published_layers_sidecars_and_thumbnail(self):
        with self.fail_composite_publication(), self.assertRaisesRegex(OSError, 'injected'):
            self.config.save_image_atomic(self.output())
        self.assertFalse((self.root / 'sample.png').exists())
        self.assertFalse((self.root / 'sample.png.layers.json').exists())
        self.assertFalse(list((self.root / '.layers').iterdir()))
        self.assertFalse(list((self.root / '.thumbs').iterdir()))
        self.assertFalse(list((self.root / '.tmp').iterdir()))
        self.assertEqual(Path(self.config.output_folder), self.root)

    def test_failed_replacement_restores_previous_sample_and_owned_assets(self):
        self.config.save_image_atomic(self.output())
        previous = {path.relative_to(self.root): path.read_bytes()
                    for path in self.root.rglob('*') if path.is_file()}
        with self.fail_composite_publication(), self.assertRaisesRegex(OSError, 'injected'):
            self.config.save_image_atomic(self.output('blue'))
        current = {path.relative_to(self.root): path.read_bytes()
                   for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(current, previous)
        self.assertEqual(len(list((self.root / '.layers').iterdir())), 1)

    def test_thumbnail_failure_does_not_publish_partial_thumbnail_as_sidecar(self):
        def broken_thumbnail(media, thumbnail):
            Path(thumbnail).write_bytes(b'partial')
            raise OSError('thumbnail decode failed')

        with patch.object(self.config, '_generate_thumbnail', side_effect=broken_thumbnail):
            self.config.save_image_atomic(self.output())
        self.assertTrue((self.root / 'sample.png').exists())
        self.assertFalse((self.root / 'sample.png.thumb').exists())


if __name__ == '__main__':
    unittest.main()
