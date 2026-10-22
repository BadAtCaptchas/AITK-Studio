import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image
from toolkit.config_modules import GenerateImageConfig
from toolkit.layered_output import LayeredImageOutput
from toolkit.sample_metadata import publish_sample_files


class SampleMetadataTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='aitk-sample-provenance-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def config(self, **kwargs):
        return GenerateImageConfig(output_path=str(self.root / 'sample.png'), prompt='original --seed 97 --w 640',
                                   sample_step=100, sample_prompt_index=2, sample_sampler='flowmatch',
                                   sample_iteration_id='iteration-one', add_prompt_file=True, **kwargs)

    def test_records_effective_generation_values_and_prompt_sidecar(self):
        config = self.config()
        config.prompt = 'effective rewritten prompt'
        config.save_image_atomic(Image.new('RGB', (64, 64)))
        metadata = json.loads((self.root / 'sample.png.metadata.json').read_text(encoding='utf-8'))
        self.assertEqual((metadata['prompt'], metadata['seed'], metadata['width']), ('effective rewritten prompt', 97, 640))
        self.assertEqual((metadata['step'], metadata['prompt_index'], metadata['iteration_id']), (100, 2, 'iteration-one'))
        self.assertIn('--seed 97', (self.root / 'sample.txt').read_text(encoding='utf-8'))
        config.prompt = 'later configuration change'
        self.assertEqual(json.loads((self.root / 'sample.png.metadata.json').read_text())['prompt'], 'effective rewritten prompt')

    def test_random_seed_is_recorded_and_arbitrary_kwargs_are_excluded(self):
        config = GenerateImageConfig(output_path=str(self.root / 'random.png'), prompt='test', seed=-1,
                                     extra_kwargs={'api_token': 'do-not-write'})
        config.save_image_atomic(Image.new('RGB', (64, 64)))
        text = (self.root / 'random.png.metadata.json').read_text()
        self.assertEqual(json.loads(text)['seed'], config.seed)
        self.assertGreaterEqual(config.seed, 0)
        self.assertNotIn('do-not-write', text)

    def test_prompt_identity_survives_reordering_and_tracks_control_changes(self):
        config = self.config(negative_prompt=None)
        config.save_image_atomic(Image.new('RGB', (64, 64)))
        first = json.loads((self.root / 'sample.png.metadata.json').read_text())
        self.assertEqual(first['negative_prompt'], '')
        config.sample_prompt_index = 7
        config.save_image_atomic(Image.new('RGB', (64, 64)))
        reordered = json.loads((self.root / 'sample.png.metadata.json').read_text())
        self.assertEqual(first['prompt_identity'], reordered['prompt_identity'])
        config.ctrl_img_1 = 'different-reference.png'
        config.save_image_atomic(Image.new('RGB', (64, 64)))
        changed = json.loads((self.root / 'sample.png.metadata.json').read_text())
        self.assertNotEqual(first['prompt_identity'], changed['prompt_identity'])
        self.assertNotIn('different-reference.png', json.dumps(changed))

    def test_metadata_failure_does_not_publish_media(self):
        config = self.config()
        with mock.patch('toolkit.sample_metadata.write_sample_metadata', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                config.save_image_atomic(Image.new('RGB', (64, 64)))
        self.assertFalse((self.root / 'sample.png').exists())
        self.assertFalse((self.root / 'sample.png.metadata.json').exists())

    def test_commit_failure_restores_existing_media_metadata_and_thumbnail(self):
        staging = self.root / '.stage'
        staging.mkdir()
        for folder, text in [(self.root, 'old'), (staging, 'new')]:
            (folder / 'sample.png').write_text(text)
            (folder / 'sample.png.metadata.json').write_text(text)
        (self.root / '.thumbs').mkdir()
        thumbnail = self.root / '.thumbs' / 'sample.png.png'
        thumbnail.write_text('old thumbnail')
        staged_thumbnail = staging / 'sample.png.thumb'
        staged_thumbnail.write_text('new thumbnail')
        replace = os.replace
        def fail_media(source, target):
            if Path(source) == staging / 'sample.png':
                raise OSError('locked media')
            replace(source, target)
        with mock.patch('toolkit.sample_metadata.os.replace', side_effect=fail_media):
            with self.assertRaises(OSError):
                publish_sample_files(staging, self.root, 'sample.png', thumbnail=(staged_thumbnail, '.thumbs/sample.png.png'))
        self.assertEqual((self.root / 'sample.png').read_text(), 'old')
        self.assertEqual((self.root / 'sample.png.metadata.json').read_text(), 'old')
        self.assertEqual(thumbnail.read_text(), 'old thumbnail')

    def test_layered_and_text_samples_publish_metadata(self):
        config = self.config()
        image = Image.new('RGBA', (64, 64))
        config.save_image_atomic(LayeredImageOutput(image, [image]))
        self.assertTrue((self.root / 'sample.png.metadata.json').exists())
        config.save_image_atomic('generated answer')
        self.assertEqual((self.root / 'sample.txt').read_text(), 'generated answer')
        self.assertEqual(json.loads((self.root / 'sample.txt.metadata.json').read_text())['step'], 100)

    def test_publication_rejects_traversal_and_non_file_media(self):
        staging = self.root / '.stage'
        staging.mkdir()
        with self.assertRaises(ValueError):
            publish_sample_files(staging, self.root, '../outside.png')
        (staging / 'sample.png').mkdir()
        with self.assertRaises(ValueError):
            publish_sample_files(staging, self.root, 'sample.png')
        self.assertFalse((self.root / 'sample.png').exists())


if __name__ == '__main__':
    unittest.main()
