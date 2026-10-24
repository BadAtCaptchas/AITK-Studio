"""Preservation setup regressions using temporary datasets only."""
import hashlib
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from toolkit.preservation_manifest import prepare_preservation_manifest, validate_preservation_manifest


class PreservationManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'examples'
        self.root.mkdir()
        self.image = self.root / 'photo.png'
        self.caption = self.root / 'photo.txt'
        self.image.write_bytes(b'image fixture')
        self.caption.write_text('a candid portrait', encoding='utf-8')
        self.snapshot = Path(self.tmp.name) / 'job' / '.continuation' / 'preservation-0000.json'
        self.source = 'a' * 64

    def prepare(self, **kwargs):
        return prepare_preservation_manifest(self.root, self.source, self.snapshot, **kwargs)

    def test_first_start_pins_manual_selection_without_writing_the_dataset(self):
        before = {file.name: file.read_bytes() for file in self.root.iterdir()}
        self.assertEqual(self.prepare(), self.snapshot)
        manifest = json.loads(self.snapshot.read_text())
        self.assertEqual(manifest['origin'], 'user_selected')
        self.assertEqual(manifest['source_sha256'], self.source)
        self.assertNotIn('generation_job', manifest)
        self.assertEqual(manifest['examples'][0]['caption_sha256'], hashlib.sha256(self.caption.read_bytes()).hexdigest())
        self.assertEqual({file.name: file.read_bytes() for file in self.root.iterdir()}, before)
        self.assertEqual(self.prepare(), self.snapshot)

    def test_missing_empty_and_non_utf8_captions_do_not_publish_a_snapshot(self):
        self.caption.unlink()
        with self.assertRaisesRegex(ValueError, 'matching .txt caption'):
            self.prepare()
        for data, message in [(b' \n ', 'cannot be blank'), (b'\xff', 'UTF-8')]:
            self.caption.write_bytes(data)
            with self.assertRaisesRegex(ValueError, message):
                self.prepare()
        self.assertFalse(self.snapshot.exists())

    def test_existing_manual_selection_cannot_be_recreated_after_changes(self):
        self.prepare()
        before = self.snapshot.read_bytes()
        self.caption.write_text('different', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed after curation or selection'):
            self.prepare()
        self.assertEqual(self.snapshot.read_bytes(), before)

    def test_new_or_missing_images_are_rejected_on_retry(self):
        self.prepare()
        extra = self.root / 'extra.png'
        extra.write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'uncurated'):
            self.prepare()
        extra.unlink()
        self.image.unlink()
        with self.assertRaisesRegex(ValueError, 'missing or inaccessible'):
            self.prepare()

    def test_resume_uses_pinned_manifest_and_never_recreates_missing_snapshots(self):
        self.prepare()
        pinned = self.snapshot.with_name('pinned.json')
        self.snapshot.rename(pinned)
        self.assertEqual(self.prepare(previous_snapshot=pinned), pinned)
        pinned.unlink()
        with self.assertRaisesRegex(ValueError, 'manifest is missing'):
            self.prepare(previous_snapshot=pinned)
        self.assertFalse(self.snapshot.exists())

    def test_invalid_curated_manifest_is_never_replaced_by_a_manual_snapshot(self):
        curated = self.root / '.aitk-preservation.json'
        curated.write_text('{invalid', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'manifest is corrupt'):
            self.prepare()
        self.assertFalse(self.snapshot.exists())
        self.assertEqual(curated.read_text(), '{invalid')
        curated.write_text(json.dumps(dict(version=1, source_sha256='b' * 64)), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'does not match the source LoRA'):
            self.prepare()

    def test_nested_datasets_have_portable_relative_paths(self):
        nested = self.root / 'portraits'
        nested.mkdir()
        self.image.rename(nested / self.image.name)
        self.caption.rename(nested / self.caption.name)
        manifest = json.loads(self.prepare().read_text())
        self.assertEqual(manifest['examples'][0]['image'], 'portraits/photo.png')
        self.assertEqual(manifest['examples'][0]['caption'], 'portraits/photo.txt')
        validate_preservation_manifest(self.root, self.source, self.snapshot)

    def test_manifest_paths_cannot_escape_the_dataset(self):
        self.prepare()
        manifest = json.loads(self.snapshot.read_text())
        for name in ['../photo.png', '/photo.png', 'C:/photo.png', 'nested/../photo.png', 'nested\\photo.png']:
            manifest['examples'][0]['image'] = name
            self.snapshot.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unsafe preservation asset'):
                validate_preservation_manifest(self.root, self.source, self.snapshot)

    def test_empty_and_oversized_selections_fail_before_publication(self):
        self.image.unlink()
        with self.assertRaisesRegex(ValueError, '1-512'):
            self.prepare()
        for index in range(513):
            (self.root / f'{index}.png').touch()
        with self.assertRaisesRegex(ValueError, '1-512'):
            self.prepare()
        self.assertFalse(self.snapshot.exists())

    def test_failed_and_concurrent_publication_remains_atomic(self):
        with patch('toolkit.continuation_assets.os.fsync', side_effect=OSError('interrupted')):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                self.prepare()
        self.assertFalse(self.snapshot.exists())
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(lambda _: self.prepare(), range(8)))
        self.assertEqual(paths, [self.snapshot] * 8)
        self.assertEqual(list(self.snapshot.parent.iterdir()), [self.snapshot])


if __name__ == '__main__':
    unittest.main()
