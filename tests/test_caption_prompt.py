import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from toolkit.caption_prompt import expand_caption_tokens, MAX_CAPTION_BYTES


class CaptionTokenTests(unittest.TestCase):
    def test_siblings_json_missing_invalid_and_nonrecursive(self):
        with tempfile.TemporaryDirectory() as folder:
            media = Path(folder) / 'photo.png'
            (Path(folder) / 'photo.txt').write_text('  a cat <caption.json>  ', encoding='utf-8')
            (Path(folder) / 'photo.json').write_text('{"x":1}', encoding='utf-8')
            (Path(folder) / 'photo.bad').write_bytes(b'\xff')
            (Path(folder) / 'photo.huge').write_bytes(b'x' * (MAX_CAPTION_BYTES + 1))
            self.assertEqual(expand_caption_tokens('<caption.txt>|<caption.json>|<caption.none>|<caption.bad>|<caption.huge>', str(media)),
                'a cat <caption.json>|{"x":1}|||')
            self.assertEqual(expand_caption_tokens('<caption../secret>', str(media)), '<caption../secret>')
            self.assertIsNone(expand_caption_tokens(None, str(media)))

    def test_encrypted_reader_is_authoritative_and_reads_repeated_token_once(self):
        reader = Mock(side_effect=lambda extension: 'secret caption' if extension == 'txt' else '')
        with patch('toolkit.caption_prompt.os.open', side_effect=AssertionError('plaintext access')):
            result = expand_caption_tokens('<caption.txt>|<caption.txt>|<caption.json>', '/data/file.png', caption_reader=reader)
        self.assertEqual(result, 'secret caption|secret caption|')
        self.assertEqual(reader.call_count, 2)

    def test_symlink_escape_is_blank(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / 'secret.txt'
            secret.write_text('must not read')
            try:
                (Path(folder) / 'photo.txt').symlink_to(secret)
            except OSError:
                self.skipTest('symlinks unavailable')
            self.assertEqual(expand_caption_tokens('<caption.txt>', str(Path(folder) / 'photo.png')), '')
