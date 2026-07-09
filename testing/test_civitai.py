import hashlib
import os
import tempfile
import unittest
from unittest import mock

from toolkit import civitai


class FakeResponse:
    def __init__(self, *, payload=None, content=b'', content_length=None):
        self._payload = payload
        self._content = content
        self.headers = {
            'content-length': str(len(content) if content_length is None else content_length),
        }

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def iter_content(self, _block_size):
        yield self._content

    def close(self):
        return None


class CivitaiDownloadSecurityTest(unittest.TestCase):
    def test_prefers_safetensors_over_primary_pickle(self):
        response = FakeResponse(
            payload={
                'modelVersions': [
                    {
                        'id': 7,
                        'files': [
                            {
                                'name': 'unsafe.ckpt',
                                'primary': True,
                                'metadata': {'format': 'PickleTensor', 'fp': 'fp16'},
                            },
                            {
                                'name': 'safe.safetensors',
                                'primary': False,
                                'metadata': {'format': 'SafeTensor', 'fp': 'fp32'},
                            },
                        ],
                    }
                ]
            }
        )
        with mock.patch.object(civitai, 'assert_url_allowed'), mock.patch.object(
            civitai.requests, 'get', return_value=response
        ):
            selected, version_id = civitai.get_model_download_info(1)

        self.assertEqual(version_id, 7)
        self.assertEqual(selected['name'], 'safe.safetensors')

    def test_download_sanitizes_name_and_verifies_sha256(self):
        content = b'safe model bytes'
        digest = hashlib.sha256(content).hexdigest()
        metadata = FakeResponse(
            payload={
                'modelVersions': [
                    {
                        'id': 9,
                        'files': [
                            {
                                'name': '../../model.safetensors',
                                'primary': True,
                                'sizeKB': len(content) / 1024,
                                'downloadUrl': 'https://civitai.com/api/download/models/9',
                                'hashes': {'SHA256': digest},
                                'metadata': {'format': 'SafeTensor', 'fp': 'fp16'},
                            }
                        ],
                    }
                ]
            }
        )
        download = FakeResponse(content=content)

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(civitai, 'MODELS_PATH', temp_dir), mock.patch.object(
            civitai, 'assert_url_allowed'
        ), mock.patch.object(civitai.requests, 'get', side_effect=[metadata, download]), mock.patch.object(
            civitai.tqdm, 'tqdm', return_value=mock.Mock()
        ):
            model_path = civitai.get_model_path_from_url('https://civitai.com/models/1?modelVersionId=9')

            self.assertEqual(os.path.commonpath([temp_dir, model_path]), os.path.realpath(temp_dir))
            self.assertEqual(os.path.basename(model_path), 'model.safetensors')
            with open(model_path, 'rb') as handle:
                self.assertEqual(handle.read(), content)


if __name__ == '__main__':
    unittest.main()
