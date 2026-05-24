import os
import tempfile
import unittest
from unittest import mock

from toolkit.hf_model_paths import resolve_hub_single_file_path


class HubSingleFilePathResolverTest(unittest.TestCase):
    def test_local_file_path_is_returned_without_download(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as model_file:
            with mock.patch("toolkit.hf_model_paths.hf_hub_download") as hf_hub_download:
                resolved_path = resolve_hub_single_file_path(model_file.name)

        self.assertEqual(resolved_path, model_file.name)
        hf_hub_download.assert_not_called()

    def test_hub_file_path_is_downloaded(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": ""}), mock.patch(
            "toolkit.hf_model_paths._print_download_status"
        ), mock.patch(
            "toolkit.hf_model_paths.hf_hub_download",
            return_value="C:/cache/model-1k-merge.safetensors",
        ) as hf_hub_download:
            resolved_path = resolve_hub_single_file_path(
                "zhen-nan/L2P/model-1k-merge.safetensors"
            )

        self.assertEqual(resolved_path, "C:/cache/model-1k-merge.safetensors")
        hf_hub_download.assert_called_once_with(
            repo_id="zhen-nan/L2P",
            filename="model-1k-merge.safetensors",
            token=None,
        )

    def test_nested_hub_file_path_is_downloaded(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": ""}), mock.patch(
            "toolkit.hf_model_paths._print_download_status"
        ), mock.patch(
            "toolkit.hf_model_paths.hf_hub_download",
            return_value="C:/cache/model.safetensors",
        ) as hf_hub_download:
            resolve_hub_single_file_path("org/repo/subdir/model.safetensors")

        hf_hub_download.assert_called_once_with(
            repo_id="org/repo",
            filename="subdir/model.safetensors",
            token=None,
        )

    def test_repo_id_and_local_like_paths_are_returned_without_download(self):
        local_like_paths = [
            "org/repo",
            "models/model.safetensors",
            "./models/model.safetensors",
            "../models/model.safetensors",
            "/models/model.safetensors",
            "C:/models/model.safetensors",
            "https://huggingface.co/org/repo/blob/main/model.safetensors",
        ]

        with mock.patch("toolkit.hf_model_paths.hf_hub_download") as hf_hub_download:
            resolved_paths = [
                resolve_hub_single_file_path(model_path)
                for model_path in local_like_paths
            ]

        self.assertEqual(resolved_paths, local_like_paths)
        hf_hub_download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
