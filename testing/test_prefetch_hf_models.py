import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


script_path = Path(__file__).resolve().parents[1] / "scripts" / "prefetch_hf_models.py"
spec = importlib.util.spec_from_file_location("prefetch_hf_models", script_path)
prefetch_hf_models = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prefetch_hf_models
spec.loader.exec_module(prefetch_hf_models)


class PrefetchHfModelsTest(unittest.TestCase):
    def test_l2p_single_file_reference_is_downloaded_to_diffusion_models(self):
        with mock.patch.object(prefetch_hf_models, "is_existing_local_path", return_value=False), mock.patch(
            "prefetch_hf_models.os.path.isfile",
            return_value=False,
        ), mock.patch("prefetch_hf_models.os.makedirs"), mock.patch(
            "prefetch_hf_models.hf_hub_download",
            return_value="models/diffusion_models/model-1k-merge.safetensors",
        ) as hf_hub_download:
            download = prefetch_hf_models.prefetch_reference(
                "zhen-nan/L2P/model-1k-merge.safetensors",
                token="token",
            )

        self.assertEqual(download["kind"], "file")
        hf_hub_download.assert_called_once_with(
            repo_id="zhen-nan/L2P",
            filename="model-1k-merge.safetensors",
            local_dir=mock.ANY,
            token="token",
        )

    def test_repo_reference_is_snapshot_downloaded(self):
        with mock.patch.object(prefetch_hf_models, "is_existing_local_path", return_value=False), mock.patch(
            "prefetch_hf_models.snapshot_download",
            return_value="hf-cache/Tongyi-MAI/Z-Image-Turbo",
        ) as snapshot_download:
            download = prefetch_hf_models.prefetch_reference("Tongyi-MAI/Z-Image-Turbo", token=None)

        self.assertEqual(download["kind"], "snapshot")
        snapshot_download.assert_called_once_with(repo_id="Tongyi-MAI/Z-Image-Turbo", token=None)

    def test_local_like_paths_are_not_downloaded(self):
        values = [
            "models/foo/model.safetensors",
            "./models/foo/model.safetensors",
            "../models/foo/model.safetensors",
            "C:/models/foo/model.safetensors",
            "https://civitai.com/models/123",
        ]

        with mock.patch("prefetch_hf_models.hf_hub_download") as hf_hub_download, mock.patch(
            "prefetch_hf_models.snapshot_download"
        ) as snapshot_download:
            for value in values:
                self.assertIsNone(prefetch_hf_models.classify_hf_reference(value))

        hf_hub_download.assert_not_called()
        snapshot_download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
