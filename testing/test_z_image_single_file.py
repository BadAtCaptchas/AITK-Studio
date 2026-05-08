import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

paths_module_path = (
    Path(__file__).resolve().parents[1]
    / "extensions_built_in"
    / "diffusion_models"
    / "z_image"
    / "paths.py"
)
spec = importlib.util.spec_from_file_location("z_image_paths", paths_module_path)
z_image_paths = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = z_image_paths
spec.loader.exec_module(z_image_paths)
resolve_single_file_model_path = z_image_paths.resolve_single_file_model_path


class ZImageSingleFileResolverTest(unittest.TestCase):
    def test_local_file_path_is_returned_without_download(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as model_file:
            with mock.patch(
                "z_image_paths.huggingface_hub.hf_hub_download"
            ) as hf_hub_download:
                resolved_path = resolve_single_file_model_path(model_file.name)

        self.assertEqual(resolved_path, model_file.name)
        hf_hub_download.assert_not_called()

    def test_hub_file_path_is_downloaded(self):
        with mock.patch(
            "z_image_paths.huggingface_hub.hf_hub_download",
            return_value="C:/cache/Juggernaut_Z_V1_by_RunDiffusion.safetensors",
        ) as hf_hub_download:
            resolved_path = resolve_single_file_model_path(
                "RunDiffusion/Juggernaut-Z-Image/"
                "Juggernaut_Z_V1_by_RunDiffusion.safetensors"
            )

        self.assertEqual(
            resolved_path,
            "C:/cache/Juggernaut_Z_V1_by_RunDiffusion.safetensors",
        )
        hf_hub_download.assert_called_once_with(
            repo_id="RunDiffusion/Juggernaut-Z-Image",
            filename="Juggernaut_Z_V1_by_RunDiffusion.safetensors",
        )

    def test_invalid_hub_file_path_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_single_file_model_path("Juggernaut_Z_V1_by_RunDiffusion.safetensors")


if __name__ == "__main__":
    unittest.main()
