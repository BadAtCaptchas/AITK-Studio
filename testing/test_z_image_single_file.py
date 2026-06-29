import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
paths_module_path = (
    PROJECT_ROOT
    / "extensions_built_in"
    / "diffusion_models"
    / "z_image"
    / "paths.py"
)
z_image_module_path = (
    PROJECT_ROOT
    / "extensions_built_in"
    / "diffusion_models"
    / "z_image"
    / "z_image.py"
)
spec = importlib.util.spec_from_file_location("z_image_paths", paths_module_path)
z_image_paths = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = z_image_paths
spec.loader.exec_module(z_image_paths)
resolve_single_file_model_path = z_image_paths.resolve_single_file_model_path


def load_z_image_conversion_functions():
    source = z_image_module_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(z_image_module_path))
    wanted = {"convert_single_file_to_diffusers", "convert_diffusers_to_single_file"}
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    test_module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(test_module)
    namespace = {"torch": torch}
    exec(compile(test_module, str(z_image_module_path), "exec"), namespace)
    return (
        namespace["convert_single_file_to_diffusers"],
        namespace["convert_diffusers_to_single_file"],
    )


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

    def test_comfy_single_file_conversion_round_trips(self):
        convert_to_diffusers, convert_to_single_file = load_z_image_conversion_functions()
        state_dict = {
            "layers.0.attention.qkv.weight": torch.arange(18, dtype=torch.float32).reshape(6, 3),
            "layers.0.attention.out.weight": torch.ones(2, 2),
            "layers.0.attention.q_norm.weight": torch.full((2,), 2.0),
            "layers.0.attention.k_norm.weight": torch.full((2,), 3.0),
            "x_embedder.proj.weight": torch.full((2, 2), 4.0),
            "final_layer.linear.weight": torch.full((2, 2), 5.0),
            "layers.0.feed_forward.weight": torch.full((2, 2), 6.0),
        }

        diffusers = convert_to_diffusers(state_dict)

        self.assertIn("layers.0.attention.to_q.weight", diffusers)
        self.assertIn("layers.0.attention.to_k.weight", diffusers)
        self.assertIn("layers.0.attention.to_v.weight", diffusers)
        self.assertIn("layers.0.attention.to_out.0.weight", diffusers)
        self.assertIn("layers.0.attention.norm_q.weight", diffusers)
        self.assertIn("layers.0.attention.norm_k.weight", diffusers)
        self.assertIn("all_x_embedder.2-1.proj.weight", diffusers)
        self.assertIn("all_final_layer.2-1.linear.weight", diffusers)

        round_tripped = convert_to_single_file(diffusers)

        self.assertEqual(set(round_tripped), set(state_dict))
        for key, value in state_dict.items():
            self.assertTrue(torch.equal(round_tripped[key], value), key)


if __name__ == "__main__":
    unittest.main()
