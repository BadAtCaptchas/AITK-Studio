from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from extensions_built_in.diffusion_models.flux2.flux2_model import Flux2Model
from toolkit.models.wan21.wan21 import Wan21


REPO_ROOT = Path(__file__).resolve().parents[1]


class RecordingModule(torch.nn.Module):
    def __init__(self, block_path: str = "blocks"):
        super().__init__()
        self.to_calls = []
        setattr(
            self,
            block_path,
            torch.nn.ModuleList(
                [torch.nn.Sequential(torch.nn.Linear(32, 32, bias=False))]
            ),
        )

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


class OrbitStagedLoadPathTest(unittest.TestCase):
    def test_flux2_orbit_helper_forwards_workspace_and_staging_contract(self):
        model = object.__new__(Flux2Model)
        model.device_torch = torch.device("cuda")
        model.torch_dtype = torch.bfloat16
        model.model_config = SimpleNamespace(
            quantize_kwargs={"kernel": "torch", "max_workspace_mb": 24}
        )
        model.print_and_status_update = mock.Mock()
        component = RecordingModule()

        with (
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_model.patch_dequantization_on_save"
            ) as patch_save,
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_model.quantize_component_in_stages"
            ) as staged_quantize,
        ):
            used = model._quantize_orbit_component(
                component,
                "orbit4",
                block_paths=["blocks"],
                component_label="transformer",
                exclude=["output"],
            )

        self.assertTrue(used)
        patch_save.assert_called_once_with(component)
        staged_quantize.assert_called_once_with(
            component,
            weights="orbit4",
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            block_paths=["blocks"],
            exclude=["output"],
            options={"kernel": "torch", "max_workspace_mb": 24},
            component_label="transformer",
        )

    def test_flux2_mistral_orbit_never_moves_whole_encoder_to_cuda_first(self):
        model = object.__new__(Flux2Model)
        model.device_torch = torch.device("cuda")
        model.torch_dtype = torch.bfloat16
        model.model_config = SimpleNamespace(
            quantize_te=True,
            qtype_te="orbit4",
            quantize_kwargs={"kernel": "torch", "max_workspace_mb": 16},
            layer_offloading=False,
            layer_offloading_text_encoder_percent=0.0,
        )
        model.print_and_status_update = mock.Mock()
        model._get_mistral_source_candidates = lambda: [
            {"label": "test", "allow_fallback": False}
        ]
        text_encoder = RecordingModule()
        model._load_mistral_text_encoder = lambda _source, _dtype: text_encoder
        processor = object()
        model._load_mistral_processor = lambda _source: processor

        with (
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_model.patch_dequantization_on_save"
            ),
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_model.quantize_component_in_stages"
            ) as staged_quantize,
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_model.flush"
            ),
        ):
            loaded_encoder, loaded_processor = model.load_te()

        self.assertIs(loaded_encoder, text_encoder)
        self.assertIs(loaded_processor, processor)
        self.assertEqual(text_encoder.to_calls, [])
        self.assertEqual(staged_quantize.call_args.kwargs["weights"], "orbit4")
        self.assertEqual(
            staged_quantize.call_args.kwargs["options"],
            {"kernel": "torch", "max_workspace_mb": 16},
        )

    def test_wan_orbit_transformer_uses_staged_quantization_from_cpu(self):
        model = object.__new__(Wan21)
        model.torch_dtype = torch.bfloat16
        model.device_torch = torch.device("cuda")
        model.model_config = SimpleNamespace(
            quantize=True,
            qtype="orbit4",
            quantize_kwargs={"kernel": "torch", "max_workspace_mb": 32},
            split_model_over_gpus=False,
            low_vram=False,
            assistant_lora_path=None,
            inference_lora_path=None,
            lora_path=None,
            layer_offloading=False,
            layer_offloading_transformer_percent=0.0,
        )
        model.print_and_status_update = mock.Mock()
        transformer = RecordingModule()

        with (
            mock.patch(
                "toolkit.models.wan21.wan21.WanTransformer3DModel.from_pretrained",
                return_value=transformer,
            ),
            mock.patch(
                "toolkit.models.wan21.wan21.patch_dequantization_on_save"
            ),
            mock.patch(
                "toolkit.models.wan21.wan21.quantize_component_in_stages"
            ) as staged_quantize,
            mock.patch("toolkit.models.wan21.wan21.quantize_model") as legacy_quantize,
            mock.patch("toolkit.models.wan21.wan21.flush"),
        ):
            loaded = model.load_wan_transformer("test-model")

        self.assertIs(loaded, transformer)
        legacy_quantize.assert_not_called()
        staged_quantize.assert_called_once_with(
            transformer,
            weights="orbit4",
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            block_paths=["blocks"],
            exclude=None,
            options={"kernel": "torch", "max_workspace_mb": 32},
            component_label="transformer",
        )
        requested_devices = [
            kwargs.get("device", args[0] if args else None)
            for args, kwargs in transformer.to_calls
        ]
        self.assertNotIn(torch.device("cuda"), requested_devices)
        self.assertNotIn("cuda", requested_devices)

    def test_flux1_source_guards_whole_component_cuda_moves_for_orbit(self):
        source = (REPO_ROOT / "toolkit" / "stable_diffusion_model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "if not self.low_vram and not orbit_transformer_quantization:", source
        )
        self.assertIn("weights=self.model_config.qtype_te", source)
        self.assertIn('block_paths=["encoder.block"]', source)


if __name__ == "__main__":
    unittest.main()
