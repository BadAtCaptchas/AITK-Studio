import os
import tempfile
import unittest
from unittest import mock

import torch

from extensions_built_in.diffusion_models.flux2.flux2_model import (
    MISTRAL_PATH,
    Flux2Model,
)
from toolkit.config_modules import ModelConfig


def make_flux2_model(name_or_path="black-forest-labs/FLUX.2-dev", **model_kwargs):
    model = object.__new__(Flux2Model)
    model.model_config = ModelConfig(
        arch="flux2",
        name_or_path=name_or_path,
        low_vram=True,
        quantize_te=False,
        **model_kwargs,
    )
    model.torch_dtype = torch.float32
    model.device_torch = torch.device("cpu")
    model.print_and_status_update = mock.Mock()
    return model


class Flux2MistralSourceTest(unittest.TestCase):
    def test_load_te_prefers_local_text_encoder_subfolder(self):
        with tempfile.TemporaryDirectory() as model_path:
            os.makedirs(os.path.join(model_path, "text_encoder"))
            with open(
                os.path.join(model_path, "text_encoder", "config.json"),
                "w",
                encoding="utf-8",
            ) as config_file:
                config_file.write("{}")

            model = make_flux2_model(name_or_path=model_path)
            text_encoder = mock.Mock()
            processor = object()

            with (
                mock.patch(
                    "extensions_built_in.diffusion_models.flux2.flux2_model.Mistral3ForConditionalGeneration.from_pretrained",
                    return_value=text_encoder,
                ) as text_encoder_from_pretrained,
                mock.patch(
                    "extensions_built_in.diffusion_models.flux2.flux2_model.AutoProcessor.from_pretrained",
                    return_value=processor,
                ) as processor_from_pretrained,
                mock.patch(
                    "extensions_built_in.diffusion_models.flux2.flux2_model.HF_TOKEN",
                    None,
                ),
            ):
                loaded_text_encoder, loaded_processor = model.load_te()

        self.assertIs(loaded_text_encoder, text_encoder)
        self.assertIs(loaded_processor, processor)
        text_encoder_from_pretrained.assert_called_once_with(
            model_path,
            torch_dtype=torch.float32,
            use_safetensors=True,
            token=None,
            subfolder="text_encoder",
        )
        processor_from_pretrained.assert_called_once_with(
            model_path,
            token=None,
            subfolder="text_encoder",
        )

    def test_load_te_falls_back_when_no_flux2_text_encoder_candidate_exists(self):
        model = make_flux2_model(name_or_path="custom/model")
        text_encoder = mock.Mock()
        processor = object()

        with (
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_model.Mistral3ForConditionalGeneration.from_pretrained",
                return_value=text_encoder,
            ) as text_encoder_from_pretrained,
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_model.AutoProcessor.from_pretrained",
                return_value=processor,
            ) as processor_from_pretrained,
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_model.HF_TOKEN",
                None,
            ),
        ):
            loaded_text_encoder, loaded_processor = model.load_te()

        self.assertIs(loaded_text_encoder, text_encoder)
        self.assertIs(loaded_processor, processor)
        text_encoder_from_pretrained.assert_called_once_with(
            MISTRAL_PATH,
            torch_dtype=torch.float32,
            use_safetensors=True,
            token=None,
        )
        processor_from_pretrained.assert_called_once_with(
            MISTRAL_PATH,
            token=None,
        )

    def test_explicit_te_name_or_path_wins_over_bundled_text_encoder(self):
        with tempfile.TemporaryDirectory() as model_path:
            os.makedirs(os.path.join(model_path, "text_encoder"))
            model = make_flux2_model(
                name_or_path=model_path,
                te_name_or_path="custom/mistral-text-encoder",
            )
            text_encoder = mock.Mock()
            processor = object()

            with (
                mock.patch(
                    "extensions_built_in.diffusion_models.flux2.flux2_model.Mistral3ForConditionalGeneration.from_pretrained",
                    return_value=text_encoder,
                ) as text_encoder_from_pretrained,
                mock.patch(
                    "extensions_built_in.diffusion_models.flux2.flux2_model.AutoProcessor.from_pretrained",
                    return_value=processor,
                ) as processor_from_pretrained,
                mock.patch(
                    "extensions_built_in.diffusion_models.flux2.flux2_model.HF_TOKEN",
                    None,
                ),
            ):
                loaded_text_encoder, loaded_processor = model.load_te()

        self.assertIs(loaded_text_encoder, text_encoder)
        self.assertIs(loaded_processor, processor)
        text_encoder_from_pretrained.assert_called_once_with(
            "custom/mistral-text-encoder",
            torch_dtype=torch.float32,
            use_safetensors=True,
            token=None,
        )
        processor_from_pretrained.assert_called_once_with(
            "custom/mistral-text-encoder",
            token=None,
        )

    def test_remote_flux2_subfolder_failure_falls_back_to_mistral_path(self):
        model = make_flux2_model()
        text_encoder = mock.Mock()
        processor = object()

        with (
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_model.Mistral3ForConditionalGeneration.from_pretrained",
                side_effect=[RuntimeError("missing remote subfolder"), text_encoder],
            ) as text_encoder_from_pretrained,
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_model.AutoProcessor.from_pretrained",
                return_value=processor,
            ) as processor_from_pretrained,
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_model.HF_TOKEN",
                "hf_test_token",
            ),
        ):
            loaded_text_encoder, loaded_processor = model.load_te()

        self.assertIs(loaded_text_encoder, text_encoder)
        self.assertIs(loaded_processor, processor)
        self.assertEqual(
            text_encoder_from_pretrained.call_args_list[0],
            mock.call(
                "black-forest-labs/FLUX.2-dev",
                torch_dtype=torch.float32,
                use_safetensors=True,
                token="hf_test_token",
                subfolder="text_encoder",
            ),
        )
        self.assertEqual(
            text_encoder_from_pretrained.call_args_list[1],
            mock.call(
                MISTRAL_PATH,
                torch_dtype=torch.float32,
                use_safetensors=True,
                token="hf_test_token",
            ),
        )
        processor_from_pretrained.assert_called_once_with(
            MISTRAL_PATH,
            token="hf_test_token",
        )
        status_messages = [
            call.args[0] for call in model.print_and_status_update.call_args_list
        ]
        self.assertTrue(
            any(f"falling back to {MISTRAL_PATH}" in msg for msg in status_messages)
        )

    def test_broken_local_text_encoder_raises_without_remote_fallback(self):
        with tempfile.TemporaryDirectory() as model_path:
            os.makedirs(os.path.join(model_path, "text_encoder"))
            model = make_flux2_model(name_or_path=model_path)

            with (
                mock.patch(
                    "extensions_built_in.diffusion_models.flux2.flux2_model.Mistral3ForConditionalGeneration.from_pretrained",
                    side_effect=RuntimeError("bad local text encoder"),
                ) as text_encoder_from_pretrained,
                mock.patch(
                    "extensions_built_in.diffusion_models.flux2.flux2_model.AutoProcessor.from_pretrained",
                ) as processor_from_pretrained,
                mock.patch(
                    "extensions_built_in.diffusion_models.flux2.flux2_model.HF_TOKEN",
                    None,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Failed to load FLUX.2 text encoder",
                ):
                    model.load_te()

        text_encoder_from_pretrained.assert_called_once_with(
            model_path,
            torch_dtype=torch.float32,
            use_safetensors=True,
            token=None,
            subfolder="text_encoder",
        )
        processor_from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
