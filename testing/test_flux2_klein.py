import os
import unittest
from typing import get_args
from unittest import mock

import torch
import yaml

from extensions_built_in.diffusion_models.flux2.flux2_klein_model import (
    Flux2Klein4BModel,
    Flux2Klein9BModel,
)
from toolkit.config_modules import ModelArch, ModelConfig
from toolkit.util.get_model import get_model_class


OFFICIAL_LORA_FILTER = [
    "img_attn.qkv",
    "img_attn.proj",
    "transformer.single_blocks.",
]


def make_klein_model(model_cls=Flux2Klein4BModel, name_or_path=None, **model_kwargs):
    model = object.__new__(model_cls)
    model.model_config = ModelConfig(
        arch=model_cls.arch,
        name_or_path=name_or_path or "black-forest-labs/FLUX.2-klein-base-4B",
        low_vram=True,
        quantize_te=False,
        **model_kwargs,
    )
    model.torch_dtype = torch.float32
    model.device_torch = torch.device("cpu")
    model.print_and_status_update = mock.Mock()
    return model


class Flux2KleinCompatibilityTest(unittest.TestCase):
    def test_registry_and_model_arch_include_plain_klein_models(self):
        self.assertIn("flux2_klein_4b", get_args(ModelArch))
        self.assertIn("flux2_klein_9b", get_args(ModelArch))

        config_4b = ModelConfig(
            arch="flux2_klein_4b",
            name_or_path="black-forest-labs/FLUX.2-klein-base-4B",
        )
        config_9b = ModelConfig(
            arch="flux2_klein_9b",
            name_or_path="black-forest-labs/FLUX.2-klein-base-9B",
        )

        self.assertIs(get_model_class(config_4b), Flux2Klein4BModel)
        self.assertIs(get_model_class(config_9b), Flux2Klein9BModel)

    def test_load_te_prefers_bfl_subfolders(self):
        model = make_klein_model()
        text_encoder = mock.Mock()
        tokenizer = object()

        with (
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_klein_model.Qwen3ForCausalLM.from_pretrained",
                return_value=text_encoder,
            ) as text_encoder_from_pretrained,
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_klein_model.Qwen2TokenizerFast.from_pretrained",
                return_value=tokenizer,
            ) as tokenizer_from_pretrained,
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            loaded_text_encoder, loaded_tokenizer = model.load_te()

        self.assertIs(loaded_text_encoder, text_encoder)
        self.assertIs(loaded_tokenizer, tokenizer)
        text_encoder_from_pretrained.assert_called_once_with(
            "black-forest-labs/FLUX.2-klein-base-4B",
            token=None,
            torch_dtype=torch.float32,
            subfolder="text_encoder",
        )
        tokenizer_from_pretrained.assert_called_once_with(
            "black-forest-labs/FLUX.2-klein-base-4B",
            local_files_only=False,
            token=None,
            subfolder="tokenizer",
        )

    def test_load_te_passes_hf_token_to_bfl_subfolders(self):
        model = make_klein_model()
        text_encoder = mock.Mock()
        tokenizer = object()

        with (
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_klein_model.Qwen3ForCausalLM.from_pretrained",
                return_value=text_encoder,
            ) as text_encoder_from_pretrained,
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_klein_model.Qwen2TokenizerFast.from_pretrained",
                return_value=tokenizer,
            ) as tokenizer_from_pretrained,
            mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}, clear=False),
        ):
            loaded_text_encoder, loaded_tokenizer = model.load_te()

        self.assertIs(loaded_text_encoder, text_encoder)
        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertEqual(text_encoder_from_pretrained.call_args.kwargs["token"], "hf_test_token")
        self.assertEqual(tokenizer_from_pretrained.call_args.kwargs["token"], "hf_test_token")

    def test_load_te_falls_back_to_standalone_qwen_repo(self):
        model = make_klein_model()
        text_encoder = mock.Mock()
        tokenizer = object()

        with (
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_klein_model.Qwen3ForCausalLM.from_pretrained",
                side_effect=[RuntimeError("missing official text encoder"), text_encoder],
            ) as text_encoder_from_pretrained,
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_klein_model.Qwen2TokenizerFast.from_pretrained",
                return_value=tokenizer,
            ) as tokenizer_from_pretrained,
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            loaded_text_encoder, loaded_tokenizer = model.load_te()

        self.assertIs(loaded_text_encoder, text_encoder)
        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertEqual(
            text_encoder_from_pretrained.call_args_list[0],
            mock.call(
                "black-forest-labs/FLUX.2-klein-base-4B",
                token=None,
                torch_dtype=torch.float32,
                subfolder="text_encoder",
            ),
        )
        self.assertEqual(
            text_encoder_from_pretrained.call_args_list[1],
            mock.call("Qwen/Qwen3-4B", token=None, torch_dtype=torch.float32),
        )
        tokenizer_from_pretrained.assert_called_once_with(
            "Qwen/Qwen3-4B",
            local_files_only=False,
            token=None,
        )
        status_messages = [
            call.args[0] for call in model.print_and_status_update.call_args_list
        ]
        self.assertTrue(any("falling back to Qwen/Qwen3-4B" in msg for msg in status_messages))

    def test_quantized_text_encoder_cache_key_includes_source_subfolders(self):
        model = make_klein_model()
        official_source, fallback_source = model._get_qwen_source_candidates()

        official_key, official_payload = model._get_qwen_cache_key(official_source)
        fallback_key, fallback_payload = model._get_qwen_cache_key(fallback_source)

        self.assertNotEqual(official_key, fallback_key)
        self.assertEqual(
            official_payload["values"]["text_encoder_subfolder"],
            "text_encoder",
        )
        self.assertEqual(
            official_payload["values"]["tokenizer_subfolder"],
            "tokenizer",
        )
        self.assertIsNone(fallback_payload["values"]["text_encoder_subfolder"])
        self.assertIsNone(fallback_payload["values"]["tokenizer_subfolder"])

    def test_quantized_text_encoder_cache_load_uses_cached_config(self):
        model = make_klein_model()
        source = model._get_qwen_source_candidates()[0]
        text_encoder = mock.Mock()
        cache = mock.Mock()
        cache.has_entry.return_value = True
        cache.load_metadata.return_value = {
            "text_encoder_config": {"model_type": "qwen3"}
        }
        cache.load.return_value = cache.load_metadata.return_value
        model._get_quantized_cache = mock.Mock(return_value=cache)

        with (
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_klein_model.AutoConfig.from_pretrained",
            ) as auto_config_from_pretrained,
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_klein_model._qwen3_from_config",
                return_value=text_encoder,
            ) as qwen3_from_config,
        ):
            loaded = model._load_qwen_quantized_cache(source)

        self.assertIs(loaded, text_encoder)
        auto_config_from_pretrained.assert_not_called()
        qwen3_from_config.assert_called_once()
        cache.load.assert_called_once()
        cache.update_metadata.assert_not_called()
        self.assertTrue(text_encoder._aitk_loaded_from_quantized_cache)

    def test_quantized_text_encoder_cache_save_persists_config(self):
        model = make_klein_model()
        source = model._get_qwen_source_candidates()[0]
        cache = mock.Mock()
        model._get_quantized_cache = mock.Mock(return_value=cache)
        config = mock.Mock()
        config.to_dict.return_value = {
            "model_type": "qwen3",
            "hidden_size": 128,
        }
        text_encoder = mock.Mock(config=config)

        model._save_qwen_quantized_cache(text_encoder, source)

        extra_metadata = cache.save.call_args.kwargs["extra_metadata"]
        self.assertEqual(
            extra_metadata["text_encoder_config"],
            {"model_type": "qwen3", "hidden_size": 128},
        )
        self.assertEqual(
            extra_metadata["source_path"],
            "black-forest-labs/FLUX.2-klein-base-4B",
        )

    def test_quantized_text_encoder_cache_failure_raises_in_offline_mode(self):
        model = make_klein_model()
        source = model._get_qwen_source_candidates()[0]
        cache = mock.Mock()
        cache.has_entry.return_value = True
        cache.load_metadata.return_value = {}
        model._get_quantized_cache = mock.Mock(return_value=cache)

        with (
            mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=True),
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_klein_model.AutoConfig.from_pretrained",
                side_effect=RuntimeError("missing config"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "offline mode is enabled"):
                model._load_qwen_quantized_cache(source)

        cache.load.assert_not_called()

    def test_qwen_text_encoder_load_honors_offline_env(self):
        model = make_klein_model()
        source = model._get_qwen_source_candidates()[0]
        text_encoder = object()

        with (
            mock.patch.dict(os.environ, {"TRANSFORMERS_OFFLINE": "1"}, clear=True),
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_klein_model.Qwen3ForCausalLM.from_pretrained",
                return_value=text_encoder,
            ) as text_encoder_from_pretrained,
        ):
            loaded = model._load_qwen_text_encoder(source, torch.float32)

        self.assertIs(loaded, text_encoder)
        text_encoder_from_pretrained.assert_called_once_with(
            "black-forest-labs/FLUX.2-klein-base-4B",
            token=None,
            torch_dtype=torch.float32,
            subfolder="text_encoder",
            local_files_only=True,
        )

    def test_qwen_tokenizer_does_not_retry_online_in_offline_mode(self):
        model = make_klein_model()
        source = model._get_qwen_source_candidates()[0]

        with (
            mock.patch.dict(os.environ, {"HF_DATASETS_OFFLINE": "1"}, clear=True),
            mock.patch(
                "extensions_built_in.diffusion_models.flux2.flux2_klein_model.Qwen2TokenizerFast.from_pretrained",
                side_effect=RuntimeError("missing tokenizer"),
            ) as tokenizer_from_pretrained,
        ):
            with self.assertRaisesRegex(RuntimeError, "missing tokenizer"):
                model._load_qwen_tokenizer(source)

        tokenizer_from_pretrained.assert_called_once_with(
            "black-forest-labs/FLUX.2-klein-base-4B",
            local_files_only=True,
            token=None,
            subfolder="tokenizer",
        )

    def test_plain_klein_lora_save_and_load_key_compatibility(self):
        model = object.__new__(Flux2Klein4BModel)
        peft_state_dict = {
            "transformer.double_blocks.0.img_attn.qkv.lora_A.weight": torch.zeros(1, 1),
            "transformer.single_blocks.0.linear1.lora_B.weight": torch.zeros(1, 1),
        }
        legacy_state_dict = {
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": torch.zeros(1, 1),
            "transformer.single_blocks.0.linear1.lora_B.weight": torch.zeros(1, 1),
        }

        self.assertIs(model.convert_lora_weights_before_save(peft_state_dict), peft_state_dict)
        loaded = model.convert_lora_weights_before_load(legacy_state_dict)

        self.assertIn("transformer.double_blocks.0.img_attn.qkv.lora_A.weight", loaded)
        self.assertIn("transformer.single_blocks.0.linear1.lora_B.weight", loaded)
        self.assertNotIn("diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight", loaded)

    def test_plain_klein_example_configs_use_official_lora_filter(self):
        examples = [
            (
                "train_lora_flux2_klein_4b.yaml",
                "flux2_klein_4b",
                "black-forest-labs/FLUX.2-klein-base-4B",
            ),
            (
                "train_lora_flux2_klein_9b.yaml",
                "flux2_klein_9b",
                "black-forest-labs/FLUX.2-klein-base-9B",
            ),
        ]

        for filename, arch, name_or_path in examples:
            with self.subTest(filename=filename):
                path = os.path.join("config", "examples", filename)
                with open(path, "r", encoding="utf-8") as config_file:
                    config = yaml.safe_load(config_file)

                process = config["config"]["process"][0]
                self.assertEqual(process["model"]["arch"], arch)
                self.assertEqual(process["model"]["name_or_path"], name_or_path)
                self.assertEqual(process["train"]["timestep_type"], "weighted")
                self.assertEqual(
                    process["network"]["network_kwargs"]["only_if_contains"],
                    OFFICIAL_LORA_FILTER,
                )


if __name__ == "__main__":
    unittest.main()
