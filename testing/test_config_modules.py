import unittest
import types
import importlib.util
import sys
from importlib.machinery import ModuleSpec
from unittest import mock

album_artwork_module = types.ModuleType("toolkit.audio.album_artwork")
album_artwork_module.add_album_artwork = mock.Mock()

prompt_utils_module = types.ModuleType("toolkit.prompt_utils")
prompt_utils_module.PromptEmbeds = type("PromptEmbeds", (), {})

mocked_modules = {
    "toolkit.audio.album_artwork": album_artwork_module,
    "toolkit.prompt_utils": prompt_utils_module,
}

if importlib.util.find_spec("torchaudio") is None:
    torchaudio_module = types.ModuleType("torchaudio")
    torchaudio_module.__spec__ = ModuleSpec("torchaudio", loader=None)
    mocked_modules["torchaudio"] = torchaudio_module

torchao_module = types.ModuleType("torchao")
torchao_quantization_module = types.ModuleType("torchao.quantization")
torchao_quant_primitives_module = types.ModuleType("torchao.quantization.quant_primitives")
torchao_module.__spec__ = ModuleSpec("torchao", loader=None)
torchao_quantization_module.__spec__ = ModuleSpec("torchao.quantization", loader=None)
torchao_quant_primitives_module.__spec__ = ModuleSpec("torchao.quantization.quant_primitives", loader=None)
torchao_quant_primitives_module._DTYPE_TO_BIT_WIDTH = {}
memory_management_module = types.ModuleType("toolkit.memory_management")
memory_management_module.__path__ = []
memory_offload_module = types.ModuleType("toolkit.memory_management.offload")
memory_offload_module.is_block_offload_arch_supported = mock.Mock(return_value=False)
mocked_modules.update({
    "torchao": torchao_module,
    "torchao.quantization": torchao_quantization_module,
    "torchao.quantization.quant_primitives": torchao_quant_primitives_module,
    "toolkit.memory_management": memory_management_module,
    "toolkit.memory_management.offload": memory_offload_module,
})

with mock.patch.dict(
    "sys.modules",
    mocked_modules,
):
    from toolkit.config_modules import NetworkConfig, ModelConfig, SaveConfig, TrainConfig, validate_configs
    from toolkit.base_lora_metadata import add_base_lora_metadata

sys.modules["toolkit.memory_management"] = memory_management_module
sys.modules["toolkit.memory_management.offload"] = memory_offload_module



class NetworkConfigTest(unittest.TestCase):
    def test_network_weights_alias_sets_pretrained_lora_path(self):
        config = NetworkConfig(network_weights="C:/models/example.safetensors")

        self.assertEqual(config.pretrained_lora_path, "C:/models/example.safetensors")

    def test_pretrained_lora_path_takes_precedence_over_network_weights(self):
        config = NetworkConfig(
            pretrained_lora_path="C:/models/canonical.safetensors",
            network_weights="C:/models/legacy.safetensors",
        )

        self.assertEqual(config.pretrained_lora_path, "C:/models/canonical.safetensors")

    def test_lokr_normal_dropout_is_ignored(self):
        config = NetworkConfig(type="lokr", dropout=0.05)

        self.assertIsNone(config.dropout)

    def test_lokr_defaults_to_upstream_factorization(self):
        config = NetworkConfig(type="lokr")

        self.assertTrue(config.lokr_legacy_factorization)

    def test_lokr_explicit_balanced_factorization_is_preserved(self):
        config = NetworkConfig(type="lokr", lokr_legacy_factorization=False)

        self.assertFalse(config.lokr_legacy_factorization)

    def test_lokr_factor_alias_is_supported(self):
        self.assertEqual(NetworkConfig(type="lokr", factor=8).lokr_factor, 8)
        self.assertEqual(NetworkConfig(type="lokr", network_kwargs={"factor": 16}).lokr_factor, 16)

    def test_validate_rejects_network_without_trainable_target(self):
        with self.assertRaisesRegex(ValueError, "train.train_unet"):
            validate_configs(
                TrainConfig(train_unet=False, train_text_encoder=False),
                ModelConfig(arch="zimage:turbo", name_or_path="Tongyi-MAI/Z-Image-Turbo"),
                SaveConfig(save_format="diffusers"),
                [],
                NetworkConfig(type="lora"),
            )

    def test_validate_rejects_zero_rank_lora_without_conv_target(self):
        with self.assertRaisesRegex(ValueError, "network.linear"):
            validate_configs(
                TrainConfig(),
                ModelConfig(arch="zimage:turbo", name_or_path="Tongyi-MAI/Z-Image-Turbo"),
                SaveConfig(save_format="diffusers"),
                [],
                NetworkConfig(type="lora", linear=0, linear_alpha=0, conv=None),
            )

    def test_validate_rejects_unsupported_zimage_network_type(self):
        with self.assertRaisesRegex(ValueError, "Z-Image"):
            validate_configs(
                TrainConfig(),
                ModelConfig(arch="zimage:turbo", name_or_path="Tongyi-MAI/Z-Image-Turbo"),
                SaveConfig(save_format="diffusers"),
                [],
                NetworkConfig(type="locon"),
            )


class BaseLoraConfigTest(unittest.TestCase):
    def test_model_config_base_lora_defaults(self):
        config = ModelConfig(name_or_path="base-model")

        self.assertIsNone(config.base_lora_path)
        self.assertEqual(config.base_lora_strength, 1.0)

    def test_model_config_base_lora_strength_parses(self):
        config = ModelConfig(
            name_or_path="base-model",
            base_lora_path="C:/models/upstream.safetensors",
            base_lora_strength="0.75",
        )

        self.assertEqual(config.base_lora_path, "C:/models/upstream.safetensors")
        self.assertEqual(config.base_lora_strength, 0.75)

    def test_validate_rejects_base_lora_with_inference_lora(self):
        with self.assertRaisesRegex(ValueError, "base_lora_path.*inference_lora_path"):
            validate_configs(
                TrainConfig(),
                ModelConfig(
                    name_or_path="base-model",
                    base_lora_path="C:/models/upstream.safetensors",
                    inference_lora_path="C:/models/sample-only.safetensors",
                ),
                SaveConfig(save_format="diffusers"),
                [],
                NetworkConfig(type="lora"),
            )

    def test_base_lora_metadata_emission(self):
        config = ModelConfig(
            name_or_path="base-model",
            base_lora_path="C:/models/upstream.safetensors",
            base_lora_strength=0.5,
        )
        meta = {}

        add_base_lora_metadata(meta, config)

        self.assertTrue(meta["aitk_trained_on_adapted_base"])
        self.assertEqual(meta["aitk_base_lora_path"], "C:/models/upstream.safetensors")
        self.assertEqual(meta["aitk_base_lora_strength"], "0.5")


class FluxGuidanceBypassConfigTest(unittest.TestCase):
    def test_validate_accepts_official_flux_without_guidance_bypass(self):
        cases = [
            ("flux", "black-forest-labs/FLUX.1-dev"),
            ("flux", "black-forest-labs/FLUX.1-schnell"),
            ("flux_kontext", "black-forest-labs/FLUX.1-Kontext-dev"),
        ]

        for arch, name_or_path in cases:
            with self.subTest(name_or_path=name_or_path):
                validate_configs(
                    TrainConfig(bypass_guidance_embedding=False),
                    ModelConfig(arch=arch, name_or_path=name_or_path),
                    SaveConfig(save_format="diffusers"),
                    [],
                    NetworkConfig(type="lora"),
                )

    def test_validate_rejects_official_flux_guidance_bypass(self):
        cases = [
            ("flux", "black-forest-labs/FLUX.1-dev"),
            ("flux", "black-forest-labs/FLUX.1-schnell"),
            ("flux_kontext", "black-forest-labs/FLUX.1-Kontext-dev"),
        ]

        for arch, name_or_path in cases:
            with self.subTest(name_or_path=name_or_path):
                with self.assertRaisesRegex(ValueError, "bypass_guidance_embedding.*official FLUX"):
                    validate_configs(
                        TrainConfig(bypass_guidance_embedding=True),
                        ModelConfig(arch=arch, name_or_path=name_or_path),
                        SaveConfig(save_format="diffusers"),
                        [],
                        NetworkConfig(type="lora"),
                    )

    def test_validate_accepts_ideogram_and_klein_without_guidance_bypass(self):
        cases = [
            ("ideogram4", "ideogram-ai/ideogram-4-nf4"),
            ("ideogram4:fp8", "ideogram-ai/ideogram-4-fp8"),
            ("ideogram4:nvfp4", "Comfy-Org/Ideogram-4"),
            ("flux2_klein_4b", "black-forest-labs/FLUX.2-klein-base-4B"),
            ("flux2_klein_9b", "black-forest-labs/FLUX.2-klein-base-9B"),
            ("asymflux2_klein_9b", "Lakonik/AsymFLUX.2-klein-9B"),
        ]

        for arch, name_or_path in cases:
            with self.subTest(name_or_path=name_or_path):
                validate_configs(
                    TrainConfig(bypass_guidance_embedding=False),
                    ModelConfig(arch=arch, name_or_path=name_or_path),
                    SaveConfig(save_format="diffusers"),
                    [],
                    NetworkConfig(type="lora"),
                )

    def test_validate_rejects_ideogram_and_klein_guidance_bypass(self):
        cases = [
            ("ideogram4", "ideogram-ai/ideogram-4-nf4"),
            ("ideogram4:fp8", "ideogram-ai/ideogram-4-fp8"),
            ("ideogram4:nvfp4", "Comfy-Org/Ideogram-4"),
            ("flux2_klein_4b", "black-forest-labs/FLUX.2-klein-base-4B"),
            ("flux2_klein_9b", "black-forest-labs/FLUX.2-klein-base-9B"),
            ("asymflux2_klein_9b", "Lakonik/AsymFLUX.2-klein-9B"),
        ]

        for arch, name_or_path in cases:
            with self.subTest(name_or_path=name_or_path):
                with self.assertRaisesRegex(ValueError, "bypass_guidance_embedding.*(Ideogram 4|Klein)"):
                    validate_configs(
                        TrainConfig(bypass_guidance_embedding=True),
                        ModelConfig(arch=arch, name_or_path=name_or_path),
                        SaveConfig(save_format="diffusers"),
                        [],
                        NetworkConfig(type="lora"),
                    )

    def test_validate_accepts_flex_guidance_bypass(self):
        cases = [
            ("flex1", "ostris/Flex.1-alpha"),
            ("flex2", "ostris/Flex.2-preview"),
        ]

        for arch, name_or_path in cases:
            with self.subTest(name_or_path=name_or_path):
                validate_configs(
                    TrainConfig(bypass_guidance_embedding=True),
                    ModelConfig(arch=arch, name_or_path=name_or_path),
                    SaveConfig(save_format="diffusers"),
                    [],
                    NetworkConfig(type="lora"),
                )

    def test_validate_preserves_use_flux_cfg_guidance_bypass(self):
        train_config = TrainConfig(bypass_guidance_embedding=False)

        validate_configs(
            train_config,
            ModelConfig(arch="flux", name_or_path="black-forest-labs/FLUX.1-dev", use_flux_cfg=True),
            SaveConfig(save_format="diffusers"),
            [],
            NetworkConfig(type="lora"),
        )

        self.assertTrue(train_config.bypass_guidance_embedding)


class SegaDistillConfigTest(unittest.TestCase):
    def test_sega_distill_defaults_disabled(self):
        config = TrainConfig()

        self.assertFalse(config.sega_distill)
        self.assertEqual(config.sega_distill_weight, 1.0)
        self.assertEqual(config.sega_distill_base_resolution, 1024)

    def test_sega_distill_validation_accepts_flux2_lora(self):
        validate_configs(
            TrainConfig(sega_distill=True),
            ModelConfig(arch="flux2", name_or_path="black-forest-labs/FLUX.2-dev"),
            SaveConfig(save_format="diffusers"),
            [],
            NetworkConfig(type="lora"),
        )

    def test_sega_distill_validation_accepts_zimage_lora(self):
        validate_configs(
            TrainConfig(sega_distill=True),
            ModelConfig(arch="zimage:turbo", name_or_path="Tongyi-MAI/Z-Image-Turbo"),
            SaveConfig(save_format="diffusers"),
            [],
            NetworkConfig(type="lora"),
        )

    def test_sega_distill_rejects_unsupported_arch_and_non_lora(self):
        with self.assertRaisesRegex(ValueError, "supports"):
            validate_configs(
                TrainConfig(sega_distill=True),
                ModelConfig(arch="flex1", name_or_path="ostris/Flex.1-alpha"),
                SaveConfig(save_format="diffusers"),
                [],
                NetworkConfig(type="lora"),
            )

        with self.assertRaisesRegex(ValueError, "network.type"):
            validate_configs(
                TrainConfig(sega_distill=True),
                ModelConfig(arch="flux2", name_or_path="black-forest-labs/FLUX.2-dev"),
                SaveConfig(save_format="diffusers"),
                [],
                NetworkConfig(type="lokr"),
            )

    def test_sega_distill_rejects_conflicting_targets(self):
        conflicts = [
            ("differential output preservation", {"diff_output_preservation": True}),
            ("blank prompt preservation", {"blank_prompt_preservation": True}),
            ("prior divergence", {"do_prior_divergence": True}),
            ("inverted mask prior", {"inverted_mask_prior": True}),
            ("differential guidance", {"do_differential_guidance": True}),
        ]
        for expected_message, kwargs in conflicts:
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(ValueError, expected_message):
                    validate_configs(
                        TrainConfig(sega_distill=True, **kwargs),
                        ModelConfig(arch="flux2_klein_4b", name_or_path="black-forest-labs/FLUX.2-klein-base-4B"),
                        SaveConfig(save_format="diffusers"),
                        [],
                        NetworkConfig(type="lora"),
                    )


if __name__ == "__main__":
    unittest.main()
