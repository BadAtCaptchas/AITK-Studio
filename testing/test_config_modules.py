import unittest
import types
from unittest import mock

album_artwork_module = types.ModuleType("toolkit.audio.album_artwork")
album_artwork_module.add_album_artwork = mock.Mock()

prompt_utils_module = types.ModuleType("toolkit.prompt_utils")
prompt_utils_module.PromptEmbeds = type("PromptEmbeds", (), {})

with mock.patch.dict(
    "sys.modules",
    {
        "toolkit.audio.album_artwork": album_artwork_module,
        "toolkit.prompt_utils": prompt_utils_module,
    },
):
    from toolkit.config_modules import NetworkConfig, ModelConfig, SaveConfig, TrainConfig, validate_configs



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
