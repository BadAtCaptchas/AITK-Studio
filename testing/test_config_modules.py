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
    import toolkit.config_modules as config_modules
    from toolkit.config_modules import NetworkConfig, ModelConfig, SaveConfig, TrainConfig, validate_configs, apply_orbit4_low_vram_training_defaults
    from toolkit.base_lora_metadata import add_base_lora_metadata
    imported_numeric_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "torch"
        or name.startswith("torch.")
        or name == "numpy"
        or name.startswith("numpy.")
    }

sys.modules.update(imported_numeric_modules)
sys.modules["toolkit.memory_management"] = memory_management_module
sys.modules["toolkit.memory_management.offload"] = memory_offload_module


class TrainConfigOptimalNoisePairingTest(unittest.TestCase):
    def test_optimal_noise_pairing_default_and_upper_bound_are_allowed(self):
        self.assertEqual(TrainConfig().optimal_noise_pairing_samples, 1)
        self.assertEqual(
            TrainConfig(optimal_noise_pairing_samples=16).optimal_noise_pairing_samples,
            16,
        )

    def test_optimal_noise_pairing_rejects_invalid_values(self):
        invalid_values = [0, 17, "2", 1.5, True, None]

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "train\\.optimal_noise_pairing_samples"):
                    TrainConfig(optimal_noise_pairing_samples=value)


class OrbitModelConfigTest(unittest.TestCase):
    def test_orbit4_low_vram_training_profile_fills_raw_defaults(self):
        model = {
            "low_vram": True,
            "quantize": True,
            "qtype": "orbit4",
        }
        train = {}
        datasets = [{"folder_path": "images"}]

        self.assertTrue(
            apply_orbit4_low_vram_training_defaults(model, train, datasets)
        )

        self.assertEqual(train["batch_size"], 1)
        self.assertTrue(train["gradient_checkpointing"])
        self.assertEqual(train["optimizer"], "adamw8bit")
        self.assertTrue(train["cache_text_embeddings"])
        self.assertTrue(train["unload_text_encoder"])
        self.assertTrue(datasets[0]["cache_latents_to_disk"])
        self.assertTrue(datasets[0]["cache_text_embeddings"])

    def test_orbit4_low_vram_training_profile_preserves_raw_overrides(self):
        model = {
            "low_vram": True,
            "quantize": True,
            "qtype": "orbit4",
        }
        train = {
            "batch_size": 2,
            "gradient_checkpointing": False,
            "optimizer": "prodigy",
            "cache_text_embeddings": False,
            "unload_text_encoder": False,
        }
        datasets = [{
            "cache_latents_to_disk": False,
            "cache_text_embeddings": False,
        }]

        apply_orbit4_low_vram_training_defaults(model, train, datasets)

        self.assertEqual(train["batch_size"], 2)
        self.assertFalse(train["gradient_checkpointing"])
        self.assertEqual(train["optimizer"], "prodigy")
        self.assertFalse(train["cache_text_embeddings"])
        self.assertFalse(train["unload_text_encoder"])
        self.assertFalse(datasets[0]["cache_latents_to_disk"])
        self.assertFalse(datasets[0]["cache_text_embeddings"])

    def test_orbit4_low_vram_profile_does_not_cache_trainable_text_encoder(self):
        train = {"train_text_encoder": True}
        datasets = [{}]

        apply_orbit4_low_vram_training_defaults(
            {"low_vram": True, "quantize": True, "qtype": "orbit4"},
            train,
            datasets,
        )

        self.assertNotIn("cache_text_embeddings", train)
        self.assertNotIn("unload_text_encoder", train)
        self.assertNotIn("cache_text_embeddings", datasets[0])
        self.assertTrue(datasets[0]["cache_latents_to_disk"])

    def test_quantize_kwargs_accepts_and_copies_orbit_options(self):
        source = {
            "kernel": "auto",
            "max_workspace_mb": 64,
            "include": ["transformer_blocks.*"],
            "exclude": ["*.embed"],
        }

        config = ModelConfig(name_or_path="base-model", quantize_kwargs=source)
        source["include"].append("mutated-after-parse")

        self.assertEqual(config.quantize_kwargs["kernel"], "auto")
        self.assertEqual(config.quantize_kwargs["max_workspace_mb"], 64)
        self.assertEqual(config.quantize_kwargs["include"], ["transformer_blocks.*"])
        self.assertEqual(config.quantize_kwargs["exclude"], ["*.embed"])

    def test_quantize_kwargs_rejects_invalid_mapping_and_kernel(self):
        with self.assertRaisesRegex(ValueError, "quantize_kwargs must be a mapping"):
            ModelConfig(name_or_path="base-model", quantize_kwargs=[])

        for kernel in ("cuda", 1, []):
            with self.subTest(kernel=kernel):
                with self.assertRaisesRegex(ValueError, r"quantize_kwargs\.kernel"):
                    ModelConfig(
                        name_or_path="base-model",
                        quantize_kwargs={"kernel": kernel},
                    )

    def test_quantize_kwargs_rejects_invalid_workspace(self):
        for workspace in (0, -1, 4097, True, 1.5, "64"):
            with self.subTest(workspace=workspace):
                with self.assertRaisesRegex(ValueError, "max_workspace_mb"):
                    ModelConfig(
                        name_or_path="base-model",
                        quantize_kwargs={"max_workspace_mb": workspace},
                    )

    def test_quantize_kwargs_rejects_invalid_patterns(self):
        invalid_values = (
            {"include": "transformer_blocks.*"},
            {"include": [""]},
            {"exclude": [1]},
            {"exclude": {"*.embed"}},
        )
        for quantize_kwargs in invalid_values:
            with self.subTest(quantize_kwargs=quantize_kwargs):
                with self.assertRaisesRegex(ValueError, "list of non-empty strings"):
                    ModelConfig(
                        name_or_path="base-model",
                        quantize_kwargs=quantize_kwargs,
                    )

    def test_orbit_rejects_full_base_training(self):
        cases = (
            (
                TrainConfig(train_unet=True, train_text_encoder=False),
                ModelConfig(
                    name_or_path="base-model",
                    quantize=True,
                    qtype="orbit4",
                ),
            ),
            (
                TrainConfig(train_unet=False, train_text_encoder=True),
                ModelConfig(
                    name_or_path="base-model",
                    quantize=False,
                    quantize_te=True,
                    qtype_te="orbit4",
                ),
            ),
        )

        for train_config, model_config in cases:
            with self.subTest(qtype=model_config.qtype, qtype_te=model_config.qtype_te):
                with self.assertRaisesRegex(ValueError, "frozen-base adapter training only"):
                    validate_configs(
                        train_config,
                        model_config,
                        SaveConfig(),
                        [],
                        None,
                    )

    def test_orbit_accepts_adapter_training(self):
        for network_type in ("lora", "lokr", "dora"):
            with self.subTest(network_type=network_type):
                validate_configs(
                    TrainConfig(train_unet=True, train_text_encoder=False),
                    ModelConfig(name_or_path="base-model", quantize=True, qtype="orbit4"),
                    SaveConfig(),
                    [],
                    NetworkConfig(type=network_type),
                )

    def test_convrot_accepts_lora_and_constrained_lokr(self):
        for qtype in (
            "convrot4",
            "convrot8",
            "convrotint8",
            "convrotint7",
            "convrotint6",
            "convrotint5",
            "convrotint4",
            "convrotint3",
            "convrotint2",
            "convrotbitnet",
            "convrotcomfyw4a4",
        ):
            for network_type in ("lora", "lokr"):
                with self.subTest(qtype=qtype, network_type=network_type):
                    validate_configs(
                        TrainConfig(train_unet=True, train_text_encoder=False),
                        ModelConfig(name_or_path="base-model", quantize=True, qtype=qtype),
                        SaveConfig(),
                        [],
                        NetworkConfig(type=network_type),
                    )

    def test_convrot_rejects_full_training_dora_and_full_weight_lokr(self):
        with self.assertRaisesRegex(ValueError, "frozen-base adapter training only"):
            validate_configs(
                TrainConfig(train_unet=True),
                ModelConfig(name_or_path="base-model", quantize=True, qtype="convrot4"),
                SaveConfig(),
                [],
                None,
            )
        with self.assertRaisesRegex(ValueError, "lora or lokr only"):
            validate_configs(
                TrainConfig(),
                ModelConfig(name_or_path="base-model", quantize=True, qtype="convrot8"),
                SaveConfig(),
                [],
                NetworkConfig(type="dora"),
            )
        for network_config, message in (
            (NetworkConfig(type="lora", all_layers=True), "network\\.all_layers"),
            (NetworkConfig(type="lokr", lokr_full_rank=True), "full-rank LoKr"),
            (NetworkConfig(type="lokr", lokr_weight_decompose=True), "LoKr weight decomposition"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_configs(
                        TrainConfig(),
                        ModelConfig(name_or_path="base-model", quantize=True, qtype="convrot4"),
                        SaveConfig(),
                        [],
                        network_config,
                    )

    def test_orbit_rejects_unsupported_adapter_types(self):
        for network_type in ("locon", "lorm", "fullrank", "ara"):
            with self.subTest(network_type=network_type):
                with self.assertRaisesRegex(ValueError, "lora, lokr, or dora only"):
                    network_kwargs = {"lorm": {}} if network_type == "lorm" else {}
                    validate_configs(
                        TrainConfig(),
                        ModelConfig(
                            name_or_path="base-model",
                            quantize=True,
                            qtype="orbit4",
                        ),
                        SaveConfig(),
                        [],
                        NetworkConfig(type=network_type, **network_kwargs),
                    )

    def test_orbit_rejects_full_weight_adapter_modes(self):
        cases = (
            (NetworkConfig(type="lora", all_layers=True), "network\\.all_layers"),
            (NetworkConfig(type="lokr", lokr_full_rank=True), "full-rank LoKr"),
            (
                NetworkConfig(type="lokr", lokr_weight_decompose=True),
                "LoKr weight decomposition",
            ),
        )

        for network_config, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_configs(
                        TrainConfig(),
                        ModelConfig(
                            name_or_path="base-model",
                            quantize=True,
                            qtype="orbit4",
                        ),
                        SaveConfig(),
                        [],
                        network_config,
                    )

    def test_orbit4_low_vram_defaults_block_offloading(self):
        model_config = ModelConfig(
            name_or_path="base-model",
            arch="flux2",
            quantize=True,
            qtype="orbit4",
            low_vram=True,
        )
        with mock.patch.object(
            memory_offload_module,
            "is_block_offload_arch_supported",
            return_value=True,
        ):
            validate_configs(
                TrainConfig(),
                model_config,
                SaveConfig(),
                [],
                NetworkConfig(type="lora"),
            )

        self.assertTrue(model_config.layer_offloading)
        self.assertEqual(model_config.layer_offloading_backend, "block")
        self.assertEqual(model_config.layer_offloading_transformer_percent, 0.70)
        self.assertEqual(model_config.layer_offloading_text_encoder_percent, 0.50)

    def test_orbit4_low_vram_preserves_explicit_offloading_values(self):
        model_config = ModelConfig(
            name_or_path="base-model",
            arch="flux2",
            quantize=True,
            qtype="orbit4",
            low_vram=True,
            layer_offloading=False,
            layer_offloading_backend="legacy",
            layer_offloading_transformer_percent=0.25,
            layer_offloading_text_encoder_percent=0.10,
        )
        with mock.patch.object(
            memory_offload_module,
            "is_block_offload_arch_supported",
            return_value=True,
        ):
            validate_configs(
                TrainConfig(),
                model_config,
                SaveConfig(),
                [],
                NetworkConfig(type="lora"),
            )

        self.assertFalse(model_config.layer_offloading)
        self.assertEqual(model_config.layer_offloading_backend, "legacy")
        self.assertEqual(model_config.layer_offloading_transformer_percent, 0.25)
        self.assertEqual(model_config.layer_offloading_text_encoder_percent, 0.10)



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

    def test_sega_distill_validation_accepts_mixed_case_lora(self):
        validate_configs(
            TrainConfig(sega_distill=True),
            ModelConfig(arch="flux2", name_or_path="black-forest-labs/FLUX.2-dev"),
            SaveConfig(save_format="diffusers"),
            [],
            NetworkConfig(type="LoRA"),
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
