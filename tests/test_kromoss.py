"""Weight-free Kromoss source, holder, export and training-contract tests."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

import extensions_built_in.diffusion_models
from extensions_built_in.diffusion_models.moss_image.moss_image import MossImageModel
from extensions_built_in.diffusion_models.moss_image.src import checkpoints
from extensions_built_in.diffusion_models.moss_image.src.transformer import MossImageParams, MossImageTransformer
from toolkit.base_lora_metadata import add_distillation_lora_metadata
from toolkit.config_modules import ModelConfig, NetworkConfig, SaveConfig, TrainConfig, validate_configs
from toolkit.model_registry import resolve_model
from toolkit.models.v2.pool import ComponentPool
from extensions_built_in.diffusion_models.moss_image.src.distillation import FrozenMossDistillationAdapter, get_distillation_adapter


HOLDER = "extensions_built_in.diffusion_models.moss_image.moss_image"
SOURCES = "extensions_built_in.diffusion_models.moss_image.src.checkpoints"
TURBO = f"{checkpoints.KROMOSS_REPO}/{checkpoints.KROMOSS_DISTILLATION_FILE}"


def holder(**kwargs):
    return MossImageModel("cpu", ModelConfig(arch="moss_image", name_or_path=checkpoints.KROMOSS_REPO, **kwargs), dtype="fp32")


def tiny_model():
    return MossImageTransformer(MossImageParams(dim=32, ffn_dim=64, num_heads=4,
        num_layers=2, text_dim=16, freq_dim=16, latent_channels=4))


def tiny_distillation(model):
    weights = {}
    for index, layer in enumerate(model.layers):
        for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
            module = layer.proj["qkv" if name in ("q", "k", "v") else name]
            out_features = module.out_features // 3 if name in ("q", "k", "v") else module.out_features
            prefix = f"diffusion_model.layers.{index}.proj.{name}"
            weights[prefix + ".lora_down.weight"] = torch.randn(2, module.in_features) * .01
            weights[prefix + ".lora_up.weight"] = torch.randn(out_features, 2) * .01
            weights[prefix + ".alpha"] = torch.tensor(2.)
    return weights


class KromossTests(unittest.TestCase):
    def test_variant_registry_and_vae_precision_defaults_preserve_overrides(self):
        for arch in ("moss_image:kromoss", "moss_image:kromoss_turbo"):
            self.assertIs(resolve_model(arch), MossImageModel)
            self.assertEqual(ModelConfig(arch=arch, name_or_path="local", dtype="bf16").vae_dtype, "fp32")
        self.assertEqual(holder().vae_torch_dtype, torch.float32)
        self.assertEqual(holder(vae_dtype="bf16").vae_torch_dtype, torch.bfloat16)
        self.assertEqual(ModelConfig(arch="moss_image", name_or_path=checkpoints.SFT_REPO, dtype="bf16").vae_dtype, "bf16")
        turbo = ModelConfig(arch="moss_image:kromoss_turbo", name_or_path=checkpoints.KROMOSS_REPO,
                            model_kwargs={"max_text_length": 512})
        self.assertEqual(turbo.model_kwargs["distillation_lora_path"], TURBO)
        self.assertEqual(turbo.model_kwargs["sample_shift"], 5.0)
        self.assertEqual(turbo.model_kwargs["uncond_skip_layers"], [])
        self.assertEqual(turbo.model_kwargs["max_text_length"], 512)
        for path in (None, "", "  "):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                ModelConfig(arch="moss_image:kromoss_turbo", name_or_path=checkpoints.KROMOSS_REPO,
                            model_kwargs={"distillation_lora_path": path})

    def test_kromoss_selects_its_own_precision_candidates(self):
        for qtype in (None, "float8", "nvfp4", "convrot8"):
            with self.subTest(qtype=qtype), patch(f"{SOURCES}.resolve_comfy_candidates", return_value="chosen") as resolve:
                self.assertEqual(checkpoints.comfy_transformer_file(checkpoints.KROMOSS_REPO, qtype=qtype), "chosen")
                args = resolve.call_args
                self.assertEqual(args.kwargs["repo_id"], checkpoints.KROMOSS_REPO)
                self.assertIn("diffusion_models/kromoss_bf16.safetensors", args.args[0])
                self.assertEqual(any("int8" in name for name in args.args[0]), qtype == "convrot8")
                self.assertFalse(any("moss_image_base" in name for name in args.args[0]))

    def test_kromoss_vendor_override_keeps_repo_specific_cache(self):
        with patch(f"{SOURCES}.resolve_comfy_candidates") as candidates, patch(f"{SOURCES}.resolve_comfy_file", return_value="vendor") as resolve:
            self.assertEqual(checkpoints.comfy_transformer_file(checkpoints.KROMOSS_REPO, use_comfy_weights=False), "vendor")
            candidates.assert_not_called()
            self.assertEqual(resolve.call_args.args[1], checkpoints.KROMOSS_REPO)
            self.assertIn("vendor_bf16", resolve.call_args.args[0])
        self.assertNotEqual(checkpoints.comfy_transformer_rel(checkpoints.KROMOSS_REPO), checkpoints.comfy_transformer_rel("another/Kromoss"))

    def test_distillation_source_local_hub_and_offline_failure(self):
        with TemporaryDirectory() as root:
            filename = str(Path(root) / "adapter.safetensors")
            save_file({"test": torch.ones(1)}, filename)
            with patch(f"{SOURCES}.download_checkpoint") as download:
                self.assertEqual(checkpoints.resolve_distillation_file(filename), filename)
                download.assert_not_called()
                download.return_value = "downloaded"
                self.assertEqual(checkpoints.resolve_distillation_file(TURBO), "downloaded")
                self.assertEqual(download.call_args.args[:2], (checkpoints.KROMOSS_REPO, checkpoints.KROMOSS_DISTILLATION_FILE))
                self.assertTrue(download.call_args.args[2].endswith(os.path.join("loras", "kromoss_8_steps_dmd.safetensors")))
                download.side_effect = FileNotFoundError("offline cache miss")
                with self.assertRaisesRegex(FileNotFoundError, "offline"):
                    checkpoints.resolve_distillation_file(TURBO)
        with self.assertRaises(FileNotFoundError):
            checkpoints.resolve_distillation_file("missing.safetensors")

    def test_small_vae_selection_and_local_override(self):
        for kwargs in ({}, {"vae_path": "custom/vae.safetensors"}):
            model = holder(**kwargs)
            with patch(f"{HOLDER}.resolve_hub_file", return_value="resolved") as resolve, \
                 patch(f"{HOLDER}.load_file", return_value={}), \
                 patch(f"{HOLDER}.AutoEncoder.load_from_state_dict", return_value=Mock()) as load:
                model._load_vae()
                self.assertEqual(resolve.call_args.args[0], kwargs.get("vae_path", f"{checkpoints.KROMOSS_REPO}/{checkpoints.KROMOSS_VAE_FILE}"))
                self.assertEqual(load.call_args.args[1], torch.float32)
        with TemporaryDirectory() as root:
            local = Path(root) / checkpoints.VENDOR_VAE_FILE
            local.parent.mkdir()
            local.touch()
            model = MossImageModel("cpu", ModelConfig(arch="moss_image:kromoss", name_or_path=root), dtype="fp32")
            with patch(f"{HOLDER}.resolve_hub_file") as resolve, patch(f"{HOLDER}.load_file", return_value={}) as load, \
                 patch(f"{HOLDER}.AutoEncoder.load_from_state_dict", return_value=Mock()):
                model._load_vae()
                resolve.assert_not_called()
                self.assertEqual(Path(load.call_args.args[0]), local)

    def test_small_vae_schema_matches_kromoss_and_preserves_full_moss(self):
        from toolkit.models.v2.vae.flux2_kl import AutoEncoder, AutoEncoderParams, AutoEncoderSmallDecoderParams

        marker = "decoder.up.0.block.0.conv1.bias"
        for width, config_type in ((96, AutoEncoderSmallDecoderParams), (128, AutoEncoderParams)):
            config = AutoEncoder.aitk_config_from_state_dict({marker: torch.empty(width)})
            self.assertIs(type(config), config_type)
            self.assertEqual(config.ch_mult, [1, 2, 4, 4])
            self.assertEqual(config.num_res_blocks, 2)
            self.assertEqual(config.z_channels, 32)
            with torch.device("meta"):
                vae = AutoEncoder(config)
            self.assertEqual(vae.encoder.conv_in.out_channels, 128)
            self.assertEqual(vae.decoder.conv_out.in_channels, width)
            self.assertEqual(vae.encoder.conv_out.out_channels, 64)
            self.assertEqual(vae.decoder.conv_in.in_channels, 32)
            self.assertEqual(vae.bn.num_features, 128)
            self.assertEqual(vae.bn.eps, 1e-4)
            self.assertEqual(vae.ps, [2, 2])
            self.assertEqual(len(vae.encoder.down), 4)
            self.assertEqual(len(vae.decoder.up), 4)
            self.assertTrue(all(len(stage.block) == 2 for stage in vae.encoder.down))
            self.assertTrue(all(len(stage.block) == 3 for stage in vae.decoder.up))

    def test_pool_namespace_is_consumed_and_isolates_persistent_adapters(self):
        with TemporaryDirectory() as root:
            filename = str(Path(root) / "tiny.safetensors")
            model = tiny_model()
            save_file(model.convert_state_dict_on_save(model.state_dict()), filename)
            with patch.object(ComponentPool, "current", ComponentPool()):
                plain = MossImageTransformer.load_model(filename, dtype=torch.float32)
                turbo = MossImageTransformer.load_model(filename, dtype=torch.float32, pool_namespace="adapter-a:123")
                self.assertIsNot(plain, turbo)
                self.assertIs(turbo, MossImageTransformer.load_model(filename, dtype=torch.float32, pool_namespace="adapter-a:123"))
                self.assertIsNot(turbo, MossImageTransformer.load_model(filename, dtype=torch.float32, pool_namespace="adapter-a:124"))

    def test_pooled_full_turbo_full_switch_preserves_outputs_and_hook_identity(self):
        with TemporaryDirectory() as root:
            source = tiny_model()
            checkpoint = str(Path(root) / "base.safetensors")
            adapter = str(Path(root) / "dmd.safetensors")
            save_file(source.convert_state_dict_on_save(source.state_dict()), checkpoint)
            save_file(tiny_distillation(source), adapter)
            pool = ComponentPool()
            x, t, context = torch.randn(1, 4, 2, 2), torch.tensor([500.]), torch.randn(1, 3, 16)
            holders, outputs = [], []
            with patch.object(ComponentPool, "current", pool):
                for turbo in (False, True, True, False):
                    config = ModelConfig(arch="moss_image:kromoss", name_or_path=checkpoint,
                                         model_kwargs={"distillation_lora_path": adapter} if turbo else {})
                    model = MossImageModel("cpu", config, dtype="fp32")
                    with patch.object(model, "_load_text_encoder", return_value=Mock()), \
                         patch.object(model, "_load_tokenizer", return_value=Mock()), \
                         patch.object(model, "_load_vae", return_value=Mock()):
                        model.load_model()
                    holders.append(model)
                    with torch.no_grad():
                        outputs.append(model.model(x, t, context))
                self.assertIs(holders[0].model, holders[3].model)
                self.assertIs(holders[1].model, holders[2].model)
                self.assertIsNot(holders[0].model, holders[1].model)
                self.assertIsNone(get_distillation_adapter(holders[0].model))
                self.assertEqual(len(holders[1].model.layers[0].proj["qkv"]._forward_hooks), 1)
                torch.testing.assert_close(outputs[0], outputs[3], rtol=0, atol=0)
                torch.testing.assert_close(outputs[1], outputs[2], rtol=0, atol=0)
                self.assertFalse(torch.allclose(outputs[0], outputs[1]))
                pool.clear()
                self.assertIsNone(get_distillation_adapter(holders[1].model))

    def test_holder_resolves_adapter_before_transformer_and_never_merges_it(self):
        with TemporaryDirectory() as root:
            filename = str(Path(root) / "adapter.safetensors")
            save_file({"unimportant": torch.ones(1)}, filename)
            model = holder(model_kwargs={"distillation_lora_path": filename})
            events = []
            transformer = tiny_model()

            def load_transformer():
                events.append("transformer")
                self.assertIn(str(os.stat(filename).st_mtime_ns), model._distillation_identity)
                return transformer

            def attach(*args, **kwargs):
                events.append("adapter")
                self.assertIs(args[0], transformer)
                self.assertEqual(kwargs["identity"], model._distillation_identity)

            with patch.object(model, "_load_transformer", side_effect=load_transformer), \
                 patch.object(model, "_load_text_encoder", return_value=Mock()), \
                 patch.object(model, "_load_tokenizer", return_value=Mock()), \
                 patch.object(model, "_load_vae", return_value=Mock()), \
                 patch(f"{HOLDER}.FrozenMossDistillationAdapter.attach", side_effect=attach):
                model.load_model()
            self.assertEqual(events, ["transformer", "adapter"])
            with self.assertRaisesRegex(ValueError, "separately"):
                model.save_model("unused", {}, torch.float32)

    def test_checkpoint_prefixes_load_strictly(self):
        original = tiny_model()
        weights = original.convert_state_dict_on_save(original.state_dict())
        for prefix in ("", "diffusion_model.", "model.diffusion_model."):
            restored = MossImageTransformer.load_from_state_dict(
                {prefix + key: value for key, value in weights.items()}, torch.float32
            )
            for key, value in original.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value)
        mixed = {"model.diffusion_model." + key: value for key, value in weights.items()}
        mixed["unrelated.weight"] = torch.ones(1)
        with self.assertRaisesRegex(ValueError, "mixes"):
            MossImageTransformer.load_from_state_dict(mixed, torch.float32)
        invalid = dict(weights)
        invalid["video_embedder.weight"] = torch.ones(1)
        with self.assertRaises((RuntimeError, ValueError, IndexError)):
            MossImageTransformer.load_from_state_dict(invalid, torch.float32)

    def test_training_contract_rejects_unsupported_turbo_modes(self):
        model = ModelConfig(arch="moss_image", name_or_path=checkpoints.KROMOSS_REPO,
                            model_kwargs={"distillation_lora_path": TURBO})
        train = TrainConfig(train_unet=True, noise_scheduler="flowmatch")
        for kind in ("lora", "lokr"):
            validate_configs(train, model, SaveConfig(), [], NetworkConfig(type=kind, linear=2, linear_alpha=2))
        with self.assertRaisesRegex(ValueError, "full fine-tuning"):
            validate_configs(train, model, SaveConfig(), [], None)
        train.merge_network_on_save = True
        with self.assertRaisesRegex(ValueError, "separate adapter"):
            validate_configs(train, model, SaveConfig(), [], NetworkConfig(type="lora", linear=2))
        for field in ("base_lora_path", "assistant_lora_path", "inference_lora_path"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                holder(model_kwargs={"distillation_lora_path": TURBO}, **{field: "conflict"})
        with self.assertRaisesRegex(ValueError, "text-to-image"):
            holder(model_kwargs={"distillation_lora_path": TURBO, "edit": True})
        full = ModelConfig(arch="moss_image:kromoss", name_or_path="local/checkpoint.safetensors")
        with self.assertRaisesRegex(ValueError, "full fine-tuning"):
            validate_configs(TrainConfig(), full, SaveConfig(), [], None)
        full.model_kwargs["edit"] = True
        with self.assertRaisesRegex(ValueError, "text-to-image"):
            validate_configs(TrainConfig(), full, SaveConfig(), [], NetworkConfig(type="lora", linear=2))

    def test_generate_process_applies_exported_user_adapters_on_persistent_turbo(self):
        from jobs.process.GenerateProcess import GenerateProcess
        from toolkit.config_modules import GenerateImageConfig

        with TemporaryDirectory() as root:
            transformer = tiny_model()
            adapter_path = str(Path(root) / "distillation.safetensors")
            save_file(tiny_distillation(transformer), adapter_path)
            for kind in ("lora", "lokr"):
                # Exact output key shapes of the existing adapter export hook.
                if kind == "lora":
                    user_weights = {
                        "transformer.layers.0.proj.o.lora_A.weight": torch.randn(2, 32) * .1,
                        "transformer.layers.0.proj.o.lora_B.weight": torch.randn(32, 2) * .1,
                        "transformer.layers.0.proj.o.alpha": torch.tensor(1.),
                    }
                else:
                    user_weights = {
                        "transformer.layers.0.proj.o.lokr_w1": torch.randn(4, 4) * .1,
                        "transformer.layers.0.proj.o.lokr_w2": torch.randn(8, 8) * .1,
                        "transformer.layers.0.proj.o.alpha": torch.tensor(2.),
                    }
                user_path = str(Path(root) / f"trained-{kind}.safetensors")
                model = holder(lora_path=user_path, model_kwargs={"distillation_lora_path": adapter_path})
                save_file(model.convert_lora_weights_before_save(user_weights), user_path)
                current = tiny_model()
                current.load_state_dict(transformer.state_dict())
                process = SimpleNamespace(
                    _init_native_model=lambda: None, sd=model, model_config=model.model_config,
                    device="cpu", torch_dtype=torch.float32, generate_config=SimpleNamespace(compile=False),
                )
                with patch.object(model, "_load_transformer", return_value=current), \
                     patch.object(model, "_load_text_encoder", return_value=Mock()), \
                     patch.object(model, "_load_tokenizer", return_value=Mock()), \
                     patch.object(model, "_load_vae", return_value=Mock()):
                    GenerateProcess._load_native_model(process)
                persistent = get_distillation_adapter(current)
                self.assertIsNotNone(persistent)
                self.assertTrue(model._user_lora.entries)
                x = torch.randn(1, 4, 2, 2)
                t, context = torch.tensor([500.]), torch.randn(1, 3, 16)
                with torch.no_grad():
                    active = current(x, t, context)
                    model._user_lora.strength = 0
                    turbo_only = current(x, t, context)
                self.assertFalse(torch.allclose(active, turbo_only))
                conditional = SimpleNamespace(text_embeds=[context[0]])
                pipeline = Mock(return_value=["image"])
                result = model.generate_single_image(pipeline, GenerateImageConfig(prompt="test", width=32, height=32,
                    output_folder=root, output_ext="png", network_multiplier=.25), conditional, conditional, None, {})
                self.assertEqual(result, "image")
                self.assertEqual(model._user_lora.strength, .25)
                self.assertIs(get_distillation_adapter(current), persistent)
                current.to("meta")
                self.assertIsNone(get_distillation_adapter(current))
                self.assertFalse(hasattr(current, "_moss_user_lora"))

    def test_live_engine_applies_legacy_user_adapter_once_and_updates_strength(self):
        from extensions_built_in.inference_engine.engine import Engine, GenerationJob

        with TemporaryDirectory() as root:
            model = holder(model_kwargs={"distillation_lora_path": TURBO})
            model.model = tiny_model()
            transformer = model.model
            fixed = FrozenMossDistillationAdapter.attach(transformer, tiny_distillation(transformer), identity="fixed")
            user_path = str(Path(root) / "trained.safetensors")
            save_file(model.convert_lora_weights_before_save({
                "transformer.layers.0.proj.o.lora_A.weight": torch.randn(2, 32) * .1,
                "transformer.layers.0.proj.o.lora_B.weight": torch.randn(32, 2) * .1,
                "transformer.layers.0.proj.o.alpha": torch.tensor(2.),
            }), user_path)
            engine = Engine("cpu", str(Path(root) / "output"))
            request = {"arch": "moss_image:kromoss_turbo", "lora_path": user_path}
            x = torch.randn(1, 4, 2, 2)
            t, context = torch.tensor([500.]), torch.randn(1, 3, 16)
            with torch.no_grad():
                turbo_only = transformer(x, t, context)
            with patch.object(engine, "_build_holder", return_value=(model, False)) as build, \
                 patch("extensions_built_in.inference_engine.engine.flush"):
                engine._ensure_holder(GenerationJob(request, {}))
                self.assertNotIn("lora_path", build.call_args.args[1])
                self.assertEqual(len(engine.lora_stack.loras), 1)
                self.assertEqual(engine.lora_stack.loras[0].strength, 1.)
                self.assertEqual(len(transformer.layers[0].proj["o"]._forward_hooks), 2)
                with torch.no_grad():
                    self.assertFalse(torch.equal(transformer(x, t, context), turbo_only))
                # Explicit stack strength wins even when the same local path
                # is spelled differently in the queued-generator field.
                duplicate = dict(request, loras=[{"path": os.path.join(root, ".", "trained.safetensors"), "strength": 0.}])
                engine._ensure_holder(GenerationJob(duplicate, {}))
                build.assert_called_once()
                self.assertEqual(len(engine.lora_stack.loras), 1)
                self.assertEqual(engine.lora_stack.loras[0].strength, 0.)
                self.assertEqual(len(transformer.layers[0].proj["o"]._forward_hooks), 2)
                with torch.no_grad():
                    torch.testing.assert_close(transformer(x, t, context), turbo_only, rtol=0, atol=0)
                self.assertIs(get_distillation_adapter(transformer), fixed)
                engine._drop_holder()
                self.assertEqual(len(transformer.layers[0].proj["o"]._forward_hooks), 1)
                fixed.close()

    def test_exported_inference_hook_order_matches_bf16_training_network(self):
        from toolkit.inference_lora import InferenceLoRA
        from toolkit.lora_special import LoRASpecialNetwork

        source = tiny_model().to(dtype=torch.bfloat16)
        with torch.no_grad():
            source.layers[0].proj["o"].weight.copy_(torch.eye(32, dtype=torch.bfloat16))
        base_weights = {key: value.clone() for key, value in source.state_dict().items()}
        fixed = tiny_distillation(source)
        for key, value in fixed.items():
            if key.endswith(".weight"):
                value.zero_()
        prefix = "diffusion_model.layers.0.proj.o."
        fixed[prefix + "lora_down.weight"][0, 0] = 1
        fixed[prefix + "lora_up.weight"][:, 0] = 1 / 256
        FrozenMossDistillationAdapter.attach(source, fixed, identity="fixed")
        owner = holder(model_kwargs={"distillation_lora_path": TURBO})
        owner.model = source
        config = NetworkConfig(type="lora", linear=2, linear_alpha=2)
        network = LoRASpecialNetwork(
            text_encoder=None, unet=source, lora_dim=2, alpha=2,
            train_unet=True, train_text_encoder=False, network_config=config,
            is_transformer=True, base_model=owner, target_lin_modules=["MossImageTransformer"],
            only_if_contains=["layers.0.proj.o"],
        )
        network.apply_to(None, source, apply_text_encoder=False, apply_unet=True)
        network.force_to("cpu", dtype=torch.float32)
        network.is_active = True
        network._update_torch_multiplier()
        self.assertEqual(len(network.unet_loras), 1)
        with torch.no_grad():
            adapter = network.unet_loras[0]
            adapter.lora_down.weight.zero_()
            adapter.lora_down.weight[0, 0] = 1
            adapter.lora_up.weight.zero_()
            adapter.lora_up.weight[:, 0] = 1 / 128
        x = torch.ones(1, 1, 32, dtype=torch.bfloat16)
        expected = source.layers[0].proj["o"](x)
        with TemporaryDirectory() as root:
            path = str(Path(root) / "trained.safetensors")
            network.save_weights(path, dtype=torch.float32)
            restored = tiny_model().to(dtype=torch.bfloat16)
            restored.load_state_dict(base_weights)
            FrozenMossDistillationAdapter.attach(restored, fixed, identity="fixed")
            inference_holder = holder(model_kwargs={"distillation_lora_path": TURBO})
            inference_holder.model = restored
            user = InferenceLoRA(path).load(inference_holder).attach()
            self.assertTrue(user.prepend_hooks)
            torch.testing.assert_close(restored.layers[0].proj["o"](x), expected, rtol=0, atol=0)
            user.prepend_hooks = False
            user.attach()
            self.assertFalse(torch.equal(restored.layers[0].proj["o"](x), expected))

    def test_export_metadata_and_fused_lokr_alpha_round_trip(self):
        model = holder(model_kwargs={"distillation_lora_path": TURBO})
        meta = {}
        add_distillation_lora_metadata(meta, model.model_config)
        self.assertEqual(meta["aitk_distillation_lora_path"], TURBO)
        self.assertEqual(meta["aitk_distillation_lora_strength"], "1.0")
        self.assertIs(meta["aitk_distillation_lora_required"], True)
        original = {
            "transformer.layers.0.proj.qkv.lokr_w1": torch.randn(2, 2),
            "transformer.layers.0.proj.qkv.lokr_w2": torch.randn(12, 4),
            "transformer.layers.0.proj.qkv.alpha": torch.tensor(2.),
        }
        saved = model.convert_lora_weights_before_save(original)
        self.assertIn("diffusion_model.layers.0.proj.qkv.alpha", saved)
        restored = model.convert_lora_weights_before_load(saved)
        self.assertEqual(restored.keys(), original.keys())
        for key, value in original.items():
            torch.testing.assert_close(restored[key], value)


if __name__ == "__main__":
    unittest.main()
