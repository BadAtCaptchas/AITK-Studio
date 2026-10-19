"""Inference parity with actual training exports; no downloaded model weights."""
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from testing.test_adapter_checkpoint_roundtrip import Holder
from toolkit.config_modules import NetworkConfig
from toolkit.inference_lora import InferenceLoRA, LoRAStack
from toolkit.lora_special import LoRASpecialNetwork
from extensions_built_in.inference_engine.engine import Engine, GenerationJob


class AdapterRoot(torch.nn.Module):
    def __init__(self, conv=False, kernel=3, quantized=False, width=8):
        super().__init__()
        self.proj = (torch.nn.Conv2d(width, width, kernel, stride=2, padding=kernel // 2,
                                     dilation=1, bias=True) if conv
                     else torch.nn.Linear(width, width, bias=True))
        if quantized:
            from optimum.quanto import freeze, qint8, quantize
            quantize(self, weights=qint8)
            freeze(self)

    def forward(self, x):
        return self.proj(x)


def training_network(model, kind, **options):
    holder = Holder(model, transformer=not isinstance(model.proj, torch.nn.Conv2d))
    config = NetworkConfig(type=kind, linear=1, linear_alpha=0.7, conv=1, conv_alpha=0.7,
                           transformer_only=False, lokr_full_rank=False, **options)
    network = LoRASpecialNetwork(
        None, model, lora_dim=1, alpha=0.7, conv_lora_dim=1, conv_alpha=0.7,
        network_config=config, network_type=kind, is_transformer=holder.is_transformer,
        train_unet=True, train_text_encoder=False, base_model=holder,
        target_lin_modules=["AdapterRoot"], target_conv_modules=["AdapterRoot"],
        transformer_only=False,
    )
    network.apply_to(None, model, apply_unet=True, apply_text_encoder=False)
    network.force_to("cpu", torch.float32)
    network.is_active = True
    network._update_torch_multiplier()
    network.eval()
    with torch.no_grad():
        for name, param in network.named_parameters():
            if "magnitude" in name or "dora_scale" in name:
                param.mul_(1.4)
            else:
                param.normal_(0, 0.2)
    return holder, network


class InferenceAdapterParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(72811)
        root = Path(__file__).resolve().parents[1] / ".tmp"
        root.mkdir(exist_ok=True)
        folder = tempfile.TemporaryDirectory(prefix="inference-parity-", dir=root)
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def check_export(self, kind, *, conv=False, kernel=3, quantized=False, autocast=False,
                     modes=("hook", "merge"), **options):
        model = AdapterRoot(conv, kernel, quantized).requires_grad_(False)
        plain = copy.deepcopy(model)
        holder, network = training_network(model, kind, **options)
        path = str(self.folder / "trained.safetensors")
        network.save_weights(path, dtype=torch.float32)
        x = torch.randn(2, 8, 9, 11) if conv else torch.randn(2, 3, 8)
        for strength in (0.0, 0.5, 1.0, -0.5):
            network.multiplier = strength
            for mode in modes:
                with self.subTest(kind=kind, conv=conv, quantized=quantized, strength=strength, mode=mode):
                    other = copy.deepcopy(plain)
                    stack = LoRAStack(Holder(other, holder.is_transformer), mode).load([
                        {"path": path, "strength": strength},
                    ])
                    self.assertEqual(len(stack.loras[0].entries), 1)
                    self.assertEqual(stack.loras[0].num_skipped_keys, 0)
                    stack.apply()
                    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                        expected = model(x)
                        actual = other(x)
                    tolerance = 0.025 if autocast else (0.015 if quantized and mode == "merge" else 3e-6)
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                    stack.remove()
                    if mode == "hook":
                        torch.testing.assert_close(other(x), plain(x), rtol=0, atol=0)

    def test_dora_export_hook_and_merge_with_bias_and_strengths(self):
        self.check_export("dora")

    def test_dora_hook_bfloat16_autocast(self):
        self.check_export("dora", autocast=True, modes=("hook",))

    def test_quantized_dora_hook(self):
        self.check_export("dora", quantized=True, modes=("hook",))

    def test_quanto_merge_requantizes_magnitude_adapters(self):
        self.check_export("dora", quantized=True, modes=("merge",))
        self.check_export("lokr", quantized=True, modes=("merge",),
                          lokr_factor=2, lokr_weight_decompose=True)

    def test_packed_orbit_dora_hook_does_not_decode_full_weight(self):
        from toolkit.util.orbit_quant import OrbitQuantizer
        from toolkit.util.ostris_quant import convert_linear_to_ostris
        model = AdapterRoot(width=32).requires_grad_(False)
        self.assertTrue(convert_linear_to_ostris(model.proj, OrbitQuantizer(4, kernel="torch", max_workspace_mb=1)))
        plain = copy.deepcopy(model)
        _, network = training_network(model, "dora")
        path = str(self.folder / "orbit-dora.safetensors")
        network.save_weights(path, dtype=torch.float32)
        x = torch.randn(2, 3, 32)
        with mock.patch.object(plain.proj, "dequantize_weight", side_effect=AssertionError("full decode")):
            adapter = InferenceLoRA(path).load(Holder(plain))
            adapter.attach()
            for strength in (0.0, 0.5, 1.0, -0.5):
                network.multiplier = adapter.strength = strength
                torch.testing.assert_close(plain(x), model(x), rtol=3e-6, atol=3e-6)
            # A different input dtype invalidates the hook cache; the base
            # buffers remain packed and the saved factors remain on CPU.
            network.force_to("cpu", torch.bfloat16)
            model.to(dtype=torch.bfloat16)
            plain.to(dtype=torch.bfloat16)
            torch.testing.assert_close(plain(x.bfloat16()), model(x.bfloat16()), rtol=0.03, atol=0.03)
            adapter.detach()
            self.assertEqual(len(plain.proj._forward_hooks), 0)

    def test_lokr_magnitude_rows_and_columns(self):
        for on_output in (True, False):
            for full in (True, False):
                with self.subTest(on_output=on_output, full=full):
                    self.check_export("lokr", lokr_factor=2, lokr_full_matrix=full,
                                      lokr_weight_decompose=True, lokr_wd_on_output=on_output)

    def test_lokr_magnitude_scalar_rs_export_and_autocast(self):
        self.check_export("lokr", lokr_weight_decompose=True, lokr_factor=2,
                          lokr_use_scalar=True, lokr_rs_lora=True)
        self.check_export("lokr", lokr_weight_decompose=True, lokr_factor=2,
                          autocast=True, modes=("hook",))

    def test_quantized_lokr_magnitude_hook(self):
        self.check_export("lokr", quantized=True, modes=("hook",),
                          lokr_factor=2, lokr_weight_decompose=True)

    def test_convolution_lora_hook_and_merge(self):
        for kernel in (1, 3):
            with self.subTest(kernel=kernel):
                self.check_export("lora", conv=True, kernel=kernel)

    def test_convolution_lokr_hook_and_merge(self):
        for options in ({}, {"lokr_use_tucker": True}, {"lokr_full_matrix": True},
                        {"lokr_weight_decompose": True},
                        {"lokr_weight_decompose": True, "lokr_wd_on_output": False}):
            with self.subTest(options=options):
                self.check_export("lokr", conv=True, lokr_factor=2, **options)

    def test_unsupported_magnitude_overlap_rejected_before_mutation(self):
        model = AdapterRoot().requires_grad_(False)
        plain = copy.deepcopy(model)
        _, network = training_network(model, "dora")
        path = str(self.folder / "dora.safetensors")
        network.save_weights(path, dtype=torch.float32)
        for mode in ("hook", "merge"):
            other = copy.deepcopy(plain)
            original = copy.deepcopy(other.state_dict())
            stack = LoRAStack(Holder(other), mode).load([{"path": path}, {"path": path}])
            with self.assertRaisesRegex(ValueError, "Stacking magnitude"):
                stack.apply()
            self.assertEqual(len(other.proj._forward_hooks), 0)
            for key, value in original.items():
                torch.testing.assert_close(other.state_dict()[key], value, rtol=0, atol=0)

    def test_malformed_magnitude_rejected(self):
        path = str(self.folder / "bad.safetensors")
        save_file({"transformer.proj.lora_magnitude_vector": torch.ones(8)}, path)
        with self.assertRaisesRegex(ValueError, "missing its direction"):
            InferenceLoRA(path).load(Holder(AdapterRoot()))

    def test_merge_preflights_all_adapters_before_mutating_base(self):
        model = AdapterRoot().requires_grad_(False)
        model.second = torch.nn.Linear(8, 8, bias=False).requires_grad_(False)
        model.second.register_buffer("weight_scale", torch.ones(8))
        original = copy.deepcopy(model.state_dict())
        paths = []
        for name in ("proj", "second"):
            path = str(self.folder / f"{name}.safetensors")
            save_file({f"transformer.{name}.lora_A.weight": torch.ones(1, 8),
                       f"transformer.{name}.lora_B.weight": torch.ones(8, 1)}, path)
            paths.append({"path": path})
        stack = LoRAStack(Holder(model), "merge").load(paths)
        with self.assertRaisesRegex(ValueError, "use hook mode"):
            stack.apply()
        for key, expected in original.items():
            torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)

    def engine(self):
        engine = Engine(device="cpu", dtype="fp32", output_folder=str(self.folder / "outputs"),
                        assets_folder=str(self.folder / "assets"))
        self.addCleanup(engine._drop_holder)
        self.addCleanup(engine.pool.clear)
        base = AdapterRoot().requires_grad_(False)
        with torch.no_grad():
            base.proj.weight.zero_()
            base.proj.bias.zero_()

        def build(job, model):
            component = engine.pool.get("shared-tiny-base")
            if component is None:
                component = copy.deepcopy(base)
                engine.pool.put("shared-tiny-base", component)
            return Holder(component), False

        engine._build_holder = build
        return engine

    def write_adapter(self, value, preserve_stat=False):
        path = self.folder / "latest.safetensors"
        old = path.stat() if preserve_stat else None
        save_file({"transformer.proj.lora_A.weight": torch.ones(2, 8),
                   "transformer.proj.lora_B.weight": torch.full((8, 2), value)}, str(path))
        if old is not None:
            self.assertEqual(old.st_size, path.stat().st_size)
            os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
            self.assertEqual(old.st_mtime_ns, path.stat().st_mtime_ns)
        return str(path)

    def test_engine_reloads_same_path_content_even_when_size_and_mtime_match(self):
        for mode in ("hook", "merge"):
            with self.subTest(mode=mode):
                engine = self.engine()
                path = self.write_adapter(0.25)
                model = {"arch": "sd1", "name_or_path": "tiny-local", "lora_mode": mode,
                         "loras": [{"path": path, "strength": 1.0}]}
                job = GenerationJob(model, {})
                engine._ensure_holder(job)
                x = torch.ones(1, 8)
                torch.testing.assert_close(engine.holder.model(x), torch.full((1, 8), 4.0))
                old_stack = engine.lora_stack
                self.write_adapter(0.5, preserve_stat=True)
                job.model["loras"][0]["strength"] = 0.5
                engine._ensure_holder(job)
                self.assertIsNot(old_stack, engine.lora_stack)
                torch.testing.assert_close(engine.holder.model(x), torch.full((1, 8), 4.0))
                job.model["loras"][0]["strength"] = 1.0
                previous_stack = engine.lora_stack
                engine._ensure_holder(job)
                torch.testing.assert_close(engine.holder.model(x), torch.full((1, 8), 8.0))
                if mode == "hook":
                    self.assertIs(engine.lora_stack, previous_stack)

    def test_pool_reused_merged_component_is_not_merged_twice(self):
        engine = self.engine()
        path = self.write_adapter(0.25)
        model = {"arch": "sd1", "name_or_path": "first", "lora_mode": "merge",
                 "loras": [{"path": path, "strength": 1.0}]}
        job = GenerationJob(model, {})
        engine._ensure_holder(job)
        original_component = engine.holder.model
        job.model["name_or_path"] = "second"
        engine._ensure_holder(job)
        self.assertIs(engine.holder.model, original_component)
        torch.testing.assert_close(engine.holder.model(torch.ones(1, 8)), torch.full((1, 8), 4.0))

    def test_engine_discards_partial_hooks_or_merges_after_failure(self):
        for mode, method in (("merge", "merge"), ("hook", "attach")):
            with self.subTest(mode=mode):
                engine = self.engine()
                path = self.write_adapter(0.25)
                job = GenerationJob({"arch": "sd1", "name_or_path": "tiny-local", "lora_mode": mode,
                                     "loras": [{"path": path}, {"path": path}]}, {})
                retained = []
                build = engine._build_holder

                def retain(job, model):
                    holder, legacy = build(job, model)
                    retained.append(holder.model)
                    return holder, legacy

                engine._build_holder = retain
                original = getattr(InferenceLoRA, method)
                calls = 0

                def fail_second(adapter, *args, **kwargs):
                    nonlocal calls
                    calls += 1
                    result = original(adapter, *args, **kwargs)
                    if calls == 2:
                        raise RuntimeError("injected writeback or registration failure")
                    return result

                with mock.patch.object(InferenceLoRA, method, fail_second):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        engine._ensure_holder(job)
                self.assertIsNone(engine.holder)
                self.assertIsNone(engine.lora_stack)
                self.assertEqual(engine.pool.entries, {})
                self.assertEqual(len(retained[0].proj._forward_hooks), 0)
                job.model["loras"] = []
                engine._ensure_holder(job)
                torch.testing.assert_close(engine.holder.model(torch.ones(1, 8)), torch.zeros(1, 8))

    def test_pool_selection_rejects_adapter_changed_during_load(self):
        engine = self.engine()
        path = self.write_adapter(0.25)
        job = GenerationJob({"arch": "sd1", "name_or_path": "first", "lora_mode": "merge",
                             "loras": [{"path": path}]}, {})
        engine._ensure_holder(job)
        build = engine._build_holder

        def replace_after_pool_selection(job, model):
            result = build(job, model)
            self.write_adapter(0.5, preserve_stat=True)
            return result

        job.model["name_or_path"] = "second"
        with mock.patch.object(engine, "_build_holder", replace_after_pool_selection):
            with self.assertRaisesRegex(ValueError, "source changed"):
                engine._ensure_holder(job)
        self.assertIsNone(engine.holder)
        self.assertEqual(engine.pool.entries, {})
        engine._ensure_holder(job)
        torch.testing.assert_close(engine.holder.model(torch.ones(1, 8)), torch.full((1, 8), 8.0))


if __name__ == "__main__":
    unittest.main()
