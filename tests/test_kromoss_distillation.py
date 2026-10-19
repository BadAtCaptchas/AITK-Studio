import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from safetensors.torch import load_file
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from extensions_built_in.diffusion_models.moss_image.moss_image import MossImageModel
from extensions_built_in.diffusion_models.moss_image.src.distillation import (
    FrozenMossDistillationAdapter,
    get_distillation_adapter,
)
from extensions_built_in.diffusion_models.moss_image.src.transformer import (
    MossImageParams,
    MossImageTransformer,
)
from extensions_built_in.diffusion_models.moss_image.src.pipeline import MossImagePipeline, moss_sigmas
from toolkit.config_modules import ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork


def tiny_transformer():
    return MossImageTransformer(MossImageParams(
        dim=16, ffn_dim=32, num_heads=2, num_layers=2,
        text_dim=12, freq_dim=16, latent_channels=4,
    ))


def fixed_weights(model):
    weights = {}
    for index, layer in enumerate(model.layers):
        for projection in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
            target = layer.proj["qkv" if projection in ("q", "k", "v") else projection]
            out_features = target.out_features // 3 if projection in ("q", "k", "v") else target.out_features
            prefix = f"diffusion_model.layers.{index}.proj.{projection}."
            rank = {"q": 3, "k": 4, "v": 5}.get(projection, 3)
            weights[prefix + "lora_down.weight"] = torch.randn(rank, target.in_features) * .1
            weights[prefix + "lora_up.weight"] = torch.randn(out_features, rank) * .03
            weights[prefix + "alpha"] = torch.tensor(float(rank - 1))
    return weights


def reference_delta(weights, index, projection, inputs):
    prefix = f"diffusion_model.layers.{index}.proj.{projection}."
    down = weights[prefix + "lora_down.weight"].to(inputs.device)
    up = weights[prefix + "lora_up.weight"].to(inputs.device)
    scale = float(weights[prefix + "alpha"]) / down.shape[0]
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        return (scale * (inputs.float() @ down.T)) @ up.T


def user_network(model, holder, kind, *, rank=2, full_matrix=False):
    config = NetworkConfig(
        type=kind, linear=rank, linear_alpha=rank, transformer_only=True,
        lokr_factor=2, lokr_full_rank=False, lokr_full_matrix=full_matrix,
    )
    network = LoRASpecialNetwork(
        text_encoder=None, unet=model, lora_dim=rank, alpha=rank,
        train_unet=True, train_text_encoder=False, network_config=config,
        network_type=kind, is_transformer=True, base_model=holder,
        target_lin_modules=["MossImageTransformer"], transformer_only=True,
    )
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.force_to("cpu", dtype=torch.float32)
    network.is_active = True
    network._update_torch_multiplier()
    return network


class KromossDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(935)

    def test_all_projection_updates_match_split_fp32_reference_under_autocast(self):
        model = tiny_transformer().to(dtype=torch.bfloat16)
        state = fixed_weights(model)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        controller = FrozenMossDistillationAdapter.attach(model, state, identity="fixture")
        self.addCleanup(controller.close)
        for index, layer in enumerate(model.layers):
            for projection in ("qkv", "o", "ffn_up_gate", "ffn_down"):
                with self.subTest(layer=index, projection=projection):
                    linear = layer.proj[projection]
                    inputs = torch.randn(2, 5, linear.in_features, dtype=torch.bfloat16, requires_grad=True)
                    names = ("q", "k", "v") if projection == "qkv" else (projection,)
                    float_inputs = inputs.float()
                    parts = [reference_delta(state, index, name, float_inputs) for name in names]
                    delta = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
                    expected = F.linear(inputs, linear.weight) + delta.to(torch.bfloat16)
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        actual = linear(inputs)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    gradient = torch.randn_like(actual)
                    expected_grad = torch.autograd.grad(expected, inputs, gradient)[0]
                    actual_grad = torch.autograd.grad(actual, inputs, gradient)[0]
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        self.assertEqual(set(before), set(model.state_dict()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_identity_reuse_freeze_source_ownership_and_cleanup(self):
        model = tiny_transformer()
        state = fixed_weights(model)
        baseline = model.layers[0].proj["qkv"](torch.ones(1, 16)).detach()
        controller = FrozenMossDistillationAdapter.attach(model, state, identity="one")
        self.assertIs(FrozenMossDistillationAdapter.attach(model, {}, identity="one"), controller)
        self.assertEqual(len(model.layers[0].proj["qkv"]._forward_hooks), 1)
        with self.assertRaisesRegex(ValueError, "different"):
            FrozenMossDistillationAdapter.attach(model, state, identity="two")
        fixed = controller._targets["layers.0.proj.qkv"][0]
        before = fixed.down.clone()
        state["diffusion_model.layers.0.proj.q.lora_down.weight"].zero_()
        model.to(dtype=torch.bfloat16).requires_grad_(True)
        self.assertEqual(fixed.down.dtype, torch.float32)
        self.assertFalse(fixed.down.requires_grad)
        torch.testing.assert_close(fixed.down, before, rtol=0, atol=0)
        parameter_ids = {id(p) for p in model.parameters()}
        self.assertNotIn(id(fixed.down), parameter_ids)
        self.assertNotIn(id(fixed.up), parameter_ids)
        with self.assertRaisesRegex(ValueError, "meta"):
            controller.to("meta")
        controller.close()
        controller.close()
        self.assertIsNone(get_distillation_adapter(model))
        self.assertFalse(controller._targets)
        self.assertFalse(model.layers[0].proj["qkv"]._forward_hooks)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            controller.to("cpu")
        # Closing removes the fixed update without changing the base weights.
        actual = model.layers[0].proj["qkv"](torch.ones(1, 16, dtype=torch.bfloat16)).float()
        torch.testing.assert_close(actual, baseline, rtol=.01, atol=.01)

    def test_malformed_weights_fail_before_any_hooks_are_installed(self):
        model = tiny_transformer()
        state = fixed_weights(model)
        prefix = "diffusion_model.layers.1.proj.ffn_down."
        cases = {
            "missing": lambda s: s.pop(prefix + "alpha"),
            "unexpected": lambda s: s.update({"extra": torch.ones(1)}),
            "dtype": lambda s: s.update({prefix + "lora_down.weight": s[prefix + "lora_down.weight"].bfloat16()}),
            "non_tensor": lambda s: s.update({prefix + "alpha": 3.0}),
            "nan": lambda s: s[prefix + "lora_up.weight"].fill_(float("nan")),
            "inf": lambda s: s.update({prefix + "alpha": torch.tensor(float("inf"))}),
            "alpha_shape": lambda s: s.update({prefix + "alpha": torch.ones(1)}),
            "alpha_sign": lambda s: s.update({prefix + "alpha": torch.tensor(-1.)}),
            "rank": lambda s: s.update({prefix + "lora_up.weight": torch.ones(16, 2)}),
            "empty_rank": lambda s: s.update({prefix + "lora_down.weight": torch.empty(0, 32)}),
            "input_shape": lambda s: s.update({prefix + "lora_down.weight": torch.ones(3, 31)}),
        }
        for name, corrupt in cases.items():
            with self.subTest(name=name):
                changed = {k: v.clone() for k, v in state.items()}
                corrupt(changed)
                with self.assertRaises(ValueError):
                    FrozenMossDistillationAdapter.attach(model, changed, identity=name)
                self.assertIsNone(get_distillation_adapter(model))
                self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_eight_step_cfg_one_uses_only_positive_branch_and_fp32_euler(self):
        class Transformer:
            params = SimpleNamespace(latent_channels=4)

            def __init__(self):
                self.calls = []

            def __call__(self, latents, timestep, context, mask, **kwargs):
                self.calls.append((latents.clone(), timestep.clone(), context.clone(), kwargs))
                return (latents * .125 + timestep.reshape(1, 1, 1, 1) / 10000).bfloat16()

        transformer = Transformer()
        decoded = []

        def decode(latents, **kwargs):
            decoded.append(latents.clone())
            return torch.zeros(1, 3, 32, 32)

        model = SimpleNamespace(
            device_torch=torch.device("cpu"), torch_dtype=torch.bfloat16,
            transformer=transformer, decode_latents=decode,
        )
        positive = SimpleNamespace(text_embeds=[torch.full((3, 12), 1.)])
        negative = SimpleNamespace(text_embeds=[torch.full((3, 12), -1.)])
        initial = torch.randn(1, 4, 2, 2)
        MossImagePipeline(model)(
            conditional_embeds=positive, unconditional_embeds=negative,
            height=32, width=32, latents=initial.clone(),
            num_inference_steps=8, guidance_scale=1.,
        )
        sigmas = moss_sigmas(8)
        expected = initial.clone()
        self.assertEqual(len(transformer.calls), 8)
        for index, (latents, timestep, context, kwargs) in enumerate(transformer.calls):
            torch.testing.assert_close(latents, expected, rtol=0, atol=0)
            self.assertEqual(latents.dtype, torch.float32)
            self.assertTrue(bool((context == 1).all()))
            self.assertNotIn("skip_layers", kwargs)
            torch.testing.assert_close(timestep, (sigmas[index] * 1000).reshape(1), rtol=0, atol=0)
            prediction = (expected * .125 + timestep.reshape(1, 1, 1, 1) / 10000).bfloat16().float()
            expected = expected + (sigmas[index + 1] - sigmas[index]) * prediction
        torch.testing.assert_close(decoded[0], expected, rtol=0, atol=0)

    def test_lora_and_lokr_train_and_export_without_frozen_adapter_weights(self):
        for kind in ("lora", "lokr"):
            with self.subTest(kind=kind):
                model = tiny_transformer().requires_grad_(False)
                pristine = copy.deepcopy(model.state_dict())
                fixed = fixed_weights(model)
                controller = FrozenMossDistillationAdapter.attach(model, fixed, identity=kind)
                self.addCleanup(controller.close)
                holder = MossImageModel("cpu", ModelConfig(arch="moss_image", name_or_path="unused"), dtype="fp32")
                holder.model = model
                network = user_network(model, holder, kind)
                if kind == "lokr":
                    # Exercise a saved adapter whose alpha differs from the
                    # fresh network's construction default. This is common
                    # when importing externally trained LoKr adapters.
                    for module in network.get_all_modules():
                        module.alpha.fill_(1.0)
                        module._set_runtime_scale(.5)
                model.enable_gradient_checkpointing()
                inputs = torch.randn(2, 4, 2, 2, requires_grad=True)
                times = torch.tensor([800., 300.])
                context = torch.randn(2, 3, 12)
                mask = torch.tensor([[True, True, True], [True, True, False]])
                optimizer = torch.optim.SGD(network.parameters(), lr=.01)
                prediction = model(inputs, times, context, mask)
                prediction.square().mean().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())
                grads = [p.grad for p in network.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
                for deltas in controller._targets.values():
                    for delta in deltas:
                        self.assertIsNone(delta.down.grad)
                        self.assertIsNone(delta.up.grad)
                        self.assertFalse(delta.down.requires_grad)
                optimizer.step()
                with torch.no_grad():
                    expected = model(inputs, times, context, mask)
                with tempfile.TemporaryDirectory() as directory:
                    path = str(Path(directory) / "user.safetensors")
                    network.save_weights(path, dtype=torch.float32)
                    exported = load_file(path)
                self.assertTrue(exported)
                if kind == "lora":
                    self.assertTrue(all(v.shape[0] == 2 for k, v in exported.items() if "lora_A" in k))
                else:
                    self.assertTrue(all("lokr" in k or k.endswith(".alpha") for k in exported))
                restored = tiny_transformer().requires_grad_(False)
                restored.load_state_dict(pristine)
                second = FrozenMossDistillationAdapter.attach(restored, fixed, identity=kind)
                self.addCleanup(second.close)
                restored_holder = MossImageModel("cpu", ModelConfig(arch="moss_image", name_or_path="unused"), dtype="fp32")
                restored_holder.model = restored
                restored_network = user_network(restored, restored_holder, kind)
                restored_network.load_weights(exported)
                if kind == "lokr":
                    self.assertTrue(all(module.scale == .5 for module in restored_network.get_all_modules()))
                with torch.no_grad():
                    actual = restored(inputs, times, context, mask)
                torch.testing.assert_close(actual, expected)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, pristine[key], rtol=0, atol=0)

    def test_full_matrix_lokr_load_preserves_scale_when_configured_rank_changes(self):
        source = tiny_transformer().requires_grad_(False)
        restored = copy.deepcopy(source)
        source_holder = MossImageModel("cpu", ModelConfig(arch="moss_image", name_or_path="unused"), dtype="fp32")
        restored_holder = MossImageModel("cpu", ModelConfig(arch="moss_image", name_or_path="unused"), dtype="fp32")
        source_holder.model, restored_holder.model = source, restored
        source_network = user_network(source, source_holder, "lokr", rank=2, full_matrix=True)
        restored_network = user_network(restored, restored_holder, "lokr", rank=4, full_matrix=True)
        for module in source_network.get_all_modules():
            self.assertTrue(module.use_w1 and module.use_w2)
            with torch.no_grad():
                module.lokr_w2.normal_(std=.1)
        restored_network.load_weights(source_network.get_state_dict(dtype=torch.float32))
        self.assertTrue(all(module.scale == 1 for module in restored_network.get_all_modules()))
        inputs, times, context = torch.randn(1, 4, 2, 2), torch.tensor([500.]), torch.randn(1, 3, 12)
        with torch.no_grad():
            torch.testing.assert_close(restored(inputs, times, context), source(inputs, times, context))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU offload coverage")
    def test_cuda_offload_checkpoint_and_fp32_device_lifecycle(self):
        from toolkit.memory_management.manager_modules import LinearLayerMemoryManager
        from toolkit.memory_management.manager import MemoryManager

        model = tiny_transformer().bfloat16().cuda().requires_grad_(False)
        fixed = fixed_weights(model)
        linear = model.layers[0].proj["qkv"]
        inputs = torch.randn(1, 7, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        baseline = linear(inputs).detach()
        manager = MemoryManager(model, torch.device("cuda"))
        LinearLayerMemoryManager.attach(linear, manager)
        self.assertEqual(linear.weight.device.type, "cpu")
        controller = FrozenMossDistillationAdapter.attach(model, fixed, identity="offload")
        self.addCleanup(controller.close)
        updates = torch.cat([reference_delta(fixed, 0, p, inputs) for p in ("q", "k", "v")], -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = checkpoint(linear, inputs, use_reentrant=False)
        torch.testing.assert_close(result, baseline + updates.to(baseline.dtype), rtol=0, atol=0)
        result.float().square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(float(inputs.grad.abs().sum()), 0)
        self.assertEqual(controller._targets["layers.0.proj.qkv"][0].down.device.type, "cuda")
        controller.to("cpu")
        self.assertTrue(all(delta.down.device.type == "cpu" for values in controller._targets.values() for delta in values))
        model.to(dtype=torch.bfloat16)
        self.assertTrue(all(delta.down.dtype == torch.float32 for values in controller._targets.values() for delta in values))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for ConvRot coverage")
    def test_quantized_projection_hook_uses_original_input_basis(self):
        from toolkit.util.convrot_quant import get_convrot_quantizer
        from toolkit.util.ostris_quant import convert_linear_to_ostris

        model = tiny_transformer().bfloat16().cuda().requires_grad_(False)
        fixed = fixed_weights(model)
        linear = model.layers[0].proj["qkv"]
        self.assertTrue(convert_linear_to_ostris(linear, get_convrot_quantizer("convrot8", kernel="torch")))
        inputs = torch.randn(1, 7, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        baseline = linear(inputs).detach()
        controller = FrozenMossDistillationAdapter.attach(model, fixed, identity="convrot")
        self.addCleanup(controller.close)
        delta = torch.cat([reference_delta(fixed, 0, p, inputs) for p in ("q", "k", "v")], -1)
        result = linear(inputs)
        torch.testing.assert_close(result, baseline + delta.to(baseline.dtype), rtol=0, atol=0)
        result.float().square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(float(inputs.grad.abs().sum()), 0)
        model.to("cpu")
        self.assertTrue(all(value.down.device.type == "cpu" for parts in controller._targets.values() for value in parts))
        model.to("meta")
        self.assertIsNone(get_distillation_adapter(model))
        self.assertFalse(controller._targets)


if __name__ == "__main__":
    unittest.main()
