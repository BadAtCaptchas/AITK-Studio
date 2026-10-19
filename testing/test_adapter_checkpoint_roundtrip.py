"""Real adapter math and checkpoint continuation, without model downloads."""
import copy
import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from toolkit.base_lora import fuse_lora_network, load_lora_network_for_model
from toolkit.config_modules import NetworkConfig
from toolkit.ema import ExponentialMovingAverage
from toolkit.inference_lora import InferenceLoRA
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.network_mixins import ToolkitNetworkMixin
from toolkit.optimizer_checkpoint import (
    OptimizerCheckpointValidationError,
    atomic_save_optimizer_checkpoint,
    load_optimizer_checkpoint_for_resume,
)


class TinyRoot(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.proj(x)


class Holder:
    arch = "tiny"
    use_old_lokr_format = False
    is_xl = is_ssd = is_v2 = is_v3 = is_pixart = is_auraflow = False
    is_flux = is_lumina2 = is_vega = False

    def __init__(self, model, transformer=True):
        self.model = self.unet = model
        self.text_encoder = None
        self.is_transformer = transformer
        self.device_torch = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.model_config = SimpleNamespace()
        self.target_lora_modules = ["TinyRoot"]

    def get_transformer_block_names(self):
        return None

    def convert_lora_weights_before_load(self, state):
        return state

    def convert_lora_weights_before_save(self, state):
        return state

    def get_model_to_train(self):
        return self.model


def make_network(kind, base=None, rank=2, alpha=None, transformer=True, quantized=False, **kwargs):
    model = TinyRoot().requires_grad_(False)
    if base is not None:
        model.load_state_dict(base)
    if quantized:
        from optimum.quanto import freeze, qint8, quantize
        quantize(model, weights=qint8)
        freeze(model)
    holder = Holder(model, transformer)
    alpha = rank if alpha is None else alpha
    config = NetworkConfig(
        type=kind, linear=rank, linear_alpha=alpha,
        transformer_only=False, lokr_full_rank=False, **kwargs,
    )
    network = LoRASpecialNetwork(
        None, model, lora_dim=rank, alpha=alpha,
        train_unet=True, train_text_encoder=False,
        network_config=config, network_type=kind,
        is_transformer=transformer, base_model=holder,
        target_lin_modules=["TinyRoot"], transformer_only=False,
    )
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.force_to("cpu", torch.float32)
    network.is_active = True
    network._update_torch_multiplier()
    network.eval()
    return model, holder, network, network.get_all_modules()[0]


def clone_parameters(network):
    return {name: value.detach().clone() for name, value in network.named_parameters()}


def train_step(model, optimizer, inputs, target):
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(inputs), target)
    loss.backward()
    optimizer.step()
    return loss.detach()


class AdapterCheckpointRoundTripTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7001)
        self.inputs = torch.randn(2, 3, 8)
        self.target = torch.randn_like(self.inputs)
        root = Path(__file__).resolve().parents[1] / ".tmp"
        root.mkdir(exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(prefix="adapter-roundtrip-", dir=root)
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)

    def assert_parameters_equal(self, expected, network):
        actual = dict(network.named_parameters())
        self.assertEqual(expected.keys(), actual.keys())
        for name, value in expected.items():
            torch.testing.assert_close(value, actual[name], rtol=0, atol=0)

    def test_dora_magnitude_survives_peft_and_legacy_reload(self):
        for transformer in (True, False):
            with self.subTest(transformer=transformer):
                model, holder, network, adapter = make_network("dora", transformer=transformer)
                base = copy.deepcopy(model.state_dict())
                with torch.no_grad():
                    adapter.magnitude.mul_(2)
                    adapter.lora_up.weight.normal_(0, 0.05)
                expected = model(self.inputs).detach()
                path = str(self.path / "dora.safetensors")
                network.save_weights(path, dtype=torch.float32)
                other, other_holder, reloaded, other_adapter = make_network(
                    "dora", base, transformer=transformer,
                )
                self.assertIsNone(reloaded.load_weights(path))
                torch.testing.assert_close(other_adapter.magnitude, adapter.magnitude, rtol=0, atol=0)
                torch.testing.assert_close(other(self.inputs), expected, rtol=0, atol=0)
                other(self.inputs).sum().backward()
                self.assertTrue(torch.isfinite(other_adapter.magnitude.grad).all())
                self.assertGreater(other_adapter.magnitude.grad.abs().sum().item(), 0)

    def test_legacy_alpha_refreshes_forward_and_base_lora_fusion(self):
        model, holder, network, adapter = make_network("lora", rank=2, alpha=1.5, transformer=False)
        base = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            adapter.lora_up.weight.fill_(0.3)
        expected = model(self.inputs).detach()
        path = str(self.path / "legacy.safetensors")
        network.save_weights(path, dtype=torch.float32)
        other, other_holder, reloaded, other_adapter = make_network(
            "lora", base, rank=2, alpha=2, transformer=False,
        )
        reloaded.load_weights(path)
        self.assertEqual(other_adapter.scale, 0.75)
        self.assertEqual(other_adapter._runtime_scale.item(), 0.75)
        torch.testing.assert_close(other(self.inputs), expected, rtol=0, atol=0)
        fused = TinyRoot()
        fused.load_state_dict(base)
        fused_holder = Holder(fused, transformer=False)
        loaded = load_lora_network_for_model(fused_holder, path=path)
        self.assertEqual(loaded.network.get_all_modules()[0].scale, 0.75)
        fuse_lora_network(loaded, strength=1.0, base_model=fused_holder)
        torch.testing.assert_close(fused(self.inputs), expected, rtol=1e-6, atol=1e-7)

    def test_quantized_base_reload_preserves_outputs_and_adapter_gradients(self):
        base = copy.deepcopy(TinyRoot().state_dict())
        for kind in ("dora", "lora"):
            with self.subTest(kind=kind):
                model, holder, network, adapter = make_network(
                    kind, base, alpha=1.5, transformer=kind == "dora", quantized=True,
                )
                with torch.no_grad():
                    adapter.lora_up.weight.fill_(0.2)
                    if kind == "dora":
                        adapter.magnitude.mul_(1.5)
                other, other_holder, reloaded, _ = make_network(
                    kind, base, alpha=2, transformer=kind == "dora", quantized=True,
                )
                reloaded.load_weights(network.get_state_dict(dtype=torch.float32))
                torch.testing.assert_close(other(self.inputs), model(self.inputs), rtol=0, atol=0)
                F.mse_loss(model(self.inputs), self.target).backward()
                F.mse_loss(other(self.inputs), self.target).backward()
                for name, parameter in network.named_parameters():
                    other_parameter = dict(reloaded.named_parameters())[name]
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    torch.testing.assert_close(other_parameter.grad, parameter.grad, rtol=0, atol=0)

    def test_full_factor_rs_exports_match_inference_and_repeat_exports(self):
        for scalar in (False, True):
            with self.subTest(scalar=scalar):
                kwargs = dict(lokr_factor=2, lokr_full_matrix=True, lokr_rs_lora=True, lokr_use_scalar=scalar)
                model, holder, network, adapter = make_network("lokr", rank=3, **kwargs)
                base = copy.deepcopy(model.state_dict())
                with torch.no_grad():
                    adapter.lokr_w1.fill_(0.2)
                    adapter.lokr_w2.fill_(0.3)
                    if scalar:
                        adapter.scalar.fill_(0.25)
                original_parameters = clone_parameters(network)
                expected = model(self.inputs).detach()
                exported = network.get_state_dict(dtype=torch.float32)
                self.assert_parameters_equal(original_parameters, network)
                self.assertEqual(next(value.item() for key, value in exported.items() if key.endswith(".alpha")), 1)
                path = str(self.path / "rs.safetensors")
                save_file(exported, path)
                inference_model = TinyRoot()
                inference_model.load_state_dict(base)
                inference = InferenceLoRA(path).load(Holder(inference_model)).attach()
                self.assertEqual(inference.unmatched, [])
                torch.testing.assert_close(inference_model(self.inputs), expected, rtol=1e-6, atol=2e-7)
                inference.detach()
                other, other_holder, reloaded, other_adapter = make_network("lokr", base, rank=3, **kwargs)
                self.assertIsNone(reloaded.load_weights(path))
                self.assertEqual(other_adapter.scale, 1.0)
                torch.testing.assert_close(other(self.inputs), expected, rtol=1e-6, atol=2e-7)
                repeated = reloaded.get_state_dict(dtype=torch.float32)
                for key in exported:
                    torch.testing.assert_close(repeated[key], exported[key], rtol=0, atol=0)
                # A fresh optimizer may continue from the portable parameterization.
                optimizer = torch.optim.AdamW(reloaded.parameters(), lr=0.01, foreach=False)
                train_step(other, optimizer, self.inputs, self.target)
                third, third_holder, third_network, _ = make_network("lokr", base, rank=3, **kwargs)
                third_network.load_weights(reloaded.get_state_dict(dtype=torch.float32))
                torch.testing.assert_close(third(self.inputs), other(self.inputs), rtol=1e-6, atol=2e-7)

    def test_legacy_full_factors_keep_rank_independent_scale(self):
        model, holder, network, adapter = make_network("lokr", rank=4, lokr_factor=2, lokr_full_matrix=True)
        base = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            adapter.lokr_w1.fill_(0.2)
            adapter.lokr_w2.fill_(0.3)
        exported = network.get_state_dict(dtype=torch.float32)
        self.assertEqual(next(value.item() for key, value in exported.items() if key.endswith(".alpha")), 4)
        other, other_holder, reloaded, other_adapter = make_network(
            "lokr", base, rank=1, lokr_factor=2, lokr_full_matrix=True,
        )
        reloaded.load_weights(exported)
        self.assertEqual(other_adapter.scale, 1.0)
        torch.testing.assert_close(other(self.inputs), model(self.inputs), rtol=0, atol=0)

    def test_legacy_full_rs_can_be_reexported_with_original_training_config(self):
        kwargs = dict(lokr_factor=2, lokr_full_matrix=True, lokr_rs_lora=True)
        model, holder, network, adapter = make_network("lokr", rank=4, **kwargs)
        base = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            adapter.lokr_w1.fill_(0.2)
            adapter.lokr_w2.fill_(0.3)
        legacy = {
            "transformer.proj.lokr_w1": adapter.lokr_w1.detach().clone(),
            "transformer.proj.lokr_w2": adapter.lokr_w2.detach().clone(),
            "transformer.proj.alpha": adapter.alpha.detach().clone(),
        }
        self.assertEqual(legacy["transformer.proj.alpha"].item(), 8.0)
        other, other_holder, reloaded, other_adapter = make_network("lokr", base, rank=4, **kwargs)
        reloaded.load_weights(legacy)
        self.assertEqual(other_adapter.scale, 2.0)
        reloaded.restore_optimizer_resume_state(None)
        torch.testing.assert_close(other(self.inputs), model(self.inputs), rtol=0, atol=0)
        path = str(self.path / "migrated.safetensors")
        reloaded.save_weights(path, dtype=torch.float32)
        canonical, canonical_holder, canonical_network, _ = make_network("lokr", base, rank=4, **kwargs)
        canonical_network.load_weights(path)
        with self.assertRaisesRegex(OptimizerCheckpointValidationError, "no raw training parameters"):
            canonical_network.restore_optimizer_resume_state(None)
        inference_model = TinyRoot()
        inference_model.load_state_dict(base)
        inference = InferenceLoRA(path).load(Holder(inference_model)).attach()
        try:
            torch.testing.assert_close(inference_model(self.inputs), model(self.inputs), rtol=1e-6, atol=1e-7)
        finally:
            inference.detach()

    def test_raw_resume_has_identical_next_adam_step_for_scalar_and_rs(self):
        for scalar, rs, rank, full in ((True, False, 1, False), (False, True, 3, True), (True, True, 3, True)):
            with self.subTest(scalar=scalar, rs=rs):
                kwargs = dict(lokr_factor=2, lokr_full_matrix=full, lokr_rs_lora=rs, lokr_use_scalar=scalar)
                model, holder, network, adapter = make_network("lokr", rank=rank, **kwargs)
                base = copy.deepcopy(model.state_dict())
                with torch.no_grad():
                    for parameter in network.parameters():
                        parameter.fill_(0.2)
                    if scalar:
                        adapter.scalar.fill_(0.25)
                optimizer = torch.optim.AdamW(network.parameters(), lr=0.03, weight_decay=0.01, foreach=False)
                train_step(model, optimizer, self.inputs, self.target)
                raw = network.get_optimizer_resume_state()
                path = str(self.path / "adapter.safetensors")
                # Even a low-precision portable export must not round raw resume weights.
                network.save_weights(path, dtype=torch.bfloat16)
                optimizer_path = str(self.path / "optimizer.pt")
                atomic_save_optimizer_checkpoint(optimizer_path, optimizer.state_dict(), 1, path, raw)
                other, other_holder, reloaded, _ = make_network("lokr", base, rank=rank, **kwargs)
                reloaded.load_weights(path)
                other_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=0.03, weight_decay=0.01, foreach=False)
                moments, raw_loaded = load_optimizer_checkpoint_for_resume(optimizer_path, 1, path)
                reloaded.restore_optimizer_resume_state(raw_loaded)
                other_optimizer.load_state_dict(moments)
                self.assert_parameters_equal(clone_parameters(network), reloaded)
                torch.testing.assert_close(other(self.inputs), model(self.inputs), rtol=0, atol=0)
                train_step(model, optimizer, self.inputs, self.target)
                train_step(other, other_optimizer, self.inputs, self.target)
                self.assert_parameters_equal(clone_parameters(network), reloaded)
                torch.testing.assert_close(other(self.inputs), model(self.inputs), rtol=0, atol=0)

    def test_raw_resume_rejects_missing_malformed_and_changed_topology_without_mutation(self):
        model, holder, network, _ = make_network("lokr", rank=1, lokr_factor=2, lokr_use_scalar=True)
        raw = network.get_optimizer_resume_state()
        originals = clone_parameters(network)
        bad_states = [None]
        for kind in ("shape", "name", "nan", "config", "scale"):
            bad = copy.deepcopy(raw)
            name = next(iter(bad["parameters"]))
            if kind == "shape":
                bad["parameters"][name] = torch.ones(13)
            elif kind == "name":
                bad["parameters"]["unexpected"] = bad["parameters"].pop(name)
            elif kind == "nan":
                bad["parameters"][name].fill_(float("nan"))
            elif kind == "config":
                next(iter(bad["modules"].values()))["scalar"] = False
            else:
                next(iter(bad["scales"].values()))["runtime_scale"] = float("inf")
            bad_states.append(bad)
        for bad in bad_states:
            with self.assertRaises(OptimizerCheckpointValidationError):
                network.restore_optimizer_resume_state(bad)
            self.assert_parameters_equal(originals, network)
        ordinary, ordinary_holder, ordinary_network, _ = make_network("lokr", rank=1, lokr_factor=2)
        self.assertIsNone(ordinary_network.get_optimizer_resume_state())
        ordinary_network.restore_optimizer_resume_state(None)
        with self.assertRaises(OptimizerCheckpointValidationError):
            ordinary_network.restore_optimizer_resume_state(raw)
        # Other mixin clients, including LyCORIS, need not define network_type.
        other_network = ToolkitNetworkMixin()
        self.assertIsNone(other_network.get_optimizer_resume_state())
        other_network.restore_optimizer_resume_state(None)

    def test_optimizer_raw_state_is_bound_to_file_contents_and_accepts_rename(self):
        model, holder, network, _ = make_network("lokr", rank=1, lokr_factor=2, lokr_use_scalar=True)
        model_path = self.path / "adapter.safetensors"
        network.save_weights(str(model_path), dtype=torch.float32)
        optimizer = torch.optim.AdamW(network.parameters())
        optimizer_path = str(self.path / "optimizer.pt")
        atomic_save_optimizer_checkpoint(optimizer_path, optimizer.state_dict(), 7, model_path, network.get_optimizer_resume_state())
        renamed = self.path / "renamed.safetensors"
        model_path.rename(renamed)
        _, raw = load_optimizer_checkpoint_for_resume(optimizer_path, 7, renamed)
        self.assertIsNotNone(raw)
        changed = load_file(str(renamed))
        next(value for key, value in changed.items() if "lokr_w" in key).add_(0.1)
        save_file(changed, str(renamed))
        with self.assertRaisesRegex(OptimizerCheckpointValidationError, "does not match"):
            load_optimizer_checkpoint_for_resume(optimizer_path, 7, renamed)

    def test_training_process_saves_raw_parameters_before_ema_and_restores_moments(self):
        process_module = importlib.import_module("jobs.process.BaseSDTrainProcess")
        process_class = process_module.BaseSDTrainProcess
        model, holder, network, adapter = make_network("lokr", rank=1, lokr_factor=2, lokr_use_scalar=True)
        base = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.fill_(0.2)
            adapter.scalar.fill_(0.25)
        ema = ExponentialMovingAverage(network.parameters(), decay=0.9)
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.03, foreach=False)
        train_step(model, optimizer, self.inputs, self.target)
        originals = clone_parameters(network)
        process = SimpleNamespace(
            accelerator=SimpleNamespace(is_main_process=True), network=network,
            optimizer=optimizer, ema=ema, save_root=str(self.path), step_num=1,
            job=SimpleNamespace(name="ema_adapter"), meta={}, adapter=None,
            is_fine_tuning=False, train_config=SimpleNamespace(merge_network_on_save=False),
            named_lora=False, embedding=None, decorator=None, snr_gos=None,
            save_config=SimpleNamespace(dtype="float32"), update_training_metadata=lambda: None,
            clean_up_saves=lambda: None, post_save_hook=lambda path: None,
        )
        with mock.patch.object(process_module, "flush"):
            process_class.save(process, step=1)
        self.assert_parameters_equal(originals, network)
        model_path = str(self.path / "ema_adapter_000000001.safetensors")
        other, other_holder, reloaded, _ = make_network("lokr", base, rank=1, lokr_factor=2, lokr_use_scalar=True)
        reloaded.load_weights(model_path)
        self.assertFalse(torch.equal(other(self.inputs), model(self.inputs)))
        other_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=0.03, foreach=False)
        resume = SimpleNamespace(
            save_root=str(self.path), optimizer=other_optimizer, network=reloaded,
            _resume_checkpoint_step=1, _resume_checkpoint_path=model_path,
        )
        with mock.patch.object(process_module, "flush"):
            process_class.load_optimizer_state(resume)
        self.assert_parameters_equal(originals, reloaded)
        self.assertTrue(other_optimizer.state)
        train_step(model, optimizer, self.inputs, self.target)
        train_step(other, other_optimizer, self.inputs, self.target)
        self.assert_parameters_equal(clone_parameters(network), reloaded)

    def test_training_process_warns_and_skips_legacy_scalar_moments(self):
        process_module = importlib.import_module("jobs.process.BaseSDTrainProcess")
        model, holder, network, _ = make_network("lokr", rank=1, lokr_factor=2, lokr_use_scalar=True)
        base = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.03, foreach=False)
        train_step(model, optimizer, self.inputs, self.target)
        model_path = str(self.path / "legacy.safetensors")
        network.save_weights(model_path, dtype=torch.float32)
        atomic_save_optimizer_checkpoint(str(self.path / "optimizer.pt"), optimizer.state_dict(), 1, model_path)
        other, other_holder, reloaded, _ = make_network("lokr", base, rank=1, lokr_factor=2, lokr_use_scalar=True)
        reloaded.load_weights(model_path)
        originals = clone_parameters(reloaded)
        other_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=0.03, foreach=False)
        resume = SimpleNamespace(
            save_root=str(self.path), optimizer=other_optimizer, network=reloaded,
            _resume_checkpoint_step=1, _resume_checkpoint_path=model_path,
        )
        with mock.patch.object(process_module, "print_acc") as printed:
            process_module.BaseSDTrainProcess.load_optimizer_state(resume)
        self.assertEqual(len(other_optimizer.state), 0)
        self.assert_parameters_equal(originals, reloaded)
        self.assertTrue(any("no raw training parameters" in str(call) for call in printed.call_args_list))


if __name__ == "__main__":
    unittest.main()
