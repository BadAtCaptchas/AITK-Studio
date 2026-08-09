import ast
import contextlib
import inspect
import math
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from toolkit.memory_management import manager_modules
from toolkit.util import convrot_quant


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DFE_PATH = PROJECT_ROOT / "toolkit" / "models" / "diffusion_feature_extraction.py"


def _load_dfe6_test_class():
    source = DFE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(DFE_PATH))
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DiffusionFeatureExtractor6"
    )
    methods = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"prepare_inputs", "forward"}
    ]
    fold_frames = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_fold_frames_to_batch"
    )
    test_class = ast.ClassDef(
        name="DiffusionFeatureExtractor6ForTest",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    test_module = ast.Module(body=[fold_frames, test_class], type_ignores=[])
    ast.fix_missing_locations(test_module)
    namespace = {
        "torch": torch,
        "F": F,
        "math": math,
        "DataLoaderBatchDTO": object,
        "CustomFlowMatchEulerDiscreteScheduler": object,
    }
    exec(compile(test_module, str(DFE_PATH), "exec"), namespace)
    return namespace["DiffusionFeatureExtractor6ForTest"], class_node


class ConvRotCompatibilityTest(unittest.TestCase):
    def setUp(self):
        convrot_quant._hadamard_cache.clear()
        convrot_quant._e4m3_triton_ok_cache.clear()

    def test_regular_hadamard_uses_an_fp32_cpu_seed(self):
        calls = []
        torch_tensor = torch.tensor

        def capture_tensor(data, *args, **kwargs):
            calls.append(kwargs.copy())
            return torch_tensor(data, *args, **kwargs)

        with mock.patch.object(
            convrot_quant.torch, "tensor", side_effect=capture_tensor
        ):
            matrix = convrot_quant.regular_hadamard(
                64, torch.device("cpu"), torch.float32
            )

        self.assertEqual(calls[0]["dtype"], torch.float32)
        self.assertEqual(calls[0]["device"], "cpu")
        self.assertEqual(matrix.dtype, torch.float32)
        self.assertTrue(torch.equal(matrix @ matrix, torch.eye(64)))

    def test_e4m3_triton_gate_is_cached_and_fails_closed(self):
        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 0)
        ) as capability:
            self.assertFalse(convrot_quant._e4m3_triton_ok("cuda:0"))
            self.assertFalse(convrot_quant._e4m3_triton_ok("cuda:0"))
            capability.assert_called_once_with("cuda:0")

        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 9)
        ):
            self.assertTrue(convrot_quant._e4m3_triton_ok("cuda:1"))

        with mock.patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=RuntimeError("unavailable"),
        ):
            self.assertFalse(convrot_quant._e4m3_triton_ok("cuda:2"))

        dequant_source = inspect.getsource(convrot_quant.dequantize_nvfp4)
        self.assertIn("_e4m3_triton_ok(packed.device)", dequant_source)

    def test_lazy_triton_import_publishes_names_for_old_jit_resolvers(self):
        fake_triton = types.ModuleType("triton")
        fake_triton.__path__ = []
        fake_language = types.ModuleType("triton.language")
        fake_triton.language = fake_language
        old_triton = getattr(convrot_quant, "triton", None)
        old_tl = getattr(convrot_quant, "tl", None)
        had_triton = hasattr(convrot_quant, "triton")
        had_tl = hasattr(convrot_quant, "tl")
        try:
            with mock.patch.dict(
                sys.modules,
                {"triton": fake_triton, "triton.language": fake_language},
            ):
                imported = convrot_quant._import_triton()
            self.assertEqual(imported, (fake_triton, fake_language))
            self.assertIs(convrot_quant.triton, fake_triton)
            self.assertIs(convrot_quant.tl, fake_language)
        finally:
            if had_triton:
                convrot_quant.triton = old_triton
            else:
                convrot_quant.__dict__.pop("triton", None)
            if had_tl:
                convrot_quant.tl = old_tl
            else:
                convrot_quant.__dict__.pop("tl", None)

    def test_every_lazy_kernel_builder_uses_the_compat_import(self):
        for name in (
            "_get_kernel",
            "_get_dequant_kernel",
            "_get_int8_kernels",
            "_get_intn_kernel",
            "_get_intn_grouped_kernel",
            "_get_bitnet_kernel",
            "_get_int_gemv_kernel",
        ):
            with self.subTest(builder=name):
                self.assertIn(
                    "_import_triton()",
                    inspect.getsource(getattr(convrot_quant, name)),
                )


class DiffusionFeatureExtractionOptimizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dfe_class, cls.dfe_ast = _load_dfe6_test_class()

    def _build_dfe(self):
        dfe = self.dfe_class()
        dfe.processor = SimpleNamespace(
            do_normalize=True,
            do_rescale=True,
            image_mean=[0.5, 0.4, 0.3],
            image_std=[0.25, 0.5, 0.2],
        )
        dfe.image_mean = torch.tensor(dfe.processor.image_mean).view(1, 3, 1, 1)
        dfe.image_std = torch.tensor(dfe.processor.image_std).view(1, 3, 1, 1)
        dfe.losses = {}
        dfe.log_every = 100
        dfe.step = 0
        return dfe

    def test_prepare_inputs_reuses_cached_constants_with_same_numerics(self):
        dfe = self._build_dfe()
        value = torch.linspace(-0.2, 1.2, 3 * 16 * 16).view(1, 3, 16, 16)
        actual = dfe.prepare_inputs(value)["pixel_values"]
        expected = (
            value.clamp(0.0, 1.0) - dfe.image_mean
        ) / dfe.image_std
        self.assertTrue(torch.equal(actual, expected))

        prepare_node = next(
            node
            for node in self.dfe_ast.body
            if isinstance(node, ast.FunctionDef) and node.name == "prepare_inputs"
        )
        init_node = next(
            node
            for node in self.dfe_ast.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        prepare_source = ast.unparse(prepare_node)
        init_source = ast.unparse(init_node)
        self.assertNotIn(".item()", prepare_source)
        self.assertNotIn("torch.tensor", prepare_source)
        self.assertIn("self.image_mean = torch.tensor", init_source)
        self.assertIn("self.image_std = torch.tensor", init_source)

    def test_loss_accounting_stays_on_device_until_logging(self):
        dfe = self._build_dfe()

        class FakeVae:
            device = torch.device("cpu")
            dtype = torch.float32
            config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

            @staticmethod
            def decode(latents):
                return latents

        class FakeVision:
            @staticmethod
            def __call__(pixel_values):
                return SimpleNamespace(pooler_output=pixel_values.flatten(1))

        dfe.vae = FakeVae()
        dfe.model = FakeVision()
        prediction = torch.linspace(-0.8, 0.8, 3 * 16 * 16).view(
            1, 3, 16, 16
        ).requires_grad_(True)
        batch = SimpleNamespace(
            tensor=torch.linspace(-1.0, 1.0, 3 * 16 * 16).view(1, 3, 16, 16)
        )
        loss = dfe.forward(
            noise=torch.zeros_like(prediction),
            noise_pred=prediction,
            noisy_latents=torch.zeros_like(prediction),
            timesteps=torch.tensor([500.0]),
            batch=batch,
            scheduler=None,
            model=SimpleNamespace(x0_pred=True),
        )

        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(torch.is_tensor(dfe.losses["dinov3"]))
        self.assertFalse(dfe.losses["dinov3"].requires_grad)

        forward_node = next(
            node
            for node in self.dfe_ast.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        forward_source = ast.unparse(forward_node)
        self.assertIn("dino_loss.detach()", forward_source)
        self.assertNotIn("dino_loss.item()", forward_source)


class _FakeEvent:
    def __init__(self):
        self.record_count = 0
        self.synchronize_count = 0

    def record(self):
        self.record_count += 1

    def synchronize(self):
        self.synchronize_count += 1


class _FakeStream:
    def __init__(self):
        self.waited = []
        self.synchronize_count = 0

    def wait_event(self, event):
        self.waited.append(event)

    def synchronize(self):
        self.synchronize_count += 1


class _FakeAutogradContext:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class GradientTransferSynchronizationTest(unittest.TestCase):
    def _state(self):
        return {
            "transfer_grad_stream": _FakeStream(),
            "grad_compute_done": [_FakeEvent()],
            "grad_xfer_done": [_FakeEvent()],
        }

    def test_first_gradient_copy_remains_async(self):
        state = self._state()
        weight = torch.nn.Parameter(torch.ones(2, 2))
        with mock.patch.object(
            manager_modules.torch.cuda,
            "stream",
            return_value=contextlib.nullcontext(),
        ):
            grad_weight, _ = manager_modules._stage_grads_to_cpu(
                state, 0, torch.ones(2, 2), None, weight, None
            )
        self.assertIsNotNone(grad_weight)
        self.assertEqual(state["grad_xfer_done"][0].synchronize_count, 0)

    def test_accumulated_gradient_waits_for_its_async_copy(self):
        state = self._state()
        weight = torch.nn.Parameter(torch.ones(2, 2))
        weight.grad = torch.zeros_like(weight)
        with mock.patch.object(
            manager_modules.torch.cuda,
            "stream",
            return_value=contextlib.nullcontext(),
        ):
            manager_modules._stage_grads_to_cpu(
                state, 0, torch.ones(2, 2), None, weight, None
            )
        self.assertEqual(state["grad_xfer_done"][0].synchronize_count, 1)

    def test_public_sync_joins_every_pending_device_stream(self):
        first = _FakeStream()
        second = _FakeStream()
        original = manager_modules._DEVICE_STATE.copy()
        try:
            manager_modules._DEVICE_STATE.clear()
            manager_modules._DEVICE_STATE.update(
                {
                    torch.device("cuda:0"): {"transfer_grad_stream": first},
                    torch.device("cuda:1"): {"transfer_grad_stream": second},
                    torch.device("cpu"): {},
                }
            )
            manager_modules.sync_grad_transfers()
        finally:
            manager_modules._DEVICE_STATE.clear()
            manager_modules._DEVICE_STATE.update(original)

        self.assertEqual(first.synchronize_count, 1)
        self.assertEqual(second.synchronize_count, 1)

    def test_bounced_linear_and_conv_enter_the_process_device(self):
        device = torch.device("cuda:3")
        entered = []

        @contextlib.contextmanager
        def fake_device(selected):
            entered.append(selected)
            yield

        linear_context = _FakeAutogradContext()
        linear_input = torch.ones(1, 2)
        linear_weight = torch.eye(2)
        linear_bias = torch.zeros(2)
        with (
            mock.patch.object(manager_modules, "_get_device_state", return_value={}),
            mock.patch.object(
                manager_modules,
                "_stage_forward_weight",
                return_value=(0, linear_weight, linear_bias),
            ),
            mock.patch.object(manager_modules, "_release_forward_slot"),
            mock.patch.object(
                manager_modules.torch.cuda, "device", side_effect=fake_device
            ),
        ):
            linear_output = manager_modules._BouncingLinearFn.forward(
                linear_context,
                linear_input,
                linear_weight,
                linear_bias,
                None,
                device,
                None,
            )

        conv_context = _FakeAutogradContext()
        conv_input = torch.ones(1, 1, 3, 3)
        conv_weight = torch.ones(1, 1, 1, 1)
        conv_bias = torch.zeros(1)
        with (
            mock.patch.object(manager_modules, "_get_device_state", return_value={}),
            mock.patch.object(
                manager_modules,
                "_stage_forward_weight",
                return_value=(0, conv_weight, conv_bias),
            ),
            mock.patch.object(manager_modules, "_release_forward_slot"),
            mock.patch.object(
                manager_modules.torch.cuda, "device", side_effect=fake_device
            ),
        ):
            conv_output = manager_modules._BouncingConv2dFn.forward(
                conv_context,
                conv_input,
                conv_weight,
                conv_bias,
                device,
                (1, 1),
                (0, 0),
                (1, 1),
                1,
            )

        self.assertTrue(torch.equal(linear_output, linear_input))
        self.assertTrue(torch.equal(conv_output, conv_input))
        self.assertEqual(entered, [device, device])


if __name__ == "__main__":
    unittest.main()
