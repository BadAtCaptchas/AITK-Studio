import importlib.util
import unittest
from unittest import mock

import torch

from toolkit.kohya_lora import LoRAModule as KohyaLoRAModule
from toolkit.lora_special import LoRAModule
from toolkit.lycoris_special import LoConSpecialModule
from toolkit.memory_management import MemoryManager
from toolkit.models.DoRA import DoRAModule
from toolkit.models.lokr import LokrModule
from toolkit.util.quantize import get_qtype, quantize


class _Network:
    network_type = "lora"
    is_lorm = False
    is_active = True
    is_merged_in = False
    _multiplier = 1.0

    def __init__(self):
        self.torch_multiplier = torch.ones(1)


class _AuthenNetwork(_Network):
    def get_authenlora_rank_scale(self, rank, device, dtype):
        return torch.full((1, rank), 0.75, device=device, dtype=dtype)


def _linear(device=None, dtype=None):
    return torch.nn.Linear(
        8,
        8,
        bias=False,
        device=device,
        dtype=dtype,
    )


def _cuda_compile_available():
    return (
        torch.cuda.is_available()
        and importlib.util.find_spec("triton") is not None
    )


class AdapterScaleTest(unittest.TestCase):
    def test_adapters_keep_float_metadata_and_nonpersistent_runtime_buffer(self):
        network = _Network()
        modules = [
            LoRAModule(
                "lora_scale",
                _linear(),
                lora_dim=4,
                alpha=torch.tensor(8, dtype=torch.bfloat16),
                network=network,
            ),
            KohyaLoRAModule(
                "kohya_scale",
                _linear(),
                lora_dim=4,
                alpha=torch.tensor(8, dtype=torch.bfloat16),
            ),
            LoConSpecialModule(
                "locon_scale",
                _linear(),
                lora_dim=4,
                alpha=torch.tensor(8, dtype=torch.bfloat16),
                network=network,
            ),
            DoRAModule(
                "dora_scale",
                _linear(),
                lora_dim=4,
                alpha=torch.tensor(8, dtype=torch.bfloat16),
                network=network,
            ),
            LokrModule(
                "lokr_scale",
                _linear(),
                lora_dim=2,
                alpha=torch.tensor(4, dtype=torch.bfloat16),
                network=network,
            ),
            LokrModule(
                "lokr_rs_bypass_scale",
                _linear(),
                lora_dim=2,
                alpha=torch.tensor(4, dtype=torch.bfloat16),
                network=network,
                rs_lora=True,
                bypass_mode=True,
            ),
        ]

        for module in modules:
            with self.subTest(module=type(module).__name__):
                self.assertIs(type(module.scale), float)
                self.assertAlmostEqual(
                    module._runtime_scale.item(),
                    module.scale,
                    places=6,
                )
                self.assertNotIn("_runtime_scale", module.state_dict())
                self.assertFalse(module._runtime_scale.requires_grad)

                runtime_scale = module._runtime_scale
                module._set_runtime_scale(module.scale / 2)
                self.assertIs(module._runtime_scale, runtime_scale)
                self.assertAlmostEqual(
                    module._runtime_scale.item(),
                    module.scale,
                    places=6,
                )

    def test_extract_weight_synchronizes_runtime_scale(self):
        module = LoRAModule(
            "extract_scale",
            _linear(),
            lora_dim=4,
            alpha=torch.tensor(8, dtype=torch.bfloat16),
            network=_Network(),
        )
        runtime_scale = module._runtime_scale
        down = torch.randn(2, 8)
        up = torch.randn(8, 2)

        with mock.patch(
            "toolkit.network_mixins.extract_linear",
            return_value=(down, up, 2, None),
        ):
            module.extract_weight(extract_mode="fixed", extract_mode_param=2)

        self.assertIs(module._runtime_scale, runtime_scale)
        self.assertEqual(module.scale, 1.0)
        self.assertEqual(module._runtime_scale.item(), 1.0)

    def test_authenlora_forward_reads_runtime_scale(self):
        torch.manual_seed(3)
        network = _AuthenNetwork()
        original = _linear()
        module = LoRAModule(
            "authen_scale",
            original,
            lora_dim=4,
            alpha=8,
            network=network,
        )
        module.org_forward = original.forward
        with torch.no_grad():
            module.lora_up.weight.normal_()

        value = torch.randn(2, 8)
        base = original(value)
        original_delta = module(value) - base
        module._set_runtime_scale(module.scale / 2)
        updated_delta = module(value) - base

        torch.testing.assert_close(
            updated_delta,
            original_delta / 2,
        )

    def test_lokr_rebuild_and_bypass_paths_read_runtime_scale(self):
        torch.manual_seed(5)
        for bypass_mode in (False, True):
            with self.subTest(bypass_mode=bypass_mode):
                network = _Network()
                original = _linear()
                module = LokrModule(
                    "lokr_dynamic_scale",
                    original,
                    lora_dim=2,
                    alpha=4,
                    network=network,
                    rs_lora=True,
                    bypass_mode=bypass_mode,
                )
                module.org_forward = original.forward
                with torch.no_grad():
                    for parameter in module.parameters():
                        parameter.normal_()

                value = torch.randn(2, 8)
                base = original(value)
                original_delta = module._call_forward(value) - base
                module._set_runtime_scale(module.scale / 2)
                updated_delta = module._call_forward(value) - base

                torch.testing.assert_close(
                    updated_delta,
                    original_delta / 2,
                )

    def test_orbit_backed_lora_and_lokr_paths_read_runtime_scale(self):
        torch.manual_seed(6)
        original = torch.nn.Linear(64, 16, bias=False)
        quantize(original, weights=get_qtype('orbit4'))
        self.assertTrue(getattr(original, 'is_ostris_quantized', False))
        value = torch.randn(2, 64)
        base = original(value)

        network = _Network()
        lora = LoRAModule(
            'orbit_lora_scale',
            original,
            lora_dim=4,
            alpha=8,
            network=network,
        )
        lora.org_forward = original.forward
        with torch.no_grad():
            lora.lora_up.weight.normal_()
        original_delta = lora(value) - base
        lora._set_runtime_scale(lora.scale / 2)
        updated_delta = lora(value) - base
        torch.testing.assert_close(updated_delta, original_delta / 2)

        lokr = LokrModule(
            'orbit_lokr_scale',
            original,
            lora_dim=2,
            alpha=4,
            network=network,
            bypass_mode=True,
        )
        lokr.org_forward = original.forward
        with torch.no_grad():
            for parameter in lokr.parameters():
                parameter.normal_()
        original_delta = lokr._call_forward(value) - base
        lokr._set_runtime_scale(lokr.scale / 2)
        updated_delta = lokr._call_forward(value) - base
        torch.testing.assert_close(updated_delta, original_delta / 2)

        dora = DoRAModule(
            'orbit_dora_scale',
            original,
            lora_dim=4,
            alpha=8,
            network=network,
        )
        dora.org_forward = original.forward
        with torch.no_grad():
            dora.lora_up.weight.normal_()
        original_output = dora(value)
        dora._set_runtime_scale(dora.scale / 2)
        updated_output = dora(value)
        self.assertTrue(torch.isfinite(updated_output).all())
        self.assertFalse(torch.equal(updated_output, original_output))

    def test_dora_factorized_norm_reads_runtime_scale(self):
        torch.manual_seed(7)
        original = _linear()
        module = DoRAModule(
            "dora_dynamic_scale",
            original,
            lora_dim=4,
            alpha=8,
            network=_Network(),
        )
        with torch.no_grad():
            module.lora_up.weight.normal_()

        def expected_norm():
            weight = original.weight.detach().float()
            down = module.lora_down.weight.detach().float()
            up = module.lora_up.weight.detach().float()
            delta = (up @ down) * module.scale
            return (weight + delta).norm(dim=1)

        torch.testing.assert_close(
            module._get_weight_norm_from_factors(1.0),
            expected_norm(),
        )
        module._set_runtime_scale(module.scale / 2)
        torch.testing.assert_close(
            module._get_weight_norm_from_factors(1.0),
            expected_norm(),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_dora_factorized_norm_aligns_scale_with_offloaded_factors(self):
        torch.manual_seed(9)
        network = _Network()
        original = _linear()
        module = DoRAModule(
            "dora_offloaded_scale",
            original,
            lora_dim=4,
            alpha=8,
            network=network,
        ).to("cuda")
        with torch.no_grad():
            module.lora_up.weight.normal_()

        MemoryManager.attach(module, torch.device("cuda"))
        try:
            self.assertEqual(module._runtime_scale.device.type, "cuda")
            self.assertEqual(module.lora_down.weight.device.type, "cpu")
            self.assertEqual(module.lora_up.weight.device.type, "cpu")

            actual = module._get_weight_norm_from_factors(1.0)
            down = module.lora_down.weight.detach().float()
            up = module.lora_up.weight.detach().float()
            expected = (
                original.weight.detach().float()
                + (up @ down) * module.scale
            ).norm(dim=1)

            self.assertEqual(actual.device.type, "cpu")
            torch.testing.assert_close(actual, expected)
        finally:
            MemoryManager.detach(module)

    def test_cpu_compile_scale_update_does_not_recompile(self):
        torch.manual_seed(11)
        torch._dynamo.reset()
        network = _Network()
        original = _linear()
        original.requires_grad_(False)
        module = LoRAModule(
            "compiled_cpu_scale",
            original,
            lora_dim=4,
            alpha=8,
            network=network,
        )
        module.org_forward = original.forward
        with torch.no_grad():
            module.lora_up.weight.normal_()

        compile_count = 0

        def counting_backend(graph_module, _example_inputs):
            nonlocal compile_count
            compile_count += 1
            return graph_module.forward

        compiled = torch.compile(
            module,
            backend=counting_backend,
            fullgraph=False,
            dynamic=True,
        )
        value = torch.randn(2, 8)
        base = original(value)
        first = compiled(value)
        initial_compile_count = compile_count
        module._set_runtime_scale(module.scale / 2)
        second = compiled(value)

        torch.testing.assert_close(
            second - base,
            (first - base) / 2,
        )
        self.assertGreater(initial_compile_count, 0)
        self.assertEqual(compile_count, initial_compile_count)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_runtime_scale_follows_module_device(self):
        module = LoRAModule(
            "device_scale",
            _linear(),
            lora_dim=4,
            alpha=8,
            network=_Network(),
        ).to("cuda")

        self.assertEqual(
            module._runtime_scale.device,
            module.lora_down.weight.device,
        )
        self.assertEqual(module._runtime_scale.device.type, "cuda")

    @unittest.skipUnless(
        _cuda_compile_available(),
        "CUDA and Triton are required",
    )
    def test_dynamic_compile_stays_cuda_and_scale_updates_do_not_recompile(self):
        torch.manual_seed(0)
        torch._dynamo.reset()
        from torch._inductor import metrics

        metrics.reset()
        network = _Network()
        network.torch_multiplier = torch.ones(1, device="cuda")
        original = _linear(device="cuda", dtype=torch.bfloat16)
        original.requires_grad_(False)
        module = LoRAModule(
            "compiled_scale",
            original,
            lora_dim=4,
            alpha=torch.tensor(8, dtype=torch.bfloat16),
            network=network,
        ).to("cuda")
        module.org_forward = original.forward
        with torch.no_grad():
            module.lora_up.weight.normal_()

        value = torch.randn(
            2,
            8,
            device="cuda",
            dtype=torch.bfloat16,
        )
        eager = module(value)
        eager.square().mean().backward()
        eager_down_grad = module.lora_down.weight.grad.detach().clone()
        eager_up_grad = module.lora_up.weight.grad.detach().clone()
        module.zero_grad(set_to_none=True)

        compiled = torch.compile(module, fullgraph=False, dynamic=True)
        actual = compiled(value)
        actual.square().mean().backward()
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, eager, rtol=2e-2, atol=5e-2)
        torch.testing.assert_close(
            module.lora_down.weight.grad,
            eager_down_grad,
            rtol=2e-2,
            atol=5e-2,
        )
        torch.testing.assert_close(
            module.lora_up.weight.grad,
            eager_up_grad,
            rtol=2e-2,
            atol=5e-2,
        )
        self.assertEqual(
            getattr(metrics, "generated_cpp_vec_kernel_count", 0),
            0,
        )
        self.assertEqual(module._runtime_scale.device.type, "cuda")

        base = original(value)
        original_delta = actual - base
        generated_kernels = metrics.generated_kernel_count
        module._set_runtime_scale(0.5)
        updated = compiled(value)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            updated - base,
            original_delta * 0.25,
            rtol=2e-2,
            atol=5e-2,
        )
        self.assertEqual(metrics.generated_kernel_count, generated_kernels)


if __name__ == "__main__":
    unittest.main()
