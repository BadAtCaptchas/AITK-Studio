import gc
import unittest

import torch

from toolkit.memory_management.block_offload import BlockOffloadManager
from toolkit.models.lokr import LokrModule
from toolkit.util import convrot_quant
from toolkit.util.ostris_quant import OstrisLinear
from toolkit.util.quantize import quantize


CUDA_AVAILABLE = torch.cuda.is_available()


def quantized_linear(qtype: str, size: int = 256) -> OstrisLinear:
    layer = torch.nn.Linear(
        size, size, device="cuda", dtype=torch.bfloat16
    )
    quantize(layer, weights=qtype, kernel="auto", max_workspace_mb=8)
    if not isinstance(layer, OstrisLinear):
        raise AssertionError(f"{qtype} was not converted")
    return layer


class NetworkStub:
    network_type = "lokr"
    is_lorm = False
    is_active = True
    is_merged_in = False
    _multiplier = 1.0

    def __init__(self):
        self.torch_multiplier = torch.tensor([1.0], device="cuda")


class PackedBlockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(256, 256, dtype=torch.bfloat16),
                    torch.nn.GELU(),
                    torch.nn.Linear(256, 256, dtype=torch.bfloat16),
                )
                for _ in range(3)
            ]
        )
        quantize(self, weights="convrotint4", kernel="auto", max_workspace_mb=8)

    def forward(self, value):
        for block in self.blocks:
            value = value + block(value)
        return value


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is unavailable")
class ConvRotCudaTest(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_fp4_and_int8_hardware_forward_backward(self):
        device = torch.device("cuda")
        self.assertTrue(convrot_quant._fp4_gemm_supported(device))
        self.assertTrue(convrot_quant._int8_gemm_supported(device))
        for qtype in ("convrot4", "convrot8"):
            with self.subTest(qtype=qtype):
                layer = quantized_linear(qtype)
                value = torch.randn(
                    37, 256, device=device, dtype=torch.bfloat16, requires_grad=True
                )
                output = layer(value)
                output.float().square().mean().backward()
                self.assertEqual(output.shape, (37, 256))
                self.assertTrue(torch.isfinite(output).all())
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.isfinite(value.grad).all())

    def test_dynamic_shape_compile(self):
        # CUDA Inductor requires Triton even when the graph contains opaque
        # custom ops. Windows builds without Triton still exercise Dynamo/AOT,
        # dynamic guards, and registered custom-op autograd through aot_eager.
        backend = "inductor" if convrot_quant._triton_available() else "aot_eager"
        for qtype in ("convrot4", "convrot8"):
            with self.subTest(qtype=qtype):
                layer = quantized_linear(qtype)
                compiled = torch.compile(
                    layer, backend=backend, dynamic=True, fullgraph=False
                )
                for rows in (7, 33):
                    value = torch.randn(
                        rows,
                        256,
                        device="cuda",
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    )
                    output = compiled(value)
                    output.float().mean().backward()
                    self.assertEqual(output.shape, (rows, 256))
                    self.assertTrue(torch.isfinite(value.grad).all())

    def test_representative_packed_qtypes(self):
        for qtype in (
            "convrotint4",
            "convrotbitnet",
            "convrotcomfyw4a4",
        ):
            with self.subTest(qtype=qtype):
                layer = quantized_linear(qtype)
                value = torch.randn(
                    17, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
                )
                output = layer(value)
                output.float().mean().backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(value.grad).all())

    def test_factorized_lokr_training_on_packed_base(self):
        base = quantized_linear("convrotint4")
        network = NetworkStub()
        adapter = LokrModule(
            "convrot_lokr",
            base,
            lora_dim=4,
            alpha=4,
            factor=16,
            network=network,
        ).to(device="cuda", dtype=torch.bfloat16)
        adapter._test_network = network
        adapter.apply_to()
        value = torch.randn(
            5, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        output = base(value)
        output.float().square().mean().backward()
        self.assertTrue(adapter.bypass_mode)
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in adapter.parameters()
            )
        )

    def test_block_offload_training_with_packed_layers(self):
        model = PackedBlockModel()
        manager = BlockOffloadManager.attach(
            model,
            torch.device("cuda"),
            offload_fraction=2 / 3,
            block_paths=["blocks"],
        )
        try:
            value = torch.randn(
                4, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            output = model(value)
            output.float().mean().backward()
            torch.cuda.synchronize()
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertTrue(manager.active)
        finally:
            manager.deactivate_to_cpu()
            manager.detach()


if __name__ == "__main__":
    unittest.main()
