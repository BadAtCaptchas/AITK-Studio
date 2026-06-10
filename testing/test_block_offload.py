import unittest
from unittest import mock
from types import SimpleNamespace

import torch

from toolkit.memory_management.block_offload import BlockOffloadManager, LayerOffloadStrategy
from toolkit.memory_management.manager import MemoryManager
from toolkit.memory_management.offload import (
    is_block_offload_arch_supported,
    resolve_layer_offloading_backend,
)


class TinyBlockModel(torch.nn.Module):
    def __init__(self, block_count=4):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.SiLU())
            for _ in range(block_count)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class StorageAliasSensitiveTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, value):
        return torch.Tensor._make_subclass(cls, value, False)


class TensorSubclassBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            StorageAliasSensitiveTensor(torch.randn(4, 4)),
            requires_grad=False,
        )

    def forward(self, x):
        return x


class MixedTensorSubclassModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                TensorSubclassBlock(),
                torch.nn.Linear(4, 4),
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Fp8Linear(torch.nn.Module):
    def __init__(self, in_features=4, out_features=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = torch.float32
        logical_weight = torch.arange(
            1,
            out_features * in_features + 1,
            dtype=torch.float32,
        ).reshape(out_features, in_features)
        self.register_buffer("weight", logical_weight.reshape(-1, 1))
        self.register_buffer(
            "weight_scale",
            torch.linspace(0.5, 1.5, out_features, dtype=torch.float32),
        )
        self.register_buffer(
            "bias", torch.linspace(-0.25, 0.25, out_features, dtype=torch.float32)
        )

    def forward(self, x):
        weight = self.weight.reshape(self.out_features, self.in_features).to(x.dtype)
        weight = weight * self.weight_scale.to(x.dtype).unsqueeze(1)
        bias = self.bias.to(x.dtype)
        return torch.nn.functional.linear(x, weight, bias)


class FakeNf4Weight(torch.Tensor):
    @staticmethod
    def __new__(cls, logical_weight):
        packed = logical_weight.reshape(-1, 1)
        instance = torch.Tensor._make_subclass(cls, packed, False)
        instance._logical_weight = logical_weight
        return instance

    @property
    def quant_state(self):
        return True

    def dequantize(self):
        return self._logical_weight.to(self.device)


class FakeBnbQuantState:
    quant_type = "nf4"

    def __init__(self, logical_weight):
        self.logical_weight = logical_weight

    def to(self, device):
        self.logical_weight = self.logical_weight.to(device)
        return self

    def dequantize_4bit(self, data):
        return self.logical_weight.t().to(data.device)


class Params4bit(torch.nn.Parameter):
    @staticmethod
    def __new__(cls, logical_weight):
        packed = torch.zeros(
            1,
            logical_weight.numel() // 2,
            dtype=torch.uint8,
            device=logical_weight.device,
        )
        instance = torch.Tensor._make_subclass(cls, packed, False)
        instance.quant_state = FakeBnbQuantState(logical_weight)
        instance.bnb_quantized = True
        instance.quant_type = "nf4"
        instance.quant_storage = torch.uint8
        instance.data = packed
        return instance

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        logical_weight = self.quant_state.logical_weight
        if device is not None:
            logical_weight = logical_weight.to(device)
        return type(self)(logical_weight)


class Linear4bit(torch.nn.Module):
    def __init__(self, in_features=4, out_features=3, *, packed_params=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = torch.float32
        self.packed_params = packed_params
        logical_weight = torch.arange(
            1,
            out_features * in_features + 1,
            dtype=torch.float32,
        ).reshape(out_features, in_features)
        if packed_params:
            self.weight = Params4bit(logical_weight)
        else:
            self.register_buffer("weight", FakeNf4Weight(logical_weight))
        self.bias = None

    def forward(self, x):
        if self.packed_params:
            weight = self.weight.quant_state.logical_weight.to(x.device)
        else:
            weight = self.weight.dequantize()
        return torch.nn.functional.linear(
            x.to(self.compute_dtype), weight, self.bias
        )


class FakeCudaStream:
    def __init__(self):
        self.waited_streams = []
        self.waited_events = []

    def wait_stream(self, stream):
        self.waited_streams.append(stream)

    def wait_event(self, event):
        self.waited_events.append(event)


class FakeCudaStreamContext:
    def __init__(self, stream, log):
        self.stream = stream
        self.log = log

    def __enter__(self):
        self.log.append(("enter", self.stream))

    def __exit__(self, exc_type, exc, tb):
        self.log.append(("exit", self.stream))


class FakeCudaEvent:
    def __init__(self):
        self.recorded_stream = None

    def record(self, stream):
        self.recorded_stream = stream


class LayerOffloadStrategyTest(unittest.TestCase):
    def test_selects_deterministic_whole_block_suffix(self):
        first = LayerOffloadStrategy((10, 10, 10, 10), 0.7)
        second = LayerOffloadStrategy((10, 10, 10, 10), 0.7)

        self.assertEqual(first.offloaded_indices, (1, 2, 3))
        self.assertEqual(first.offloaded_indices, second.offloaded_indices)
        self.assertEqual(first.resident_indices, (0,))

    def test_uneven_layer_sizes_use_closest_suffix(self):
        strategy = LayerOffloadStrategy((60, 1, 1, 1), 0.5)

        self.assertEqual(strategy.offloaded_indices, (1, 2, 3))
        self.assertEqual(strategy.resident_indices, (0,))

    def test_windows_follow_forward_and_backward_order(self):
        strategy = LayerOffloadStrategy((10, 10, 10, 10, 10), 0.6)

        self.assertEqual(strategy.forward_forward_window(2), (2, 3))
        self.assertEqual(strategy.forward_backward_window(3), (2, 3))
        self.assertEqual(strategy.backward_forward_window(3), (3, 4))
        self.assertEqual(strategy.forward_forward_window(0), ())


class BlockOffloadManagerTest(unittest.TestCase):
    def test_attach_is_idempotent_and_forward_still_runs_on_cpu(self):
        model = TinyBlockModel()
        x = torch.randn(2, 4)
        expected = model(x)

        manager = BlockOffloadManager.attach(
            model,
            torch.device("cpu"),
            offload_fraction=0.5,
            block_paths=["blocks"],
        )
        second = BlockOffloadManager.attach(
            model,
            torch.device("cpu"),
            offload_fraction=0.5,
            block_paths=["blocks"],
        )

        self.assertIs(manager, second)
        self.assertTrue(hasattr(model, "_block_offload_manager"))
        self.assertTrue(torch.allclose(model(x), expected))

    def test_detach_restores_original_forwards(self):
        model = TinyBlockModel()
        manager = BlockOffloadManager.attach(
            model,
            torch.device("cpu"),
            offload_fraction=1.0,
            block_paths=["blocks"],
        )

        manager.detach()

        self.assertFalse(hasattr(model, "_block_offload_manager"))
        self.assertFalse(hasattr(model, "_block_offload_original_to"))
        self.assertTrue(callable(model.blocks[0].forward))

    def test_unsupported_block_paths_fail_clearly(self):
        model = TinyBlockModel()

        with self.assertRaisesRegex(ValueError, "could not find ordered block layers"):
            BlockOffloadManager.attach(
                model,
                torch.device("cpu"),
                offload_fraction=1.0,
                block_paths=["missing.blocks"],
            )

    def test_tensor_subclass_blocks_are_left_resident(self):
        model = MixedTensorSubclassModel()
        x = torch.randn(2, 4)
        expected = model(x)

        manager = BlockOffloadManager.attach(
            model,
            torch.device("cpu"),
            offload_fraction=1.0,
            block_paths=["blocks"],
        )

        self.assertEqual([entry.name for entry in manager.layers], ["blocks.1"])
        self.assertEqual(model._aitk_block_offload_skipped_layers, ("blocks.0",))
        self.assertTrue(torch.allclose(model(x), expected))

    def test_async_offload_waits_for_compute_stream_before_copy(self):
        model = TinyBlockModel(block_count=1)
        manager = BlockOffloadManager.attach(
            model,
            torch.device("cpu"),
            offload_fraction=1.0,
            block_paths=["blocks"],
        )
        entry = manager.layers[0]
        entry.state = "device"
        manager.active = True
        manager.process_device = torch.device("cuda")
        manager.transfer_stream = FakeCudaStream()
        current_stream = FakeCudaStream()
        stream_log = []

        with (
            mock.patch.object(torch.cuda, "current_stream", return_value=current_stream),
            mock.patch.object(
                torch.cuda,
                "stream",
                side_effect=lambda stream: FakeCudaStreamContext(stream, stream_log),
            ),
            mock.patch.object(torch.cuda, "Event", FakeCudaEvent),
        ):
            manager._offload_entry(entry, async_transfer=True)

        self.assertEqual(manager.transfer_stream.waited_streams, [current_stream])
        self.assertEqual(stream_log, [("enter", manager.transfer_stream), ("exit", manager.transfer_stream)])
        self.assertEqual(entry.state, "offloading")
        self.assertIs(entry.transfer_event.recorded_stream, manager.transfer_stream)

    def test_ensure_device_waits_for_pending_offload_before_reload(self):
        model = TinyBlockModel(block_count=1)
        manager = BlockOffloadManager.attach(
            model,
            torch.device("cpu"),
            offload_fraction=1.0,
            block_paths=["blocks"],
        )
        entry = manager.layers[0]
        entry.state = "offloading"
        event = FakeCudaEvent()
        entry.transfer_event = event
        manager.active = True
        manager.process_device = torch.device("cuda")
        current_stream = FakeCudaStream()

        with (
            mock.patch.object(torch.cuda, "current_stream", return_value=current_stream),
            mock.patch.object(manager, "_move_entry") as move_entry,
        ):
            manager._ensure_entry_on_device(entry)

        self.assertEqual(current_stream.waited_events, [event])
        self.assertIsNone(entry.transfer_event)
        move_entry.assert_called_once_with(entry, torch.device("cuda"), async_transfer=False)
        self.assertEqual(entry.state, "device")

    def test_all_tensor_subclass_blocks_fail_clearly(self):
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList([TensorSubclassBlock()])

        with self.assertRaisesRegex(ValueError, "storage-swappable tensors"):
            BlockOffloadManager.attach(
                model,
                torch.device("cpu"),
                offload_fraction=1.0,
                block_paths=["blocks"],
            )

    def test_default_block_backend_falls_back_to_legacy_without_cuda(self):
        model = TinyBlockModel()
        config = SimpleNamespace(
            arch="flux",
            layer_offloading=True,
            layer_offloading_backend="block",
            _layer_offloading_backend_explicit=True,
        )

        if torch.cuda.is_available():
            self.skipTest("CUDA is available; this fallback is only CPU-safe.")

        self.assertEqual(
            resolve_layer_offloading_backend(config, model, torch.device("cpu"), ["blocks"]),
            "legacy",
        )

    def test_ideogram_is_block_offload_supported(self):
        self.assertTrue(is_block_offload_arch_supported("ideogram4"))
        self.assertTrue(is_block_offload_arch_supported("ideogram4:fp8"))

    def test_legacy_memory_manager_detach_restores_cpu_toy_module(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        input_tensor = torch.ones(1, 4)
        expected = model(input_tensor)

        MemoryManager.attach(model, torch.device("cpu"))
        self.assertTrue(hasattr(model, "_memory_manager"))
        self.assertTrue(hasattr(model[0], "_layer_memory_manager"))
        self.assertTrue(hasattr(model[0], "_memory_management_device"))

        managed = model(input_tensor)
        self.assertTrue(torch.allclose(managed, expected))

        MemoryManager.detach(model)

        self.assertFalse(hasattr(model, "_memory_manager"))
        self.assertFalse(hasattr(model, "_mm_to"))
        self.assertFalse(hasattr(model[0], "_layer_memory_manager"))
        self.assertFalse(hasattr(model[0], "_memory_management_device"))
        self.assertTrue(torch.allclose(model(input_tensor), expected))

    def test_legacy_memory_manager_handles_flat_fp8_linear_buffers(self):
        model = torch.nn.Sequential(Fp8Linear())
        input_tensor = torch.randn(2, model[0].in_features)
        expected = model(input_tensor)

        MemoryManager.attach(model, torch.device("cpu"))
        try:
            managed = model(input_tensor)
            self.assertTrue(torch.allclose(managed, expected))
        finally:
            MemoryManager.detach(model)

    def test_legacy_memory_manager_handles_nf4_dequantized_weight(self):
        model = torch.nn.Sequential(Linear4bit())
        input_tensor = torch.randn(2, model[0].in_features)
        expected = model(input_tensor)

        MemoryManager.attach(model, torch.device("cpu"))
        try:
            managed = model(input_tensor)
            self.assertTrue(torch.allclose(managed, expected))
        finally:
            MemoryManager.detach(model)

    def test_legacy_memory_manager_handles_bnb_nf4_packed_params4bit(self):
        model = torch.nn.Sequential(Linear4bit(packed_params=True))
        input_tensor = torch.randn(2, model[0].in_features)
        expected = model(input_tensor)

        MemoryManager.attach(model, torch.device("cpu"))
        try:
            managed = model(input_tensor)
            self.assertTrue(torch.allclose(managed, expected))
        finally:
            MemoryManager.detach(model)

    def test_memory_manager_detach_calls_block_manager(self):
        class FakeBlockManager:
            def __init__(self):
                self.called = False

            def detach(self):
                self.called = True

        model = torch.nn.Linear(4, 4)
        manager = FakeBlockManager()
        model._block_offload_manager = manager
        model._aitk_layer_offloading_backend = "block"

        MemoryManager.detach(model)

        self.assertTrue(manager.called)
        self.assertFalse(hasattr(model, "_block_offload_manager"))
        self.assertFalse(hasattr(model, "_aitk_layer_offloading_backend"))


if __name__ == "__main__":
    unittest.main()
