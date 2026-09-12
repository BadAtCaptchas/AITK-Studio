"""Weight-free regressions for runtime defects exposed by project-wide Pylint."""
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from toolkit.util.nvfp4_quant import swizzle_nvfp4_scales, unswizzle_nvfp4_scales
from toolkit.util.ostris_quant import OstrisLinear
from toolkit.util.quantize import attach_ara_and_quantize, quantize, quantize_module


class QuantizationLintRegressionTests(unittest.TestCase):
    """Exercise quantization paths with tiny CPU layers and in-memory adapters."""

    def test_nvfp4_scale_export_inverts_the_existing_checkpoint_decoder(self):
        """Preserve every scale byte for both aligned and padded tile dimensions."""
        for rows, cols in [(1, 1), (128, 4), (130, 9)]:
            with self.subTest(rows=rows, cols=cols):
                scales = torch.arange(rows * cols).remainder(127).to(torch.uint8)
                scales = scales.reshape(rows, cols).view(torch.float8_e4m3fn)
                restored = unswizzle_nvfp4_scales(swizzle_nvfp4_scales(scales), rows, cols)
                self.assertTrue(torch.equal(restored.view(torch.uint8), scales.view(torch.uint8)))
        with self.assertRaises(ValueError):
            swizzle_nvfp4_scales(torch.zeros(2, 3, 4))

    def test_block_quantization_reuses_matching_packed_weights(self):
        """A second quantization pass must not cast or replace packed buffers."""
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList([torch.nn.Linear(128, 16)])
        quantize_module(model, 'uint4', block_names=['blocks'], status_fn=lambda _: None)
        self.assertIsInstance(model.blocks[0], OstrisLinear)
        before = {name: value.clone() for name, value in model.named_buffers()}
        quantize_module(
            model, 'uint4', dtype=torch.bfloat16, block_names=['blocks'], status_fn=lambda _: None
        )
        after = dict(model.named_buffers())
        self.assertEqual(before.keys(), after.keys())
        self.assertTrue(all(torch.equal(value, after[name]) for name, value in before.items()))

    def test_configured_exclusions_do_not_duplicate_quantize_arguments(self):
        """Combine model exclusions and user exclusions without changing the input map."""
        model = torch.nn.Sequential(torch.nn.Linear(128, 16), torch.nn.Linear(16, 16))
        options = {'exclude': ['1']}
        quantize_module(model, 'uint4', exclude=['0'], quantize_kwargs=options, status_fn=lambda _: None)
        self.assertNotIsInstance(model[0], OstrisLinear)
        self.assertNotIsInstance(model[1], OstrisLinear)
        self.assertEqual(options, {'exclude': ['1']})

    def test_leaf_quantization_accepts_explicit_final_placement(self):
        """The streamed extras path supports its advertised placement keyword."""
        layer = torch.nn.Linear(128, 16)
        quantize(layer, weights='uint4', quantize_device='cpu', keep_on_quantize_device=True)
        self.assertIsInstance(layer, OstrisLinear)

    def test_both_adapter_formats_reach_shared_network_and_quantization_setup(self):
        """LoRA and LoKr adapters use locally scoped options after helper extraction."""
        states = [
            {'projection.lora_A.weight': torch.zeros(4, 128)},
            {'lycoris_projection.lokr_w1': torch.zeros(2, 2)},
        ]
        for state in states:
            with self.subTest(keys=list(state)):
                model = torch.nn.Module()
                model.projection = torch.nn.Linear(128, 16)
                model.kept = torch.nn.Linear(128, 16)
                owner = SimpleNamespace(
                    model_config=SimpleNamespace(
                        qtype='uint4', low_vram=False,
                        quantize_kwargs={'include': ['projection'], 'exclude': ['kept']},
                    ),
                    device_torch=torch.device('cpu'), torch_dtype=torch.float32,
                    is_transformer=True, print_and_status_update=lambda _: None,
                )
                network = MagicMock()
                network.unet_loras = []
                with (
                    patch('toolkit.util.quantize.os.path.exists', return_value=True),
                    patch('toolkit.util.quantize.load_file', return_value=state),
                    patch('toolkit.lora_special.LoRASpecialNetwork', return_value=network) as factory,
                ):
                    result = attach_ara_and_quantize(owner, model, 'fixture.safetensors', device='cpu')
                factory.assert_called_once()
                network.apply_to.assert_called_once()
                self.assertIs(owner.accuracy_recovery_adapter, network)
                self.assertIsInstance(model.projection, OstrisLinear)
                self.assertNotIsInstance(model.kept, OstrisLinear)
                self.assertEqual(result.quantized_modules, 1)


if __name__ == '__main__':
    unittest.main()
