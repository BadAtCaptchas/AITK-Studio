import json
import unittest
import warnings

import torch
from torch import nn

from extensions_built_in.diffusion_models.ideogram4.src.quantized_loading import (
    COMFY_FLOAT8_FORMAT,
    COMFY_NVFP4_FORMAT,
    FP8_WEIGHT_DTYPE,
    Fp8Linear,
    Nvfp4Linear,
    decode_comfy_quant_marker,
    is_comfy_quant_state_dict,
    is_nvfp4_state_dict,
    load_comfy_quant_state_dict,
    swap_linears_to_comfy_quant,
)


def comfy_marker(fmt: str) -> torch.Tensor:
    data = json.dumps({"format": fmt}, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(data), dtype=torch.uint8)


def pack_nvfp4(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2 != 0:
        raise ValueError("NVFP4 test codes must have an even number of columns")
    high = codes[..., 0::2]
    low = codes[..., 1::2]
    return ((high << 4) | low).to(torch.uint8)


def swizzle_block_scales(scales: torch.Tensor) -> torch.Tensor:
    rows, cols = scales.shape
    padded_rows = ((rows + 127) // 128) * 128
    padded_cols = ((cols + 3) // 4) * 4
    padded = torch.zeros(padded_rows, padded_cols, dtype=scales.dtype)
    padded[:rows, :cols] = scales
    n_row_blocks = padded_rows // 128
    n_col_blocks = padded_cols // 4
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    return (
        blocks.reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(padded_rows, padded_cols)
    )


class TinyTwoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 2, bias=False)
        self.b = nn.Linear(4, 2, bias=False)


class TinyEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(3, 2)


class TinyMetaTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        self.unused = nn.Parameter(torch.empty(1, device="meta"))


class Ideogram4Nvfp4LoadingTest(unittest.TestCase):
    def test_comfy_marker_detection(self):
        marker = comfy_marker(COMFY_NVFP4_FORMAT)
        state_dict = {"linear.comfy_quant": marker}

        self.assertEqual(decode_comfy_quant_marker(marker), {"format": "nvfp4"})
        self.assertTrue(is_comfy_quant_state_dict(state_dict))
        self.assertTrue(is_nvfp4_state_dict(state_dict))

    def test_nvfp4_dequant_unswizzles_scales_and_trims_padding(self):
        layer = Nvfp4Linear(17, 2, bias=False, compute_dtype=torch.float32)
        codes = torch.tensor(
            [
                list(range(16)) + [1] + [0] * 15,
                [7] * 16 + [2] + [0] * 15,
            ],
            dtype=torch.uint8,
        )
        block_scales = torch.tensor(
            [[1.0, 2.0], [3.0, 1.0]],
            dtype=torch.float32,
        )

        layer.weight.copy_(pack_nvfp4(codes))
        layer.weight_scale.copy_(swizzle_block_scales(block_scales).to(FP8_WEIGHT_DTYPE))
        layer.weight_scale_2.copy_(torch.tensor(2.0))

        actual = layer.dequantize_weight(dtype=torch.float32)
        table = Nvfp4Linear.e2m1_table(device=torch.device("cpu"), dtype=torch.float32)
        expected = table[codes.long()].view(2, 2, 16)
        expected = expected * block_scales.unsqueeze(-1) * 2.0
        expected = expected.flatten(1)[:, :17]

        self.assertEqual(tuple(actual.shape), (2, 17))
        torch.testing.assert_close(actual, expected)

    def test_comfy_fp8_loader_strips_marker_and_supports_scalar_scale(self):
        model = nn.Sequential(nn.Linear(4, 2, bias=False))
        weight = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, -2.0, 4.0]],
            dtype=torch.float32,
        )
        state_dict = {
            "0.comfy_quant": comfy_marker(COMFY_FLOAT8_FORMAT),
            "0.weight": weight.to(FP8_WEIGHT_DTYPE),
            "0.weight_scale": torch.tensor(0.5, dtype=torch.float32),
        }

        swap_linears_to_comfy_quant(model, state_dict, compute_dtype=torch.float32)
        self.assertIsInstance(model[0], Fp8Linear)
        load_comfy_quant_state_dict(
            model, state_dict, device=torch.device("cpu"), dtype=torch.float32
        )

        x = torch.ones(1, 4, dtype=torch.float32)
        torch.testing.assert_close(model(x), torch.nn.functional.linear(x, weight * 0.5))

    def test_comfy_nvfp4_swaps_only_matching_linear_modules(self):
        model = TinyTwoLinear()
        b_weight = model.b.weight.detach().clone()
        codes = torch.tensor(
            [
                [1, 2, 3, 4] + [0] * 12,
                [5, 6, 7, 8] + [0] * 12,
            ],
            dtype=torch.uint8,
        )
        state_dict = {
            "a.comfy_quant": comfy_marker(COMFY_NVFP4_FORMAT),
            "a.weight": pack_nvfp4(codes),
            "a.weight_scale": swizzle_block_scales(
                torch.ones(2, 1, dtype=torch.float32)
            ).to(FP8_WEIGHT_DTYPE),
            "a.weight_scale_2": torch.tensor(1.0, dtype=torch.float32),
            "b.weight": b_weight,
        }

        swap_linears_to_comfy_quant(model, state_dict, compute_dtype=torch.float32)
        self.assertIsInstance(model.a, Nvfp4Linear)
        self.assertIsInstance(model.b, nn.Linear)
        load_comfy_quant_state_dict(
            model, state_dict, device=torch.device("cpu"), dtype=torch.float32
        )

        self.assertEqual(tuple(model.a.weight.shape), (2, 8))
        self.assertEqual(tuple(model.a.dequantize_weight(dtype=torch.float32).shape), (2, 4))
        torch.testing.assert_close(model.b.weight, b_weight)

    def test_comfy_fp8_embedding_weights_dequantize_during_load(self):
        model = TinyEmbedding()
        weight = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]],
            dtype=torch.float32,
        )
        state_dict = {
            "emb.comfy_quant": comfy_marker(COMFY_FLOAT8_FORMAT),
            "emb.weight": weight.to(FP8_WEIGHT_DTYPE),
            "emb.weight_scale": torch.tensor(0.25, dtype=torch.float32),
        }

        load_comfy_quant_state_dict(
            model, state_dict, device=torch.device("cpu"), dtype=torch.float32
        )

        torch.testing.assert_close(model.emb.weight, weight * 0.25)

    def test_assign_load_skips_blanket_to_when_unused_meta_params_remain(self):
        model = TinyMetaTail()
        weight = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
        )
        state_dict = {
            "linear.comfy_quant": comfy_marker(COMFY_FLOAT8_FORMAT),
            "linear.weight": weight.to(FP8_WEIGHT_DTYPE),
            "linear.weight_scale": torch.tensor(1.0, dtype=torch.float32),
        }

        swap_linears_to_comfy_quant(model, state_dict, compute_dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_comfy_quant_state_dict(
                model,
                state_dict,
                device=torch.device("cpu"),
                dtype=torch.float32,
                assign=True,
                strict=False,
            )

        self.assertFalse(model.linear.weight.is_meta)
        self.assertTrue(model.unused.is_meta)
        torch.testing.assert_close(model.linear.weight.to(torch.float32), weight)

    def test_lora_discovery_marks_nvfp4_unmergeable(self):
        from toolkit import lora_special, network_mixins

        layer = Nvfp4Linear(4, 2, bias=False, compute_dtype=torch.float32)
        self.assertIn("Nvfp4Linear", lora_special.LINEAR_MODULES)
        self.assertIn("Nvfp4Linear", network_mixins.LINEAR_MODULES)
        self.assertIn("Nvfp4Linear", network_mixins.UNMERGEABLE_MODULES)
        self.assertFalse(network_mixins.is_mergeable_lora_target(layer))


if __name__ == "__main__":
    unittest.main()
