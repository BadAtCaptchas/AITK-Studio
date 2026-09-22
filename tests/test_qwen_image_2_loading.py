"""Exercise Diffusers and toolkit checkpoint loading without pretrained downloads."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from extensions_built_in.diffusion_models.qwen_image_2.src.pipeline import run_transformer
from extensions_built_in.diffusion_models.qwen_image_2.src.transformer import QwenImage21Transformer2DModel


def tiny_transformer():
    return QwenImage21Transformer2DModel(
        in_channels=4, out_channels=4, num_layers=2, num_attention_heads=1,
        attention_head_dim=8, context_in_dim=8, axes_dims_rope=(2, 2, 4),
    )


def checkpoint_state(model, split):
    state = {}
    for key, value in model.state_dict().items():
        if split and key.endswith('.img_mlp.gate_up.weight'):
            prefix = key[:-len('.gate_up.weight')]
            gate, up = value.chunk(2, dim=0)
            state[f'{prefix}.gate_layer.weight'] = gate.clone()
            state[f'{prefix}.proj.weight'] = up.clone()
        else:
            state[key] = value.clone()
    return state


def write_checkpoint(folder, model, state, sharded):
    model.save_config(folder)
    if not sharded:
        save_file(state, folder / 'diffusion_pytorch_model.safetensors')
        return
    # Deliberately separate every gate from its up projection across shard files.
    shards = [{}, {}, {}]
    for key, value in state.items():
        index = 0 if key.endswith('.gate_layer.weight') else 2 if key.endswith('.img_mlp.proj.weight') else 1
        shards[index][key] = value
    weight_map = {}
    for index, shard in enumerate(shards, 1):
        filename = f'diffusion_pytorch_model-{index:05d}-of-00003.safetensors'
        save_file(shard, folder / filename)
        weight_map.update({key: filename for key in shard})
    (folder / 'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps({
        'metadata': {'total_size': sum(value.numel() * value.element_size() for value in state.values())},
        'weight_map': weight_map,
    }), encoding='utf-8')


class QwenImage2CheckpointLoadingTests(unittest.TestCase):
    def assert_weights_preserved(self, loaded, original, dtype):
        expected = original.state_dict()
        self.assertEqual(set(loaded.state_dict()), set(expected))
        for key, value in loaded.state_dict().items():
            self.assertFalse(value.is_meta, key)
            self.assertEqual(value.dtype, dtype, key)
            torch.testing.assert_close(value, expected[key].to(dtype), rtol=0, atol=0, msg=key)

    def test_original_checkpoints_preserve_all_weights_and_loading_info(self):
        original = tiny_transformer()
        for sharded in (False, True):
            for low_cpu_mem_usage in (False, True):
                for dtype in (torch.float32, torch.bfloat16):
                    with self.subTest(sharded=sharded, low_cpu_mem_usage=low_cpu_mem_usage, dtype=dtype):
                        with TemporaryDirectory() as temporary:
                            folder = Path(temporary)
                            write_checkpoint(folder, original, checkpoint_state(original, split=True), sharded)
                            loaded, info = QwenImage21Transformer2DModel.from_pretrained(
                                folder, local_files_only=True, torch_dtype=dtype,
                                low_cpu_mem_usage=low_cpu_mem_usage, output_loading_info=True,
                            )
                            self.assertTrue(all(not value for value in info.values()), info)
                            self.assert_weights_preserved(loaded, original, dtype)

    def test_fused_checkpoints_still_load_unchanged(self):
        original = tiny_transformer()
        for sharded in (False, True):
            with self.subTest(sharded=sharded), TemporaryDirectory() as temporary:
                folder = Path(temporary)
                write_checkpoint(folder, original, checkpoint_state(original, split=False), sharded)
                loaded, info = QwenImage21Transformer2DModel.from_pretrained(
                    folder, local_files_only=True, output_loading_info=True,
                )
                self.assertTrue(all(not value for value in info.values()), info)
                self.assert_weights_preserved(loaded, original, torch.float32)

    def test_toolkit_loader_preserves_forward_and_training_gradients(self):
        original = tiny_transformer().eval()
        inputs = (
            torch.randn(1, 4, 2, 2), torch.tensor([0.5]), torch.randn(1, 3, 8),
            torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.bool),
        )
        for sharded in (False, True):
            with self.subTest(sharded=sharded), TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_checkpoint(root / 'transformer', original, checkpoint_state(original, split=True), sharded)
                with patch('diffusers.models.modeling_utils.HF_ENABLE_PARALLEL_LOADING', True):
                    loaded = QwenImage21Transformer2DModel.load(
                        str(root), use_comfy_weights=False, dtype=torch.float32, device='cpu', qtype=None,
                    )
                self.assert_weights_preserved(loaded, original, torch.float32)
                expected = run_transformer(original, *inputs)
                actual = run_transformer(loaded, *inputs)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                actual.square().mean().backward()
                expected.square().mean().backward()
                for actual_block, expected_block in zip(loaded.transformer_blocks, original.transformer_blocks):
                    grad = actual_block.img_mlp.gate_up.weight.grad
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)
                    torch.testing.assert_close(grad, expected_block.img_mlp.gate_up.weight.grad, rtol=1e-5, atol=1e-6)
                original.zero_grad()

    def test_incomplete_or_ambiguous_original_mlp_fails(self):
        original = tiny_transformer()
        for sharded in (False, True):
            for ambiguous in (False, True):
                with self.subTest(sharded=sharded, ambiguous=ambiguous), TemporaryDirectory() as temporary:
                    state = checkpoint_state(original, split=True)
                    if ambiguous:
                        state['transformer_blocks.0.img_mlp.gate_up.weight'] = original.transformer_blocks[0].img_mlp.gate_up.weight.clone()
                    else:
                        del state['transformer_blocks.0.img_mlp.proj.weight']
                    folder = Path(temporary)
                    write_checkpoint(folder, original, state, sharded)
                    with self.assertRaisesRegex(ValueError, 'Ambiguous' if ambiguous else 'Incomplete'):
                        QwenImage21Transformer2DModel.from_pretrained(folder, local_files_only=True)


if __name__ == '__main__':
    unittest.main()
