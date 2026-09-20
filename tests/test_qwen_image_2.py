"""Qwen Image 2.1 integration checks without model downloads."""
import json
import unittest
from types import SimpleNamespace

import torch

from extensions_built_in.diffusion_models import QwenImage2Model
from extensions_built_in.diffusion_models.qwen_image_2.src.pipeline import (
    QwenImage21Pipeline, pack_latents, unpack_latents, pad_prompt_batch,
    prepare_condition_image, run_transformer, tensor_to_pil,
)
from extensions_built_in.diffusion_models.qwen_image_2.src.text_encoder import QwenImage21TextEncoder
from extensions_built_in.diffusion_models.qwen_image_2.src.transformer import QwenImage21Transformer2DModel
from extensions_built_in.diffusion_models.qwen_image_2.src.vae import AutoencoderKLQwenImage21
from toolkit.config_modules import ModelConfig
from toolkit.config_contract import config_contract_errors
from toolkit.model_registry import resolve_model
from toolkit.models.registry import get_arch_entry
from toolkit.models.v2._mixin import OstrisModelMixin
from toolkit.util.comfy_quant_import import import_comfy_quantized_layers


def tiny_transformer():
    return QwenImage21Transformer2DModel(
        in_channels=4, out_channels=4, num_layers=1, num_attention_heads=1,
        attention_head_dim=8, context_in_dim=8, axes_dims_rope=(2, 2, 4),
    )


class QwenImage2Tests(unittest.TestCase):
    def test_registration_and_optional_edit_controls(self):
        self.assertIs(resolve_model('qwen_image_2'), QwenImage2Model)
        self.assertEqual(config_contract_errors({'config': {'process': [{
            'type': 'diffusion_trainer', 'model': {'arch': 'qwen_image_2'},
        }]}}), [])
        entry = get_arch_entry('qwen_image_2')
        self.assertFalse(entry['needs_control_image'])
        self.assertEqual(entry['sample']['guidance_scale'], 1)
        self.assertEqual(entry['model']['qtype'], 'convrot8')

    def test_materialized_convrot_embedding_is_not_a_missing_checkpoint_key(self):
        model = torch.nn.Module()
        model.embed = torch.nn.Embedding(3, 16, device='meta')
        model.proj = torch.nn.Linear(16, 2, bias=False, device='meta')
        state = {
            'embed.weight': torch.arange(-24, 24, dtype=torch.int8).reshape(3, 16),
            'embed.weight_scale': torch.ones(3) * 0.1,
            'embed.comfy_quant': torch.tensor(list(json.dumps({
                'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 16,
            }).encode()), dtype=torch.uint8),
            'proj.weight': torch.ones(2, 16),
        }
        remaining, count = import_comfy_quantized_layers(model, state, orig_dtype=torch.float32)
        self.assertEqual(count, 1)
        self.assertNotIn('embed.weight', remaining)
        OstrisModelMixin._load_state_dict_with_quantized(model, remaining)
        self.assertTrue(all(not p.is_meta for p in model.parameters()))
        self.assertTrue(torch.isfinite(model.proj(model.embed(torch.tensor([0])))).all())

    def test_loader_still_rejects_missing_meta_and_unexpected_parameters(self):
        model = torch.nn.Linear(2, 2, device='meta')
        with self.assertRaisesRegex(ValueError, 'missing'):
            OstrisModelMixin._load_state_dict_with_quantized(model, {'weight': torch.ones(2, 2)})
        model = torch.nn.Linear(2, 2, device='meta')
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            OstrisModelMixin._load_state_dict_with_quantized(model, {
                'weight': torch.ones(2, 2), 'bias': torch.ones(2), 'unknown': torch.ones(1),
            })

    def test_checkpoint_key_conversions(self):
        state = {key: torch.ones(1) for key in (
            'model.layers.0.attn.weight', 'model.embed_tokens.weight',
            'model.norm.weight', 'model.visual.weight',
        )}
        converted = QwenImage21TextEncoder.convert_state_dict_on_load(state)
        self.assertIn('model.language_model.embed_tokens.weight', converted)
        restored = QwenImage21TextEncoder.convert_state_dict_on_save(converted)
        self.assertEqual(set(restored), set(state))
        gate, up = torch.ones(3, 2), torch.full((3, 2), 2.0)
        fused = QwenImage21Transformer2DModel.convert_state_dict_on_load({
            'transformer_blocks.0.img_mlp.gate_layer.weight': gate,
            'transformer_blocks.0.img_mlp.proj.weight': up,
        })
        torch.testing.assert_close(fused['transformer_blocks.0.img_mlp.gate_up.weight'], torch.cat([gate, up]))
        vae = AutoencoderKLQwenImage21.convert_state_dict_on_load({'encoder.conv1.weight': torch.ones(2, 4, 1, 3, 3)})
        self.assertEqual(tuple(vae['encoder.conv_in.weight'].shape), (2, 4, 3, 3))

    def test_reference_resize_alpha_and_duplicate_normalization(self):
        image = torch.rand(1, 4, 90, 170)
        prepared = prepare_condition_image(image, 64 * 64)
        self.assertEqual(prepared.shape[-1] % 32, 0)
        self.assertEqual(prepared.shape[-2] % 32, 0)
        torch.testing.assert_close(prepare_condition_image(image, 64 * 64), prepared)
        transparent = torch.zeros(4, 2, 2)
        self.assertEqual(tensor_to_pil(transparent).getpixel((0, 0)), (255, 255, 255))
        holder = QwenImage2Model('cpu', ModelConfig(arch='qwen_image_2', name_or_path='test'), dtype='fp32')
        self.assertEqual(holder.get_bucket_divisibility(), 32)
        self.assertEqual(len(holder._normalize_control_images([[image, image, image + 1]], 1)[0]), 2)

    def test_prompt_padding_and_latent_roundtrip(self):
        embeds, mask, slots = pad_prompt_batch(
            [torch.ones(2, 8), torch.ones(3, 8)],
            [torch.ones(2), torch.ones(3)],
            [torch.tensor([True, False]), torch.tensor([True, False, False])],
            'cpu', torch.float32,
        )
        self.assertEqual(embeds.shape, (2, 3, 8))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertFalse(mask[0, -1])
        self.assertEqual(slots.dtype, torch.bool)
        latent = torch.randn(2, 4, 2, 4)
        torch.testing.assert_close(unpack_latents(pack_latents(latent), 2, 4), latent)

    def test_text_to_image_and_edit_transformer_forward_backward(self):
        model = tiny_transformer()
        latent = torch.randn(1, 4, 2, 2, requires_grad=True)
        prompt = torch.randn(1, 3, 8)
        mask = torch.ones(1, 3, dtype=torch.bool)
        for edit in (False, True):
            slots = torch.tensor([[False, edit, False]])
            output = run_transformer(model, latent, torch.tensor([0.5]), prompt, mask, slots,
                condition_latents=torch.randn(1, 4, 4) if edit else None,
                condition_shapes=[(2, 2)] if edit else [])
            self.assertEqual(output.shape, latent.shape)
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.isfinite(latent.grad).all())

    def test_inconsistent_reference_slots_fail_before_inference(self):
        args = (tiny_transformer(), torch.randn(2, 4, 2, 2), torch.ones(2), torch.randn(2, 3, 8), torch.ones(2, 3, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, 'same reference image slot layout'):
            run_transformer(*args, torch.tensor([[False, True, False], [False, False, False]]))
        with self.assertRaisesRegex(ValueError, 'reference image slots'):
            run_transformer(*args, torch.ones(2, 3, dtype=torch.bool))

    def test_rgba_vae_encode_decode(self):
        model = AutoencoderKLQwenImage21(base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[1, 1],
            num_res_blocks=1, temperal_downsample=[False], latents_mean=[0] * 4, latents_std=[1] * 4,
            scale_factor_spatial=2, scale_factor_temporal=1)
        with torch.no_grad():
            inputs = torch.randn(1, 4, 1, 8, 8)
            latent = model.encode(inputs).latent_dist.mode()
            output = model.decode(latent).sample
        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_sampler_reports_progress_and_propagates_cancellation(self):
        events = []
        holder = SimpleNamespace(device_torch=torch.device('cpu'), torch_dtype=torch.float32,
            transformer=tiny_transformer(), get_train_scheduler=QwenImage2Model.get_train_scheduler,
            encode_condition_images=lambda images: (None, []), pad_prompt_embeds=lambda value: value,
            decode_to_images=lambda latents: [latents], sample_step_hook=True,
            _emit_sample_step=lambda latent, index, total: events.append((index, total, latent.clone())))
        prompt = (torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.bool))
        pipeline = QwenImage21Pipeline(holder)
        output = pipeline(prompt, height=32, width=32, num_inference_steps=2)
        self.assertEqual([(index, total) for index, total, _ in events], [(0, 2), (1, 2)])
        self.assertTrue(torch.isfinite(output[0]).all())
        class Cancelled(Exception):
            pass
        def cancel(*args):
            raise Cancelled()
        holder._emit_sample_step = cancel
        with self.assertRaises(Cancelled):
            pipeline(prompt, height=32, width=32, num_inference_steps=2)


if __name__ == '__main__':
    unittest.main()
