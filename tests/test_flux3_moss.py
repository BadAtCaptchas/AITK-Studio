import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import load_file, save_file

from extensions_built_in.diffusion_models import Flux3Model, MossImageModel
from extensions_built_in.diffusion_models.flux3.src.transformer import Flux3Transformer, Flux3TransformerParams
from extensions_built_in.diffusion_models.moss_image.src.transformer import MossImageTransformer, MossImageParams, _attention
from extensions_built_in.diffusion_models.moss_image.src.checkpoints import merge_safetensors, comfy_transformer_rel
from toolkit.models.v2.text_encoders.qwen35 import Qwen35TextOnlyEncoder, Qwen35VLEncoder
from toolkit.models.v2._mixin import OstrisModelMixin, _markers_from_header
from toolkit.config_modules import ModelConfig
from toolkit.model_registry import resolve_model
from toolkit.sample_controls import sample_control_paths
from toolkit.util.losses import band_loss
from toolkit.models.checkpoint_download import download_checkpoint


class NewModelTests(unittest.TestCase):
    def test_registry_and_ordered_reference_contract(self):
        self.assertIs(resolve_model('flux3'), Flux3Model)
        self.assertIs(resolve_model('moss_image'), MossImageModel)
        self.assertIs(resolve_model('moss_image:edit'), MossImageModel)
        self.assertEqual(sample_control_paths(SimpleNamespace(ctrl_imgs=[], ctrl_img='old')), [])
        self.assertEqual(sample_control_paths(SimpleNamespace(ctrl_imgs=['b', 'a', 'b'])), ['b', 'a', 'b'])

    def test_flux_tiny_reference_forward_backward_and_reload(self):
        params = Flux3TransformerParams(in_channels={'image': 4, 'image_cond': 4}, context_in_dim=12, hidden_size=32, num_heads=4, depth=1, depth_single_blocks=1, axes_dim=[2, 2, 2, 2])
        model = Flux3Transformer(params)
        reloaded = Flux3Transformer.load_from_state_dict(model.state_dict(), torch.float32, config=params)
        reloaded.enable_gradient_checkpointing()
        x = torch.randn(2, 4, 4, requires_grad=True)
        arguments = dict(ctx=torch.randn(2, 3, 12), ctx_ids=torch.zeros(2, 3, 4), timesteps_ctx=torch.zeros(2, 3), x_image=x, x_image_ids=torch.zeros(2, 4, 4), x_image_timesteps=torch.full((2, 4), .5), x_image_cond=torch.randn(2, 2, 4), x_image_cond_ids=torch.ones(2, 2, 4), x_image_cond_timesteps=torch.zeros(2, 2))
        expected = model(**arguments)['x_image']
        actual = reloaded(**arguments)['x_image']
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        self.assertEqual(actual.shape, x.shape)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_flux_reference_batches_preserve_framing_and_order(self):
        model = Flux3Model("cpu", ModelConfig(arch="flux3", name_or_path="unused"), dtype="fp32")
        self.assertTrue(model.use_raw_control_images)
        latent = torch.randn(2, 4, 2, 2)
        model.model = Mock(device=torch.device("cpu"), return_value={"x_image": torch.zeros(2, 4, 4)})
        encoded = []

        def encode(images, **kwargs):
            self.assertEqual(images.ndim, 4)
            encoded.append(images.clone())
            return images.mean().expand(1, 4, images.shape[-2] // 16, images.shape[-1] // 16)

        model.encode_images = encode
        single = torch.stack([torch.full((3, 16, 64), v) for v in (.25, .75)])
        stacked = torch.stack([torch.stack([single[i], 1 - single[i]]) for i in range(2)])
        for refs in (single, stacked):
            with self.subTest(shape=refs.shape):
                encoded.clear()
                result = model.get_noise_prediction(
                    latent, torch.tensor([500., 750.]),
                    SimpleNamespace(text_embeds=[torch.randn(2, 12), torch.randn(3, 12)]),
                    batch=SimpleNamespace(control_tensor_list=None, control_tensor=refs),
                )
                self.assertEqual(result.shape, latent.shape)
                reference_count = 1 if refs.ndim == 4 else 2
                self.assertEqual(len(encoded), 2 * reference_count)
                expected_refs = refs.reshape(-1, 3, 16, 64)
                for actual, expected in zip(encoded, expected_refs):
                    torch.testing.assert_close(actual[0], expected * 2 - 1)
                kwargs = model.model.call_args.kwargs
                self.assertEqual(kwargs["x_image_cond"].shape, (2, 4 * reference_count, 4))
                expected_ids = torch.tensor([-1.] * 4 + ([-2.] * 4 if reference_count == 2 else []))
                torch.testing.assert_close(kwargs["x_image_cond_ids"][0, :, -1], expected_ids)
                self.assertTrue((kwargs["x_image_cond_timesteps"] == 0).all())

    def test_moss_lora_export_round_trip_and_legacy_weights(self):
        model = object.__new__(MossImageModel)
        model.lora_keys_use_comfy_prefix = True
        base = "transformer.layers.0.proj.qkv."
        for down_suffix, up_suffix in (("lora_A.weight", "lora_B.weight"), ("lora_down.weight", "lora_up.weight")):
            with self.subTest(format=down_suffix):
                original = {
                    base + down_suffix: torch.randn(2, 8),
                    base + up_suffix: torch.randn(24, 2),
                    base + "alpha": torch.tensor(1.),
                    "transformer.layers.1.proj.qkv.lokr_w1": torch.randn(2, 2),
                    "transformer.layers.0.ff.weight": torch.randn(8, 8),
                }
                exported = model.convert_lora_weights_before_save(original)
                self.assertNotIn(base + down_suffix, exported)
                for i, name in enumerate("qkv"):
                    prefix = f"diffusion_model.layers.0.proj.{name}."
                    torch.testing.assert_close(exported[prefix + down_suffix], original[base + down_suffix])
                    torch.testing.assert_close(exported[prefix + up_suffix], original[base + up_suffix][i * 8:(i + 1) * 8])
                    torch.testing.assert_close(exported[prefix + "alpha"], original[base + "alpha"])
                with tempfile.TemporaryDirectory() as folder:
                    filename = os.path.join(folder, "adapter.safetensors")
                    save_file(exported, filename)
                    restored = model.convert_lora_weights_before_load(load_file(filename))
                self.assertEqual(set(restored), set(original))
                for key, value in original.items():
                    torch.testing.assert_close(restored[key], value)
                    torch.testing.assert_close(model.convert_lora_weights_before_load(original)[key], value)

    def test_moss_independent_qkv_loras_keep_their_effective_weights(self):
        model = object.__new__(MossImageModel)
        model.lora_keys_use_comfy_prefix = True
        split, updates = {}, []
        for name in "qkv":
            down, up = torch.randn(2, 8), torch.randn(8, 2)
            prefix = f"diffusion_model.layers.0.proj.{name}."
            split.update({prefix + "lora_A.weight": down, prefix + "lora_B.weight": up, prefix + "alpha": torch.tensor(1.)})
            updates.append(up @ down / 2)
        restored = model.convert_lora_weights_before_load(split)
        prefix = "transformer.layers.0.proj.qkv."
        down, up = restored[prefix + "lora_A.weight"], restored[prefix + "lora_B.weight"]
        self.assertEqual(down.shape, (6, 8))
        self.assertEqual(len(split), 9)
        effective = up @ down * (restored[prefix + "alpha"] / down.shape[0])
        torch.testing.assert_close(effective, torch.cat(updates))

    def test_moss_unmasked_reference_isolation_matches_masked_attention_and_gradients(self):
        q, k, v = (torch.randn(2, 7, 16, requires_grad=True) for _ in range(3))
        phases = torch.randn(2, 7, 4)
        rope = torch.cat((phases.sin(), phases.cos()), dim=-1)
        mask = torch.ones(1, 1, 7, 7, dtype=torch.bool)
        mask[:, :, 4:, :4] = False
        expected = _attention(q, k, v, rope, heads=2, mask=mask)
        actual = _attention(q, k, v, rope, heads=2, mask=None, isolate_split=4)
        torch.testing.assert_close(actual, expected)
        grad_output = torch.randn_like(expected)
        expected_grads = torch.autograd.grad(expected, (q, k, v), grad_output)
        actual_grads = torch.autograd.grad(actual, (q, k, v), grad_output)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_moss_tiny_reference_cache_and_checkpoint_gradients(self):
        params = MossImageParams(dim=32, ffn_dim=64, num_heads=4, num_layers=2, text_dim=16, freq_dim=16, latent_channels=4)
        model = MossImageTransformer(params)
        saved = model.convert_state_dict_on_save(model.state_dict())
        reloaded = MossImageTransformer.load_from_state_dict(saved, torch.float32)
        reloaded.enable_gradient_checkpointing()
        for padded in (False, True):
            with self.subTest(padded=padded):
                x = torch.randn(2, 4, 2, 2, requires_grad=True)
                text_mask = torch.tensor([[1, 1, 1], [1, 0, 0] if padded else [1, 1, 1]])
                args = dict(hidden_states=x, timestep=torch.tensor([0., 1000.]), encoder_hidden_states=torch.randn(2, 3, 16), encoder_attention_mask=text_mask, ref_latents=[[torch.randn(4, 1, 2)], [torch.randn(4, 2, 1)]], isolate_refs=True)
                expected = model(**args)
                actual = reloaded(**args)
                torch.testing.assert_close(actual, expected)
                actual.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                cache = {}
                with torch.no_grad():
                    initial = model(**args, ref_kv_cache=cache)
                    cached = model(**args, ref_kv_cache=cache)
                torch.testing.assert_close(initial, cached)

    def test_qwen35_checkpoint_conversion(self):
        weight = torch.randn(4, 4)
        source = {'model.language_model.layers.0.weight': weight, 'model.visual.weight': weight, 'lm_head.weight': weight, 'mtp.weight': weight}
        text = Qwen35TextOnlyEncoder.convert_state_dict_on_load(source)
        self.assertEqual(set(text), {'layers.0.weight'})
        torch.testing.assert_close(Qwen35TextOnlyEncoder.convert_state_dict_on_save(text)['model.language_model.layers.0.weight'], weight)
        vision = Qwen35VLEncoder.convert_state_dict_on_load(source)
        self.assertEqual(set(vision), {'language_model.layers.0.weight', 'visual.weight'})

    def test_qwen35_tiny_encoder_gradients_and_reload(self):
        from transformers import Qwen3_5TextConfig
        config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                  num_hidden_layers=1, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=8, layer_types=['full_attention'],
                                  rope_parameters={'rope_type': 'default', 'rope_theta': 10000, 'partial_rotary_factor': 1.0, 'mrope_section': [1, 1, 2]})
        encoder = Qwen35TextOnlyEncoder(config)
        reloaded = Qwen35TextOnlyEncoder(config)
        saved = encoder.convert_state_dict_on_save(encoder.state_dict())
        reloaded.load_state_dict(reloaded.convert_state_dict_on_load(saved))
        tokens = torch.tensor([[1, 2, 3]])
        actual = reloaded(tokens, use_cache=False).last_hidden_state
        torch.testing.assert_close(actual, encoder(tokens, use_cache=False).last_hidden_state)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(reloaded.embed_tokens.weight.grad).all())

    def test_shard_merge_and_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            shards = [os.path.join(root, f'{i}.safetensors') for i in range(3)]
            for name, filename in zip('qkv', shards): save_file({f'block.proj.{name}.weight': torch.ones(2, 4)}, filename)
            output = os.path.join(root, 'merged.safetensors')
            merge_safetensors(shards, output, fuse_qkv=True, chunk_bytes=8)
            self.assertEqual(load_file(output)['block.proj.qkv.weight'].shape, (6, 4))
            self.assertEqual(load_file(output)['block.proj.qkv.weight'].dtype, torch.bfloat16)
            with open(shards[0], 'r+b') as handle: handle.truncate(20)
            before = Path(output).read_bytes()
            with self.assertRaises(Exception): merge_safetensors(shards, output, fuse_qkv=True)
            self.assertEqual(Path(output).read_bytes(), before)
            self.assertFalse(any(name.endswith('.part') for name in os.listdir(root)))
        self.assertNotEqual(comfy_transformer_rel('one/custom'), comfy_transformer_rel('two/custom'))

    def test_coordinated_download_and_failed_publication(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as root:
            target = os.path.join(root, 'weights.safetensors')
            calls = []
            def download(repo, filename, **kwargs):
                calls.append(kwargs)
                source = os.path.join(kwargs['local_dir'], filename)
                save_file({'weight': torch.ones(2, 2)}, source)
                return source
            with patch('huggingface_hub.hf_hub_download', side_effect=download), patch('toolkit.models.checkpoint_download.is_hf_offline_mode', return_value=True):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda _: download_checkpoint('test/repo', 'weights.safetensors', target), range(2)))
            self.assertEqual(results, [target, target])
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0]['local_files_only'])
            self.assertFalse(any(name.startswith('.aitk-download-') for name in os.listdir(root)))
            before = Path(target).read_bytes()
            with self.assertRaisesRegex(ValueError, 'Invalid checkpoint filename'):
                download_checkpoint('test/repo', '../weights.safetensors', target)
            self.assertEqual(Path(target).read_bytes(), before)
            invalid = os.path.join(root, 'invalid.safetensors')
            def corrupt(repo, filename, **kwargs):
                source = os.path.join(kwargs['local_dir'], filename)
                Path(source).write_bytes(b'truncated')
                return source
            with patch('huggingface_hub.hf_hub_download', side_effect=corrupt):
                with self.assertRaises(Exception): download_checkpoint('test/repo', 'weights.safetensors', invalid)
            self.assertFalse(os.path.exists(invalid))
            self.assertFalse(any(name.startswith('.aitk-download-') for name in os.listdir(root)))

    def test_incomplete_tokenizer_recovery_and_offline(self):
        class Tokenizer:
            def __init__(self, length): self.length = length
            def __len__(self): return self.length
        calls = []
        def loader(**kwargs):
            calls.append(kwargs)
            return Tokenizer(1 if kwargs.get('local_files_only') else 2000)
        with patch('toolkit.hf_offline.is_hf_offline_mode', return_value=False):
            recovered = OstrisModelMixin._local_first(loader, validate=OstrisModelMixin._tokenizer_complete)
            self.assertEqual(len(recovered), 2000)
            with self.assertRaisesRegex(ValueError, 'Incomplete'): OstrisModelMixin._local_first(lambda **kwargs: Tokenizer(1), validate=OstrisModelMixin._tokenizer_complete)
        with patch('toolkit.hf_offline.is_hf_offline_mode', return_value=True):
            with self.assertRaisesRegex(ValueError, 'Incomplete'): OstrisModelMixin._local_first(loader, local_files_only=True, validate=OstrisModelMixin._tokenizer_complete)
        self.assertEqual(len(calls), 3)

    def test_band_loss_shapes_boundaries_and_finite_gradients(self):
        for shape in [(2, 4, 8, 9), (2, 4, 3, 8, 9), (2, 4, 1, 1), (2, 4, 7)]:
            for flow in [True, False]:
                pred = torch.randn(shape, requires_grad=True)
                target, latents, noise = (torch.randn(shape) for _ in range(3))
                scheduler = SimpleNamespace(alphas_cumprod=torch.linspace(1., 0., 1000))
                result = band_loss(pred, target, latents, noise, torch.tensor([0., 1000.]), flow, scheduler)
                self.assertEqual(result.shape, pred.shape)
                self.assertTrue(torch.isfinite(result).all())
                result.mean().backward()
                self.assertTrue(torch.isfinite(pred.grad).all())
                if len(shape) == 3: torch.testing.assert_close(result, (pred - target).square())


if __name__ == '__main__':
    unittest.main()
