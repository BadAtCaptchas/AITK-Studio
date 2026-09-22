"""Ming component tests: native gradients and checkpoint semantics, no weights."""

import tempfile
import unittest
import os
from types import SimpleNamespace

import torch
from diffusers import ZImageTransformer2DModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from extensions_built_in.diffusion_models.ming_image.src.transformer import MingImageTransformer2DModel
from extensions_built_in.diffusion_models.ming_image.src.vae import encode_rgba_frames, decode_rgba_frames
from extensions_built_in.diffusion_models.ming_image.src.bailing import (
    BailingBlock, BailingGate, BailingMoE, BailingRMSNorm, apply_video_rotary, video_rotary_embeddings,
)
from extensions_built_in.diffusion_models.ming_image.src.conditioning import MingConditioner, ming_positions


def tiny_kwargs():
    return dict(in_channels=4, dim=16, n_layers=1, n_refiner_layers=1,
                n_heads=2, n_kv_heads=2, cap_feat_dim=8,
                axes_dims=(2, 2, 4), axes_lens=(256, 64, 64))


def tiny_model(layered=True):
    return MingImageTransformer2DModel(**tiny_kwargs(),
        alignment_padding_mode="learned" if layered else "zero_masked",
        multi_frame_output=layered)


class FrameVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.zeros(()))
        self.config = SimpleNamespace(scaling_factor=8.0064, shift_factor=0.25)
        self.encoded = []
        self.decoded = []

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype

    def encode(self, images):
        self.encoded.append(images.clone())
        mean = images + 0.5
        return SimpleNamespace(latent_dist=DiagonalGaussianDistribution(torch.cat((mean, torch.zeros_like(mean)), dim=1)))

    def decode(self, latents):
        self.decoded.append(latents.clone())
        return SimpleNamespace(sample=latents - 0.5)


class MingComponentsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)

    def test_learned_mode_matches_diffusers_blocks_without_extra_condition(self):
        reference = ZImageTransformer2DModel(**tiny_kwargs()).eval()
        model = tiny_model().eval()
        model.load_state_dict(reference.state_dict(), strict=True)
        x = [torch.randn(4, 3, 4, 6)]
        caption = [torch.randn(5, 8)]
        time = torch.tensor([0.35])
        expected = reference(x, time, caption, return_dict=False)[0][0]
        actual = model(x, time, caption, return_dict=False)[0][0]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_reference_is_conditioning_and_only_target_frames_returned(self):
        model = tiny_model()
        target = torch.randn(4, 3, 4, 6)
        direct = torch.randn(7, 16, requires_grad=True)
        reference = torch.randn(4, 1, 4, 6, requires_grad=True)
        prediction = model([target], torch.tensor([0.4]), [torch.randn(5, 8)],
            ref_x=[reference], cap_feats_2=[direct], return_dict=False)[0][0]
        self.assertEqual(prediction.shape, target.shape)
        prediction.square().mean().backward()
        self.assertGreater(float(direct.grad.abs().sum()), 0)
        self.assertGreater(float(reference.grad.abs().sum()), 0)
        self.assertGreater(float(model.layers[0].attention.to_q.weight.grad.abs().sum()), 0)

    def test_checkpointing_preserves_forward_and_gradients(self):
        model = tiny_model()
        checkpointed = tiny_model()
        checkpointed.load_state_dict(model.state_dict())
        checkpointed.enable_gradient_checkpointing()
        x, cap, direct = [torch.randn(4, 2, 4, 4)], [torch.randn(9, 8)], [torch.randn(3, 16)]
        args = (x, torch.tensor([0.2]), cap)
        expected = model(*args, cap_feats_2=direct, return_dict=False)[0][0]
        actual = checkpointed(*args, cap_feats_2=direct, return_dict=False)[0][0]
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(actual, expected)
        for (name, parameter), (_, other) in zip(model.named_parameters(), checkpointed.named_parameters()):
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                torch.testing.assert_close(parameter.grad, other.grad)

    def test_padding_attention_policy_and_checkpoint_keys(self):
        for layered in (False, True):
            model = tiny_model(layered)
            recorded = []
            hook = model.noise_refiner[0].register_forward_pre_hook(lambda module, args: recorded.append(args[1].clone()))
            model([torch.randn(4, 1, 4, 6)], torch.tensor([0.5]), [torch.randn(3, 8)], return_dict=False)
            hook.remove()
            mask = recorded[0][0]
            self.assertTrue(mask[:6].all())
            self.assertEqual(bool(mask[6:].all()), layered)
            self.assertEqual("x_pad_token" in model.state_dict(), layered)
            self.assertEqual("cap_pad_token" in model.state_dict(), layered)

    def test_variable_caption_batch_matches_individual_predictions(self):
        model = tiny_model(False).eval()
        x = [torch.randn(4, 1, 4, 6), torch.randn(4, 1, 6, 4)]
        caps = [torch.randn(2, 8), torch.randn(35, 8)]
        direct = [torch.randn(5, 16), torch.randn(1, 16)]
        time = torch.tensor([0.4, 0.8])
        batch = model(x, time, caps, cap_feats_2=direct, return_dict=False)[0]
        for i in range(2):
            single = model([x[i]], time[i:i + 1], [caps[i]], cap_feats_2=[direct[i]], return_dict=False)[0][0]
            torch.testing.assert_close(batch[i], single, rtol=1e-4, atol=1e-5)

    def test_native_save_and_mixin_load_round_trip(self):
        for layered in (False, True):
            model = tiny_model(layered)
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                restored = MingImageTransformer2DModel.load(directory, dtype=torch.float32, use_comfy_weights=False)
                self.assertEqual(restored.multi_frame_output, layered)
                self.assertEqual(set(model.state_dict()), set(restored.state_dict()))
                for key, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, restored.state_dict()[key], rtol=0, atol=0)

    def test_real_lora_backward_through_checkpointed_frozen_transformer(self):
        from peft import LoraConfig
        model = tiny_model()
        model.requires_grad_(False)
        model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_v"]))
        model.enable_gradient_checkpointing()
        output = model([torch.randn(4, 3, 4, 4)], torch.tensor([0.5]), [torch.randn(4, 8)],
            cap_feats_2=[torch.randn(3, 16)], return_dict=False)[0][0]
        output.square().mean().backward()
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all("lora_" in name for name, _ in trainable))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for _, parameter in trainable))
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for _, parameter in trainable), 0)

    def test_ming_fp8_storage_remains_swappable_and_lora_backward_is_finite(self):
        from peft import LoraConfig
        from extensions_built_in.diffusion_models.ming_image.src.quantization import register_ming_fp8
        from toolkit.memory_management.block_offload import BlockOffloadManager, resolve_block_layers
        from toolkit.util.ostris_quant import OstrisLinear
        from toolkit.util.quantize import get_qtype, quantize

        register_ming_fp8()
        model = tiny_model().requires_grad_(False)
        quantize(model, weights=get_qtype("ming_fp8"), exclude=[
            "t_embedder.*", "cap_embedder.*", "all_x_embedder.*", "all_final_layer.*"])
        packed = [module for module in model.modules() if isinstance(module, OstrisLinear)]
        self.assertTrue(packed)
        original = [module.ming_fp8_data.clone() for module in packed]
        model.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        for module, data in zip(packed, original):
            self.assertEqual(module.ming_fp8_data.dtype, torch.uint8)
            self.assertTrue(torch.equal(data, module.ming_fp8_data))
        blocks = resolve_block_layers(model, model.get_transformer_block_names())
        candidates, skipped = BlockOffloadManager._build_layer_candidates(blocks, set())
        self.assertEqual(len(candidates), len(blocks))
        self.assertFalse(skipped)
        model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_v"]))
        model.enable_gradient_checkpointing()
        output = model([torch.randn(4, 3, 4, 4)], torch.tensor([0.5]), [torch.randn(4, 8)],
            cap_feats_2=[torch.randn(3, 16)], return_dict=False)[0][0]
        output.square().mean().backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None and gradient.isfinite().all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)

    def test_ming_fp8_packed_cache_restores_meta_model_without_widening_weights(self):
        from toolkit.quantized_cache import QuantizedModelCache, quantized_cache_key
        from toolkit.util.ostris_quant import OstrisLinear
        from toolkit.util.quantize import get_qtype, quantize

        model = tiny_model().eval().requires_grad_(False)
        quantize(model, weights=get_qtype("ming_fp8"))
        inputs = ([torch.randn(4, 2, 4, 4)], torch.tensor([0.5]), [torch.randn(4, 8)])
        expected = model(*inputs, return_dict=False)[0][0]
        with tempfile.TemporaryDirectory() as directory:
            cache = QuantizedModelCache(directory)
            key, payload = quantized_cache_key("ming-test", {"qtype": "ming_fp8"}, sources=[])
            cache.save(model, "transformer", key, payload)
            with torch.device("meta"):
                restored = tiny_model()
            cache.load(restored, "transformer", key, device=torch.device("cpu"))
            actual = restored(*inputs, return_dict=False)[0][0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for (name, module), (_, other) in zip(model.named_modules(), restored.named_modules()):
                if isinstance(module, OstrisLinear):
                    self.assertIsInstance(other, OstrisLinear, name)
                    self.assertEqual(other.ming_fp8_data.dtype, torch.uint8)
                    self.assertTrue(torch.equal(module.ming_fp8_data, other.ming_fp8_data), name)
                    self.assertTrue(torch.equal(module.ming_fp8_scale, other.ming_fp8_scale), name)

    def test_bfloat16_checkpointed_ming_fp8_lora_gradients_are_finite(self):
        from peft import LoraConfig
        from toolkit.util.quantize import get_qtype, quantize

        for layered in (False, True):
            with self.subTest(layered=layered):
                model = tiny_model(layered).to(torch.bfloat16).requires_grad_(False)
                quantize(model, weights=get_qtype("ming_fp8"), exclude=[
                    "t_embedder.*", "cap_embedder.*", "all_x_embedder.*", "all_final_layer.*"])
                model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_v"]))
                model.to(torch.bfloat16)
                model.enable_gradient_checkpointing()
                frames = 3 if layered else 1
                output = model([torch.randn(4, frames, 4, 4, dtype=torch.bfloat16)], torch.tensor([0.5]),
                    [torch.randn(4, 8, dtype=torch.bfloat16)],
                    cap_feats_2=[torch.randn(3, 16, dtype=torch.bfloat16)], return_dict=False)[0][0]
                self.assertTrue(output.isfinite().all())
                output.float().square().mean().backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(parameter.grad.isfinite().all(), name)

    @unittest.skipUnless(os.environ.get("AITK_TEST_MING_CUDA") == "1" and torch.cuda.is_available(),
                         "Opt-in CUDA offload parity check")
    def test_cuda_toolkit_lora_offload_matches_resident_gradients_and_update(self):
        from extensions_built_in.diffusion_models.ming_image import MingImageDesignModel, MingImageDesignLayerModel
        from toolkit.config_modules import ModelConfig, NetworkConfig
        from toolkit.lora_special import LoRASpecialNetwork
        from toolkit.memory_management.block_offload import BlockOffloadManager
        from toolkit.util.quantize import get_qtype, quantize

        for layered in (False, True):
            with self.subTest(layered=layered):
                torch.manual_seed(93)
                kwargs = dict(in_channels=4, dim=128, n_layers=3, n_refiner_layers=2,
                    n_heads=2, n_kv_heads=2, cap_feat_dim=32,
                    axes_dims=(16, 24, 24), axes_lens=(512, 64, 64),
                    alignment_padding_mode="learned" if layered else "zero_masked", multi_frame_output=layered)
                models = [MingImageTransformer2DModel(**kwargs).to(torch.bfloat16) for _ in range(2)]
                models[1].load_state_dict(models[0].state_dict())
                networks, optimizers = [], []
                for index, model in enumerate(models):
                    model.requires_grad_(False).enable_gradient_checkpointing()
                    quantize(model, weights=get_qtype("ming_fp8"), exclude=model.get_quantization_exclude_modules())
                    if index:
                        BlockOffloadManager.attach(model, torch.device("cuda:0"),
                            block_paths=model.get_transformer_block_names())
                    else:
                        model.to("cuda:0")
                    cls = MingImageDesignLayerModel if layered else MingImageDesignModel
                    holder = cls("cuda:0", ModelConfig(arch=cls.arch, name_or_path="unused"), dtype="bf16")
                    holder.model, holder.text_encoder = model, []
                    network = LoRASpecialNetwork(text_encoder=[], unet=model, train_text_encoder=False,
                        train_unet=True, lora_dim=4, alpha=4, target_lin_modules=holder.target_lora_modules,
                        is_transformer=True, network_config=NetworkConfig(type="lora", linear=4, linear_alpha=4),
                        base_model=holder)
                    # Keep the base holder alive: network stores a weak ref.
                    network._test_holder = [holder]
                    network.force_to("cuda:0", dtype=torch.float32)
                    network._update_torch_multiplier()
                    network.apply_to([], model, False, True)
                    networks.append(network)
                    optimizers.append(torch.optim.AdamW(network.parameters(), lr=1e-4))
                networks[1].load_state_dict(networks[0].state_dict())
                x = [torch.randn(4, 3 if layered else 1, 16, 16, device="cuda", dtype=torch.bfloat16)]
                captions = [torch.randn(35, 32, device="cuda", dtype=torch.bfloat16)]
                direct = [torch.randn(17, 128, device="cuda", dtype=torch.bfloat16)]
                reference = [torch.randn(4, 1, 16, 16, device="cuda", dtype=torch.bfloat16)]
                time = torch.tensor([0.35], device="cuda")
                try:
                    for step in range(2):
                        outputs, gradients = [], []
                        for model, network, optimizer in zip(models, networks, optimizers):
                            optimizer.zero_grad(set_to_none=True)
                            with network:
                                output = model(x, time, captions, cap_feats_2=direct, ref_x=reference,
                                    return_dict=False)[0][0]
                                output.float().square().mean().backward()
                            outputs.append(output.detach())
                            gradients.append(torch.cat([parameter.grad.detach().flatten()
                                for parameter in network.parameters() if parameter.grad is not None]))
                        torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-4, atol=1e-6)
                        torch.testing.assert_close(gradients[1], gradients[0], rtol=1e-4, atol=1e-6)
                        relative_error = (gradients[1]-gradients[0]).norm() / gradients[0].norm().clamp_min(1e-12)
                        print(f"Ming CUDA parity layered={layered} step={step}: gradient_relative_error={float(relative_error):.8g}")
                        for optimizer in optimizers:
                            optimizer.step()
                        for first, second in zip(networks[0].parameters(), networks[1].parameters()):
                            torch.testing.assert_close(second, first, rtol=1e-4, atol=1e-6)
                finally:
                    torch.cuda.synchronize()
                    models[1].to("cpu")
                    models[1]._block_offload_manager.detach()
                    models[0].to("cpu")

    def test_rgba_frames_are_independent_deterministic_and_round_trip(self):
        vae = FrameVAE()
        images = torch.linspace(-1, 1, 2 * 4 * 3 * 4 * 6).reshape(2, 4, 3, 4, 6)
        rng = torch.random.get_rng_state().clone()
        latent = encode_rgba_frames(vae, images)
        self.assertEqual(len(vae.encoded), 6)
        self.assertTrue(all(item.shape == (1, 4, 1, 4, 6) for item in vae.encoded))
        torch.testing.assert_close(latent, (images + 0.25) * 8.0064)
        torch.testing.assert_close(decode_rgba_frames(vae, latent), images)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertFalse(vae.training)

    def test_rgb_inputs_get_opaque_alpha(self):
        vae = FrameVAE()
        images = torch.zeros(2, 3, 4, 6)
        latent = encode_rgba_frames(vae, images)
        self.assertEqual(latent.shape, (2, 4, 4, 6))
        self.assertTrue((vae.encoded[0][:, 3] == 1).all())


def tiny_bailing_config():
    return SimpleNamespace(hidden_size=16, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, use_qkv_bias=False, use_bias=False, rms_norm_eps=1e-6,
        partial_rotary_factor=0.5, rope_theta=600000, mrope_section=(0, 1, 1),
        num_experts=4, n_group=2, topk_group=1, num_experts_per_tok=2,
        routed_scaling_factor=2.5, router_type="MultiRouter", moe_intermediate_size=8,
        num_shared_experts=1, first_k_dense_replace=1, intermediate_size=32,
        image_start_token=3, image_patch_token=4, image_end_token=5, max_position_embeddings=512)


class MingConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(57)

    def test_reference_positions_and_query_suffix_follow_grid_maximum(self):
        actual = ming_positions(12, reference_start=2, reference_grid=(1, 2, 3))
        expected = torch.tensor([[0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7, 8],
                                 [0, 1, 2, 2, 2, 3, 3, 3, 5, 6, 7, 8],
                                 [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8]])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(ming_positions(5), torch.arange(5).expand(3, -1))
        with self.assertRaises(ValueError):
            ming_positions(3, reference_start=2, reference_grid=(1, 2, 3))

    def test_video_rotary_matches_upstream_axis_selection_and_preserves_tail(self):
        config = SimpleNamespace(head_dim=128, partial_rotary_factor=0.5, rope_theta=600000)
        positions = torch.tensor([[0, 2, 4], [0, 3, 6], [0, 5, 10]])
        cos, sin = video_rotary_embeddings(positions, config, torch.float32)
        inverse = 600000 ** (-torch.arange(0, 64, 2).float() / 64)
        # Original Bailing video_rope splits the duplicated frequency vector
        # into [24,8,24,8]: alternating H/W, then T, in each half.
        frequencies = positions.float()[:, :, None] * inverse
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        axes = ([1, 2] * 12 + [0] * 8) * 2
        expected = torch.stack([frequencies[axis, :, i] for i, axis in enumerate(axes)], dim=-1)
        torch.testing.assert_close(cos, expected.cos())
        torch.testing.assert_close(sin, expected.sin())
        query, key = torch.randn(1, 2, 3, 128), torch.randn(1, 1, 3, 128)
        rotated, _ = apply_video_rotary(query, key, cos, sin)
        torch.testing.assert_close(rotated[..., 64:], query[..., 64:])
        torch.testing.assert_close(rotated[..., :64].square().sum(-1), query[..., :64].square().sum(-1))

    def test_multimodal_expert_routing_uses_image_gate_and_original_scores(self):
        config = tiny_bailing_config()
        moe = BailingMoE(config)
        with torch.no_grad():
            for gate in (moe.gate, moe.image_gate, moe.audio_gate):
                gate.weight.zero_()
            # Bias determines selection; unbiased sigmoid scores determine
            # the mixture weights, which must therefore be 1.25 each.
            moe.gate.expert_bias.copy_(torch.tensor([4., 3., 0., 0.]))
            moe.image_gate.expert_bias.copy_(torch.tensor([0., 0., 4., 3.]))
        hidden = torch.randn(1, 3, 16)
        mask = torch.tensor([[False, True, False]])
        actual = moe(hidden, mask)
        expected = moe.shared_experts(hidden)
        for token, experts in enumerate(((0, 1), (2, 3), (0, 1))):
            expected[:, token] += sum(moe.experts[index](hidden[:, token]) for index in experts) * 1.25
        torch.testing.assert_close(actual, expected)

    def test_bfloat16_router_promotes_near_tie_scores_before_sigmoid_and_bias(self):
        config = tiny_bailing_config()
        config.num_experts_per_tok, config.topk_group = 1, 2
        gate = BailingGate(config).to(torch.bfloat16)
        with torch.no_grad():
            gate.weight.zero_()
            gate.weight[:, 0] = torch.tensor([0., 0.0002, 0.0004, 0.0006])
        hidden = torch.zeros(1, 16, dtype=torch.bfloat16)
        hidden[0, 0] = 1
        ids, weights = gate(hidden)
        self.assertEqual(ids.item(), 3)
        self.assertEqual(weights.dtype, torch.float32)
        expected = gate.weight[3, 0].float().sigmoid() * 2.5
        torch.testing.assert_close(weights[0, 0], expected)
        self.assertGreater(weights.item(), 1.2501)
        with torch.no_grad():
            gate.expert_bias[0] = 2 ** -11
        ids, weights = gate(hidden)
        self.assertEqual(ids.item(), 0)
        self.assertEqual(weights.item(), 1.25)

    def test_staging_dtype_changes_and_recursive_free(self):
        holder = MingConditioner()
        holder.child = torch.nn.Linear(4, 4)
        holder.to(dtype=torch.bfloat16)
        self.assertEqual(holder.dtype, torch.bfloat16)
        self.assertEqual(holder.child.weight.dtype, torch.float32)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            with holder._stage(holder.child):
                self.assertEqual(holder.child.weight.dtype, torch.bfloat16)
                raise RuntimeError("cancelled")
        self.assertEqual(holder.child.weight.device.type, "cpu")
        torch.nn.Module.to(holder, "meta")
        self.assertTrue(holder.child.weight.is_meta)

    def test_real_native_backbone_and_connector_preserve_causal_prefix_but_mix_queries(self):
        from transformers import Qwen2Config, Qwen2Model

        class Tokenizer:
            def __call__(self, text, return_tensors):
                self_test.assertEqual(text, "ab<image>" + "<imagePatch>" * 256 + "</image>")
                return {"input_ids": torch.tensor([[0, 1, 3] + [4] * 256 + [5]])}

        self_test = self
        config = tiny_bailing_config()
        holder = MingConditioner().to(dtype=torch.float32)
        holder.config, holder.selected_layers, holder.tokenizer = config, (1, 2, 3), Tokenizer()
        holder.connector_norm = holder.text_encoder_norm = False
        holder.word_embeddings = torch.nn.Embedding(8, 16)
        holder.blocks = torch.nn.ModuleList([BailingBlock(config, index) for index in range(3)])
        holder.final_norm = BailingRMSNorm(16)
        holder.query_tokens = torch.nn.Parameter(torch.randn(256, 16))
        connector_config = Qwen2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, vocab_size=8)
        connector_config._attn_implementation = "sdpa"
        holder.connector = Qwen2Model(connector_config)
        holder.connector.layers[0].self_attn.is_causal = False
        holder.proj_in, holder.proj_out = torch.nn.Linear(16, 16), torch.nn.Linear(16, 12)
        holder.direct_projector = torch.nn.Sequential(torch.nn.RMSNorm(48), torch.nn.Linear(48, 20))
        with torch.no_grad():
            for module in holder.modules():
                if isinstance(module, BailingGate):
                    module.weight.normal_(std=0.1)
        holder.eval().requires_grad_(False)
        caption, direct = holder.encode("ab")
        self.assertEqual(caption.shape, (256, 12))
        self.assertEqual(direct.shape, (2, 20))
        self.assertTrue(caption.isfinite().all())
        self.assertFalse(caption.requires_grad)
        holder.query_tokens[-1].add_(torch.arange(16) * 20)
        other_caption, other_direct = holder.encode("ab")
        # Expert batch sizes can change GEMM rounding for the same prefix.
        torch.testing.assert_close(direct, other_direct, atol=1e-6, rtol=1e-5)
        self.assertGreater(float((caption[0] - other_caption[0]).abs().max()), 1e-6)
        with self.assertRaisesRegex(ValueError, "reserved"):
            holder.encode("<image>")


if __name__ == "__main__":
    unittest.main()
