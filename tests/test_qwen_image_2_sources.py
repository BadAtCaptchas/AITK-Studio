"""Exercise Qwen's production loading policy without downloading weights."""

import unittest
from unittest.mock import Mock, call, patch

import torch

from extensions_built_in.diffusion_models.qwen_image_2 import qwen_image_2 as qwen
from toolkit.config_modules import ModelConfig


class QwenImage2SourceTests(unittest.TestCase):
    def check_text_encoder_policy(self, *, quantize_te, qtype_te, use_comfy_weights=True):
        holder = qwen.QwenImage2Model(
            "cpu",
            ModelConfig(
                arch="qwen_image_2",
                name_or_path=qwen.COMFY_REPO,
                quantize=False,
                quantize_te=quantize_te,
                qtype_te=qtype_te,
                model_kwargs={"use_comfy_weights": use_comfy_weights},
            ),
            dtype="bf16",
        )
        holder.print_and_status_update = Mock()
        text_encoder = Mock()
        transformer = Mock()
        vae = Mock()
        processor = Mock()
        events = Mock()
        expected_policy = holder.component_load_kwargs("te")

        with patch.object(qwen.QwenImage21Transformer2DModel, "load", return_value=transformer), \
             patch.object(qwen.QwenImage21TextEncoder, "load_processor", return_value=processor) as load_processor, \
             patch.object(qwen.QwenImage21TextEncoder, "load_model", return_value=text_encoder) as load_encoder, \
             patch.object(qwen.AutoencoderKLQwenImage21, "load", return_value=vae), \
             patch.object(qwen, "QwenImage21PromptEncoder") as make_prompt_encoder, \
             patch.object(qwen, "QwenImage21Pipeline") as make_pipeline, \
             patch.object(qwen, "flush"):
            events.attach_mock(load_encoder, "load")
            events.attach_mock(text_encoder.patch_vision_patch_embed, "patch_vision")
            events.attach_mock(text_encoder.aitk_post_load, "post_load")
            holder.load_model()

        requested_qtype = qtype_te if quantize_te else None
        source_qtype = (requested_qtype or "").split("|", 1)[0] or None
        # A quantization request guides checkpoint selection, but the initial
        # load must not quantize before the vision patch or quantize twice.
        self.assertEqual(events.mock_calls, [
            call.load(
                qwen.BASE_REPO,
                dtype=torch.bfloat16,
                subfolder="text_encoder",
                use_comfy_weights=use_comfy_weights,
                qtype=source_qtype,
                quantize_on_load=False,
            ),
            call.patch_vision(),
            call.post_load(**expected_policy),
        ])
        self.assertEqual(expected_policy["qtype"], requested_qtype)
        text_encoder.aitk_post_load.assert_called_once()
        text_encoder.quantize_.assert_not_called()
        text_encoder.requires_grad_.assert_called_once_with(False)
        text_encoder.eval.assert_called_once_with()
        load_processor.assert_called_once_with(qwen.BASE_REPO)
        make_prompt_encoder.assert_called_once_with(text_encoder, processor)
        make_pipeline.assert_called_once_with(holder)
        self.assertIs(holder.model, transformer)
        self.assertIs(holder.vae, vae)
        self.assertEqual(holder.text_encoder, [text_encoder])
        self.assertEqual(holder.tokenizer, [processor.tokenizer])

    def test_unquantized_text_encoder_requests_full_precision_source(self):
        # A stale qtype setting must not override quantize_te=False.
        self.check_text_encoder_policy(quantize_te=False, qtype_te="convrot8")

    def test_quantized_text_encoder_requests_matching_source(self):
        self.check_text_encoder_policy(quantize_te=True, qtype_te="convrot8")

    def test_comfy_opt_out_reaches_initial_text_encoder_load(self):
        for quantize_te in (False, True):
            with self.subTest(quantize_te=quantize_te):
                self.check_text_encoder_policy(
                    quantize_te=quantize_te, qtype_te="convrot8", use_comfy_weights=False,
                )

    def test_recovery_adapter_suffix_is_only_removed_for_source_selection(self):
        self.check_text_encoder_policy(
            quantize_te=True, qtype_te="convrot8|test/recovery.safetensors",
        )


if __name__ == "__main__":
    unittest.main()
