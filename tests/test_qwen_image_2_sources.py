"""Exercise Qwen's production loading policy without downloading weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from extensions_built_in.diffusion_models.qwen_image_2 import qwen_image_2 as qwen
from toolkit.config_modules import ModelConfig
from extensions_built_in.inference_engine.engine import Engine


class QwenImage2SourceTests(unittest.TestCase):
    def check_text_encoder_policy(
        self, *, quantize_te, qtype_te, use_comfy_weights=True, name_or_path=qwen.COMFY_REPO,
    ):
        model_kwargs = {} if use_comfy_weights is None else {"use_comfy_weights": use_comfy_weights}
        expected_comfy = name_or_path == qwen.COMFY_REPO if use_comfy_weights is None else use_comfy_weights
        holder = qwen.QwenImage2Model(
            "cpu",
            ModelConfig(
                arch="qwen_image_2",
                name_or_path=name_or_path,
                quantize=False,
                quantize_te=quantize_te,
                qtype_te=qtype_te,
                model_kwargs=dict(model_kwargs),
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

        with patch.object(qwen.QwenImage21Transformer2DModel, "load", return_value=transformer) as load_transformer, \
             patch.object(qwen.QwenImage21TextEncoder, "load_processor", return_value=processor) as load_processor, \
             patch.object(qwen.QwenImage21TextEncoder, "load_model", return_value=text_encoder) as load_encoder, \
             patch.object(qwen.AutoencoderKLQwenImage21, "load", return_value=vae) as load_vae, \
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
                use_comfy_weights=expected_comfy,
                qtype=source_qtype,
                quantize_on_load=False,
            ),
            call.patch_vision(),
            call.post_load(**expected_policy),
        ])
        self.assertEqual(expected_policy["qtype"], requested_qtype)
        self.assertIs(expected_policy["use_comfy_weights"], expected_comfy)
        self.assertIs(load_transformer.call_args.kwargs["use_comfy_weights"], expected_comfy)
        self.assertEqual(load_transformer.call_args.args, (name_or_path,))
        self.assertIs(load_vae.call_args.kwargs["use_comfy_weights"], expected_comfy)
        self.assertEqual(holder.model_config.model_kwargs, model_kwargs)
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

    def test_official_source_does_not_redirect_any_component_by_default(self):
        for quantize_te in (False, True):
            with self.subTest(quantize_te=quantize_te):
                self.check_text_encoder_policy(
                    name_or_path=qwen.BASE_REPO, use_comfy_weights=None,
                    quantize_te=quantize_te, qtype_te="convrot8",
                )

    def test_legacy_comfy_source_remains_an_opt_in(self):
        self.check_text_encoder_policy(
            name_or_path=qwen.COMFY_REPO, use_comfy_weights=None,
            quantize_te=True, qtype_te="convrot8",
        )

    def test_official_repository_can_explicitly_opt_into_repacks(self):
        self.check_text_encoder_policy(
            name_or_path=qwen.BASE_REPO, use_comfy_weights=True,
            quantize_te=True, qtype_te="convrot8",
        )

    def test_engine_defaults_preserve_explicit_source_and_quantization_choices(self):
        cases = [
            ({}, qwen.BASE_REPO, False),
            ({"name_or_path": qwen.COMFY_REPO}, qwen.COMFY_REPO, True),
            ({"model_kwargs": {"rgba": True}}, qwen.BASE_REPO, False),
            ({"model_kwargs": {"use_comfy_weights": True}, "quantize": False,
              "quantize_te": False}, qwen.BASE_REPO, True),
        ]
        for values, expected_source, expected_comfy in cases:
            with self.subTest(values=values):
                config = Engine._build_model_config(SimpleNamespace(dtype="bf16"), {
                    "arch": "qwen_image_2", **values,
                })
                holder = qwen.QwenImage2Model("cpu", config, dtype="bf16")
                self.assertEqual(config.name_or_path, expected_source)
                self.assertEqual(config.model_kwargs, values.get("model_kwargs", {}))
                self.assertEqual(config.quantize, values.get("quantize", True))
                self.assertEqual(config.quantize_te, values.get("quantize_te", True))
                for role in ("transformer", "te", "vae"):
                    self.assertIs(holder.component_load_kwargs(role)["use_comfy_weights"], expected_comfy)


if __name__ == "__main__":
    unittest.main()
