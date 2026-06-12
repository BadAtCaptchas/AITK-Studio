import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock


class OpenRouterCaptionerTest(unittest.TestCase):
    def import_openrouter_captioner(self):
        base_module = types.ModuleType("extensions_built_in.captioner.BaseCaptioner")

        class FakeCaptionConfig:
            def __init__(self, **kwargs):
                self.model_name_or_path = kwargs.get("model_name_or_path", "")
                self.output_format = kwargs.get("output_format", "text")
                self.site_url = kwargs.get("site_url", "")
                self.app_title = kwargs.get("app_title", "AI Toolkit Captioner")
                self.max_new_tokens = kwargs.get("max_new_tokens", 128)
                self.temperature = kwargs.get("temperature", 0.2)

        class FakeBaseCaptioner:
            def __init__(self, process_id, job, config, **kwargs):
                pass

        base_module.BaseCaptioner = FakeBaseCaptioner
        base_module.CaptionConfig = FakeCaptionConfig
        base_module.IDEOGRAM_JSON_SCHEMA = {}
        sys.modules.pop("extensions_built_in.captioner.OpenRouterCaptioner", None)
        with mock.patch.dict(
            sys.modules,
            {"extensions_built_in.captioner.BaseCaptioner": base_module},
        ):
            from extensions_built_in.captioner.OpenRouterCaptioner import OpenRouterCaptioner

        return OpenRouterCaptioner

    def make_captioner(self, model_name: str, output_format: str = "text"):
        OpenRouterCaptioner = self.import_openrouter_captioner()
        captioner = object.__new__(OpenRouterCaptioner)
        captioner.caption_config = SimpleNamespace(
            model_name_or_path=model_name,
            output_format=output_format,
            site_url="",
            app_title="AI Toolkit Captioner",
            max_new_tokens=128,
            temperature=0.2,
            system_prompt="",
            base_url="https://openrouter.test/api/v1",
            max_res=512,
        )
        captioner.api_key = ""
        captioner.encrypted_reader = None
        captioner.is_ideogram_json_output = lambda: output_format in {"ideogram_json", "json"}
        return captioner

    def test_load_model_loads_with_valid_api_key(self):
        captioner = self.make_captioner("x-ai/grok-4.3")
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            captioner.print_and_status_update = mock.Mock()
            captioner.load_model()
            captioner.print_and_status_update.assert_called_once_with(
                "Using OpenRouter model x-ai/grok-4.3"
            )

    def test_load_model_allows_non_grok_models(self):
        captioner = self.make_captioner("gpt-image-1")
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            captioner.print_and_status_update = mock.Mock()
            captioner.load_model()
            captioner.print_and_status_update.assert_called_once_with(
                "Using OpenRouter model gpt-image-1"
            )

    def test_message_content_text_extracts_json_schema_value(self):
        OpenRouterCaptioner = self.import_openrouter_captioner()
        captioner = object.__new__(OpenRouterCaptioner)
        payload = {
            "choices": [
                {
                    "message": {
                        "content": {
                            "type": "json_schema",
                            "value": {
                                "high_level_description": "An image of a cat.",
                                "style_description": {
                                    "aesthetics": "clean",
                                    "lighting": "soft",
                                    "photo": "photograph",
                                    "medium": "digital",
                                    "color_palette": ["#FFFFFF"],
                                },
                                "compositional_deconstruction": {
                                    "background": "A plain white surface.",
                                    "elements": [],
                                },
                            },
                        }
                    }
                }
            ]
        }

        result = captioner._message_content_text(payload)
        self.assertEqual(
            result,
            '{"high_level_description": "An image of a cat.", "style_description": {"aesthetics": "clean", "lighting": "soft", "photo": "photograph", "medium": "digital", "color_palette": ["#FFFFFF"]}, "compositional_deconstruction": {"background": "A plain white surface.", "elements": []}}',
        )

    def test_build_payload_expands_token_budget_for_json_output(self):
        captioner = self.make_captioner("x-ai/grok-4.3", output_format="ideogram_json")
        captioner._image_to_data_url = mock.Mock(return_value=("data:image/jpeg;base64,abc", (64, 64)))
        captioner.build_caption_prompt = mock.Mock(return_value="Return JSON")

        payload, _image_size = captioner._build_payload("image.png")

        self.assertEqual(payload["max_tokens"], 2048)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["provider"], {"require_parameters": True})

    def test_build_payload_preserves_token_budget_for_text_output(self):
        captioner = self.make_captioner("x-ai/grok-4.3", output_format="text")
        captioner._image_to_data_url = mock.Mock(return_value=("data:image/jpeg;base64,abc", (64, 64)))
        captioner.build_caption_prompt = mock.Mock(return_value="Describe the image")

        payload, _image_size = captioner._build_payload("image.png")

        self.assertEqual(payload["max_tokens"], 128)
        self.assertNotIn("response_format", payload)
        self.assertNotIn("provider", payload)

    def test_get_caption_retries_json_parse_failures_with_larger_budget(self):
        captioner = self.make_captioner("x-ai/grok-4.3", output_format="ideogram_json")
        captioner._build_payload = mock.Mock(return_value=({"max_tokens": 2048}, (64, 64)))
        requested_budgets = []

        def fake_request(payload):
            requested_budgets.append(payload["max_tokens"])
            return {"choices": [{"message": {"content": "{ truncated"}}]}

        captioner._request_json = mock.Mock(side_effect=fake_request)
        captioner.normalize_caption_output = mock.Mock(
            side_effect=[ValueError("Captioner returned invalid JSON"), "normalized caption"]
        )

        with mock.patch("time.sleep"):
            result = captioner.get_caption_for_file("image.png")

        self.assertEqual(result, "normalized caption")
        self.assertEqual(requested_budgets, [2048, 4096])
        self.assertEqual(captioner._request_json.call_count, 2)
        self.assertEqual(captioner.normalize_caption_output.call_count, 2)


if __name__ == "__main__":
    unittest.main()
