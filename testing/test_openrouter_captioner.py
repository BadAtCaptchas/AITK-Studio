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

    def make_captioner(self, model_name: str):
        OpenRouterCaptioner = self.import_openrouter_captioner()
        captioner = object.__new__(OpenRouterCaptioner)
        captioner.caption_config = SimpleNamespace(
            model_name_or_path=model_name,
            output_format="text",
            site_url="",
            app_title="AI Toolkit Captioner",
            max_new_tokens=128,
            temperature=0.2,
        )
        captioner.api_key = ""
        captioner.encrypted_reader = None
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


if __name__ == "__main__":
    unittest.main()
