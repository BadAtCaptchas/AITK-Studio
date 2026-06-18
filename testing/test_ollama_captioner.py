import io
import os
import sys
import types
import unittest
import urllib.error
from types import SimpleNamespace
from unittest import mock


class OllamaCaptionerTest(unittest.TestCase):
    def import_ollama_captioner(self):
        base_module = types.ModuleType("extensions_built_in.captioner.BaseCaptioner")

        class FakeCaptionConfig:
            def __init__(self, **kwargs):
                self.model_name_or_path = kwargs.get("model_name_or_path", "llava")

        class FakeBaseCaptioner:
            pass

        def fake_is_refusal_caption(value):
            return "please provide the image or video you would like me to caption" in str(value).lower()

        base_module.BaseCaptioner = FakeBaseCaptioner
        base_module.CaptionConfig = FakeCaptionConfig
        base_module.is_refusal_caption = fake_is_refusal_caption
        sys.modules.pop("extensions_built_in.captioner.OllamaCaptioner", None)
        with mock.patch.dict(
            sys.modules,
            {"extensions_built_in.captioner.BaseCaptioner": base_module},
        ):
            from extensions_built_in.captioner.OllamaCaptioner import (
                DEFAULT_OLLAMA_USER_AGENT,
                OllamaCaptioner,
            )

        return DEFAULT_OLLAMA_USER_AGENT, OllamaCaptioner

    def make_captioner(self):
        _, OllamaCaptioner = self.import_ollama_captioner()
        captioner = object.__new__(OllamaCaptioner)
        captioner.ollama_base_url = "https://ollama.test"
        captioner.ollama_auth_token = ""
        captioner.ollama_user_agent = "AITK-Test-Agent"
        captioner.ollama_model_ready = False
        captioner.caption_config = SimpleNamespace(
            model_name_or_path="llava",
            system_prompt="",
            max_res=None,
            max_new_tokens=32,
        )
        return captioner

    def prepare_get_caption_for_file(self, captioner, responses):
        calls = []
        captioner.build_caption_prompt = mock.Mock(return_value="caption this image")
        captioner._image_to_base64 = mock.Mock(return_value=("aW1n", (64, 64)))
        captioner.normalize_caption_output = mock.Mock(side_effect=lambda _path, caption, image_size=None: caption)

        def fake_generate(endpoint, body):
            calls.append((endpoint, body))
            return responses.get(endpoint, "")

        captioner._generate_caption_once = mock.Mock(side_effect=fake_generate)
        return calls

    def test_request_json_sends_user_agent_and_auth_token(self):
        captioner = self.make_captioner()
        captioner.ollama_auth_token = "remote-token"
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"models":[]}'

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            data = captioner._request_json("/api/tags", timeout=30)

        request = captured["request"]
        self.assertEqual(data, {"models": []})
        self.assertEqual(captured["timeout"], 30)
        self.assertEqual(request.full_url, "https://ollama.test/api/tags")
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.get_header("User-agent"), "AITK-Test-Agent")
        self.assertEqual(request.get_header("Authorization"), "Bearer remote-token")
        self.assertEqual(request.get_header("Accept"), "application/json")

    def test_load_model_defaults_user_agent_and_marks_ready_after_warmup(self):
        default_user_agent, OllamaCaptioner = self.import_ollama_captioner()
        captioner = object.__new__(OllamaCaptioner)
        captioner.caption_config = SimpleNamespace(model_name_or_path="llava")
        captioner.print_and_status_update = mock.Mock()
        captioner.ensure_model = mock.Mock()

        with mock.patch.dict(
            os.environ,
            {
                "AITK_OLLAMA_BASE_URL": "https://ollama.test/",
                "AITK_OLLAMA_AUTH_TOKEN": "",
                "AITK_OLLAMA_USER_AGENT": "",
            },
            clear=False,
        ):
            captioner.load_model()

        self.assertEqual(captioner.ollama_base_url, "https://ollama.test")
        self.assertEqual(captioner.ollama_user_agent, default_user_agent)
        self.assertTrue(captioner.ollama_model_ready)
        captioner.ensure_model.assert_called_once_with()

    def test_load_model_leaves_model_not_ready_when_prepare_fails(self):
        _, OllamaCaptioner = self.import_ollama_captioner()
        captioner = object.__new__(OllamaCaptioner)
        captioner.caption_config = SimpleNamespace(model_name_or_path="llava")
        captioner.print_and_status_update = mock.Mock()
        captioner.ensure_model = mock.Mock(side_effect=RuntimeError("blocked"))

        with mock.patch.dict(
            os.environ,
            {
                "AITK_OLLAMA_BASE_URL": "https://ollama.test/",
                "AITK_OLLAMA_USER_AGENT": "Custom-UA",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                captioner.load_model()

        self.assertFalse(captioner.ollama_model_ready)
        self.assertEqual(captioner.ollama_user_agent, "Custom-UA")

    def test_unload_model_skips_when_load_never_completed(self):
        captioner = self.make_captioner()
        captioner._request_json = mock.Mock()

        self.assertFalse(captioner.unload_model())
        captioner._request_json.assert_not_called()

    def test_http_403_1010_error_includes_proxy_hint(self):
        captioner = self.make_captioner()
        http_error = urllib.error.HTTPError(
            "https://ollama.test/api/tags",
            403,
            "Forbidden",
            {},
            io.BytesIO(b'{"error":"error code: 1010"}'),
        )

        with mock.patch("urllib.request.urlopen", side_effect=http_error):
            with self.assertRaisesRegex(RuntimeError, "AITK_OLLAMA_USER_AGENT"):
                captioner._request_json("/api/tags", timeout=30)

    def test_get_caption_for_file_uses_generate_before_chat_for_non_gemma(self):
        captioner = self.make_captioner()
        captioner.caption_config.model_name_or_path = "llava"
        calls = self.prepare_get_caption_for_file(captioner, {"chat": "chat caption"})

        caption = captioner.get_caption_for_file("image.png")

        self.assertEqual(caption, "chat caption")
        self.assertEqual([endpoint for endpoint, _body in calls], ["generate", "chat"])
        self.assertEqual(calls[1][1]["messages"][-1]["images"], ["aW1n"])

    def test_get_caption_for_file_uses_chat_before_generate_for_gemma(self):
        captioner = self.make_captioner()
        captioner.caption_config.model_name_or_path = "gemma4:31b"
        calls = self.prepare_get_caption_for_file(captioner, {"generate": "generate caption"})

        caption = captioner.get_caption_for_file("image.png")

        self.assertEqual(caption, "generate caption")
        self.assertEqual([endpoint for endpoint, _body in calls], ["chat", "generate"])
        self.assertEqual(calls[0][1]["messages"][-1]["images"], ["aW1n"])

    def test_get_caption_for_file_skips_gemma_chat_refusal(self):
        captioner = self.make_captioner()
        captioner.caption_config.model_name_or_path = "gemma4:31b"
        calls = self.prepare_get_caption_for_file(
            captioner,
            {
                "chat": "Please provide the image or video you would like me to caption.",
                "generate": "generate caption",
            },
        )

        caption = captioner.get_caption_for_file("image.png")

        self.assertEqual(caption, "generate caption")
        self.assertEqual([endpoint for endpoint, _body in calls], ["chat", "generate"])


if __name__ == "__main__":
    unittest.main()
