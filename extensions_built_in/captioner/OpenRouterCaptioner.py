import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Optional

from .BaseCaptioner import BaseCaptioner, CaptionConfig, IDEOGRAM_JSON_SCHEMA
from toolkit.network_policy import assert_url_allowed


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV_NAMES = ("OPENROUTER_API_KEY", "AITK_OPENROUTER_API_KEY")
OPENROUTER_JSON_MIN_TOKENS = 2048
OPENROUTER_JSON_MAX_RETRY_TOKENS = 8192


class OpenRouterCaptionConfig(CaptionConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = kwargs.get("system_prompt", "")
        # Never trust job configuration for secret source selection or API destination.
        self.api_key_env = OPENROUTER_API_KEY_ENV_NAMES[0]
        self.base_url = OPENROUTER_BASE_URL
        self.site_url = kwargs.get("site_url", "")
        self.app_title = kwargs.get("app_title", "AI Toolkit Captioner")
        self.temperature: Optional[float] = kwargs.get("temperature", 0.2)


class OpenRouterCaptioner(BaseCaptioner):
    caption_config_class = OpenRouterCaptionConfig
    caption_config: OpenRouterCaptionConfig

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(OpenRouterCaptioner, self).__init__(process_id, job, config, **kwargs)
        self.api_key = ""

    def load_model(self):
        if self.encrypted_reader is not None:
            raise ValueError(
                "OpenRouter captioning is not supported for encrypted datasets. "
                "Caption an unencrypted copy, then encrypt the finished dataset if needed."
            )
        for env_name in OPENROUTER_API_KEY_ENV_NAMES:
            value = os.environ.get(env_name, "").strip()
            if value:
                self.api_key = value
                break
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is missing. Set OPENROUTER_API_KEY or save it in the UI settings."
            )
        if not self.caption_config.model_name_or_path:
            raise ValueError("OpenRouter model is required")
        self.print_and_status_update(
            f"Using OpenRouter model {self.caption_config.model_name_or_path}"
        )

    def _image_to_data_url(self, file_path: str) -> tuple[str, tuple[int, int]]:
        image = self.load_pil_image(
            file_path, max_res=self.caption_config.max_res
        ).convert("RGB")
        image_size = image.size
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95, optimize=True)
        return (
            f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}",
            image_size,
        )

    def _headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.caption_config.site_url:
            headers["HTTP-Referer"] = self.caption_config.site_url
        if self.caption_config.app_title:
            headers["X-Title"] = self.caption_config.app_title
        return headers

    def _request_json(self, payload: dict, timeout: int = 900) -> dict:
        body = json.dumps(payload).encode("utf-8")
        url = f"{self.caption_config.base_url}/chat/completions"
        assert_url_allowed(url, "OpenRouter caption request")
        request = urllib.request.Request(
            url,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")[:1000]
            try:
                parsed = json.loads(message)
                message = (
                    parsed.get("error", {}).get("message")
                    if isinstance(parsed.get("error"), dict)
                    else parsed.get("error")
                ) or message
            except Exception:
                pass
            raise RuntimeError(f"OpenRouter request failed: {message}") from exc

    def _extract_content_text(self, content) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, dict):
            if isinstance(content.get("text"), str) and content["text"].strip():
                return content["text"].strip()
            if isinstance(content.get("value"), str) and content["value"].strip():
                return content["value"].strip()
            if content.get("value") is not None:
                return json.dumps(content["value"], ensure_ascii=False)
            if content.get("content") is not None:
                return self._extract_content_text(content["content"])
            return ""
        if isinstance(content, list):
            parts = []
            for item in content:
                part = self._extract_content_text(item)
                if part:
                    parts.append(part)
            return "\n".join(parts).strip()
        return ""

    def _message_content_text(self, data: dict) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            return ""
        return self._extract_content_text(message.get("content"))

    def _caption_max_tokens(self, attempt: int = 1) -> int:
        requested = int(self.caption_config.max_new_tokens or 0)
        if not self.is_ideogram_json_output():
            return max(1, requested)

        budget = max(OPENROUTER_JSON_MIN_TOKENS, requested * 4)
        if attempt > 1:
            budget *= 2 ** (attempt - 1)
        return min(budget, OPENROUTER_JSON_MAX_RETRY_TOKENS)

    def _build_payload(self, file_path: str) -> tuple[dict, tuple[int, int]]:
        image_data_url, image_size = self._image_to_data_url(file_path)
        messages = []
        if self.caption_config.system_prompt.strip():
            messages.append(
                {"role": "system", "content": self.caption_config.system_prompt.strip()}
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.build_caption_prompt(file_path)},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            }
        )
        payload = {
            "model": self.caption_config.model_name_or_path.strip(),
            "messages": messages,
            "stream": False,
            "max_tokens": self._caption_max_tokens(),
        }
        if self.caption_config.temperature is not None:
            payload["temperature"] = float(self.caption_config.temperature)
        if self.is_ideogram_json_output():
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ideogram_caption",
                    "strict": True,
                    "schema": IDEOGRAM_JSON_SCHEMA,
                },
            }
            payload["provider"] = {"require_parameters": True}
        return payload, image_size

    def get_caption_for_file(self, file_path: str) -> str:
        payload, image_size = self._build_payload(file_path)
        last_error = None
        for attempt in range(1, 4):
            payload["max_tokens"] = self._caption_max_tokens(attempt)
            data = self._request_json(payload)
            caption = self._message_content_text(data)
            if caption:
                try:
                    return self.normalize_caption_output(file_path, caption, image_size=image_size)
                except ValueError as exc:
                    last_error = exc
                    if not self.is_ideogram_json_output() or attempt >= 3:
                        raise
            if attempt < 3:
                time.sleep(2)
        if last_error is not None:
            raise RuntimeError(
                f"OpenRouter returned invalid JSON after retries for model "
                f"'{self.caption_config.model_name_or_path}': {last_error}"
            ) from last_error
        raise RuntimeError(
            f"OpenRouter returned an empty caption for model '{self.caption_config.model_name_or_path}'. "
            "Confirm the selected model supports image inputs."
        )
