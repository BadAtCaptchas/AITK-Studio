import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from collections import OrderedDict

from .BaseCaptioner import BaseCaptioner, CaptionConfig, is_refusal_caption


DEFAULT_OLLAMA_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36 AI-Toolkit-OllamaCaptioner"
)


class OllamaCaptionConfig(CaptionConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = kwargs.get("system_prompt", "")


class OllamaCaptioner(BaseCaptioner):
    caption_config_class = OllamaCaptionConfig
    caption_config: OllamaCaptionConfig

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(OllamaCaptioner, self).__init__(process_id, job, config, **kwargs)
        self.ollama_base_url = ""
        self.ollama_auth_token = ""
        self.ollama_user_agent = DEFAULT_OLLAMA_USER_AGENT
        self.ollama_model_ready = False

    def load_model(self):
        self.ollama_base_url = os.environ.get("AITK_OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
        self.ollama_auth_token = os.environ.get("AITK_OLLAMA_AUTH_TOKEN", "").strip()
        self.ollama_user_agent = (
            os.environ.get("AITK_OLLAMA_USER_AGENT", DEFAULT_OLLAMA_USER_AGENT).strip()
            or DEFAULT_OLLAMA_USER_AGENT
        )
        self.ollama_model_ready = False
        if not self.ollama_base_url:
            raise ValueError("Ollama base URL is missing")
        if not self.caption_config.model_name_or_path:
            raise ValueError("Ollama model is required")
        self.print_and_status_update(f"Using Ollama at {self.ollama_base_url}")
        self.ensure_model()
        self.ollama_model_ready = True

    def _request_json(self, route_path: str, payload: dict = None, timeout: int = 900) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self.ollama_user_agent:
            headers["User-Agent"] = self.ollama_user_agent
        if self.ollama_auth_token:
            headers["Authorization"] = f"Bearer {self.ollama_auth_token}"
        request = urllib.request.Request(
            f"{self.ollama_base_url}{route_path}",
            data=data,
            headers=headers,
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")[:500]
            try:
                parsed = json.loads(message)
                message = parsed.get("error") or message
            except Exception:
                pass
            if exc.code == 403 and "1010" in message:
                message = (
                    f"{message}. The remote proxy may be blocking this HTTP client; "
                    "try setting AITK_OLLAMA_USER_AGENT or using the secure remote Ollama worker."
                )
            raise RuntimeError(f"Ollama request to {route_path} failed with HTTP {exc.code}: {message}") from exc

    def _normalize_model_name(self, value: str) -> str:
        value = value.strip()
        return value if ":" in value else f"{value}:latest"

    def _is_gemma_model(self, value: str) -> bool:
        return self._normalize_model_name(value).lower().startswith("gemma")

    def _has_model(self, requested_model: str) -> bool:
        requested = self._normalize_model_name(requested_model)
        data = self._request_json("/api/tags", timeout=30)
        for model in data.get("models", []):
            for candidate in (model.get("model"), model.get("name")):
                if not candidate:
                    continue
                if candidate == requested_model or self._normalize_model_name(candidate) == requested:
                    return True
        return False

    def ensure_model(self):
        model = self.caption_config.model_name_or_path.strip()
        if not self._has_model(model):
            self.print_and_status_update(f"Downloading Ollama model {model}")
            self._request_json("/api/pull", {"model": model, "stream": False}, timeout=3600)
        self.print_and_status_update("Warming Ollama model")
        self._request_json(
            "/api/generate",
            {
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": "10m",
            },
            timeout=120,
        )
        self.print_and_status_update("Ollama model is ready")

    def _image_to_base64(self, file_path: str) -> tuple[str, tuple[int, int]]:
        image = self.load_pil_image(file_path, max_res=self.caption_config.max_res).convert("RGB")
        image_size = image.size
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii"), image_size

    def _caption_num_predict(self, attempt: int, extended_thinking_budget: bool = False) -> int:
        requested = self.caption_config.max_new_tokens or 0
        base_budget = max(2048, int(requested) * 4)
        max_budget = 8192 if extended_thinking_budget else 4096
        return min(max_budget, base_budget * (2 ** max(0, attempt - 1)))

    def _extract_caption(self, data: dict) -> str:
        response = data.get("response")
        if isinstance(response, str) and response.strip():
            return response.strip()
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        text = data.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return ""

    def _generate_caption_once(self, endpoint: str, body: dict) -> str:
        data = self._request_json(f"/api/{endpoint}", body, timeout=900)
        return self._extract_caption(data)

    def unload_model(self):
        model = self.caption_config.model_name_or_path.strip()
        if not model or not self.ollama_base_url:
            return False
        if not self.ollama_model_ready:
            return False
        try:
            self._request_json(
                "/api/generate",
                {
                    "model": model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0,
                },
                timeout=120,
            )
            return True
        except Exception as exc:
            print(f"Warning: Ollama model unload failed: {exc}")
            return False

    def run(self):
        try:
            super(OllamaCaptioner, self).run()
        finally:
            if self.unload_model():
                print("Ollama model unload requested")

    def get_caption_for_file(self, file_path: str) -> str:
        model = self.caption_config.model_name_or_path.strip()
        prompt = self.build_caption_prompt(file_path)
        image_base64, image_size = self._image_to_base64(file_path)
        gemma_model = self._is_gemma_model(model)
        if gemma_model:
            generate_body = {
                "model": model,
                "images": [image_base64],
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m",
            }
        else:
            generate_body = {
                "model": model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "keep_alive": "10m",
            }
        if self.caption_config.system_prompt.strip():
            generate_body["system"] = self.caption_config.system_prompt.strip()

        messages = []
        if self.caption_config.system_prompt.strip():
            messages.append({"role": "system", "content": self.caption_config.system_prompt.strip()})
        if gemma_model:
            messages.append({"role": "user", "images": [image_base64], "content": prompt})
            chat_body = {
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive": "10m",
            }
        else:
            messages.append({"role": "user", "content": prompt, "images": [image_base64]})
            chat_body = {
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive": "10m",
            }

        endpoint_order = ["chat", "generate"] if gemma_model else ["generate", "chat"]
        request_bodies = {
            "generate": generate_body,
            "chat": chat_body,
        }

        for attempt in range(1, 4):
            options = {"num_predict": self._caption_num_predict(attempt, extended_thinking_budget=gemma_model)}
            for endpoint in endpoint_order:
                caption = self._generate_caption_once(endpoint, {**request_bodies[endpoint], "options": options})
                if caption and not is_refusal_caption(caption):
                    return self.normalize_caption_output(file_path, caption, image_size=image_size)
            if attempt < 3:
                time.sleep(2)

        raise RuntimeError("Ollama returned an empty or refusal caption. Confirm the selected model supports image inputs.")
