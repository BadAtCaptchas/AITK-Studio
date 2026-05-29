import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections import OrderedDict

from .BaseCaptioner import BaseCaptioner, CaptionConfig
from .secure_remote_crypto import decrypt_secure_caption_json, encrypt_secure_caption_json


class SecureRemoteOllamaCaptionConfig(CaptionConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.remote_worker_id = kwargs.get("remote_worker_id", None)
        self.system_prompt = kwargs.get("system_prompt", "")


class SecureRemoteOllamaCaptioner(BaseCaptioner):
    caption_config_class = SecureRemoteOllamaCaptionConfig
    caption_config: SecureRemoteOllamaCaptionConfig

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(SecureRemoteOllamaCaptioner, self).__init__(process_id, job, config, **kwargs)
        self.remote_base_url = ""
        self.remote_token = ""

    def load_model(self):
        self.remote_base_url = os.environ.get("AITK_SECURE_CAPTION_REMOTE_BASE_URL", "").strip().rstrip("/")
        self.remote_token = os.environ.get("AITK_SECURE_CAPTION_REMOTE_TOKEN", "").strip()
        if not self.remote_base_url:
            raise ValueError("Secure remote caption worker URL is missing")
        if not self.remote_token:
            raise ValueError("Secure remote caption worker token is missing")
        self.print_and_status_update("Using secure remote Ollama caption worker")
        self.ensure_remote_model()

    def _image_to_base64(self, file_path: str) -> str:
        image = self.load_pil_image(file_path, max_res=self.caption_config.max_res).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _item_id_for_file(self, file_path: str) -> str:
        digest = hashlib.sha256(f"{self.job_id or self.job.name}:{file_path}:{uuid.uuid4()}".encode("utf-8")).hexdigest()
        return digest[:32]

    def _post_secure_caption(self, envelope: dict) -> dict:
        body = json.dumps(envelope).encode("utf-8")
        request = urllib.request.Request(
            f"{self.remote_base_url}/api/secure-caption/ollama",
            data=body,
            headers={
                "Authorization": f"Bearer {self.remote_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")[:500]
            try:
                parsed = json.loads(message)
                message = parsed.get("error") or message
            except Exception:
                pass
            raise RuntimeError(f"Remote Ollama caption failed: {message}") from exc

    def _post_secure_unload(self, envelope: dict) -> dict:
        body = json.dumps(envelope).encode("utf-8")
        request = urllib.request.Request(
            f"{self.remote_base_url}/api/secure-caption/ollama/unload",
            data=body,
            headers={
                "Authorization": f"Bearer {self.remote_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")[:500]
            try:
                parsed = json.loads(message)
                message = parsed.get("error") or message
            except Exception:
                pass
            raise RuntimeError(f"Remote Ollama unload failed: {message}") from exc

    def _post_secure_pull(self, envelope: dict) -> dict:
        body = json.dumps(envelope).encode("utf-8")
        request = urllib.request.Request(
            f"{self.remote_base_url}/api/secure-caption/ollama/pull",
            data=body,
            headers={
                "Authorization": f"Bearer {self.remote_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")[:500]
            try:
                parsed = json.loads(message)
                message = parsed.get("error") or message
            except Exception:
                pass
            raise RuntimeError(f"Remote Ollama model prepare failed: {message}") from exc

    def ensure_remote_model(self):
        model = self.caption_config.model_name_or_path
        if not model:
            raise ValueError("Ollama model is required")
        job_id = self.job_id or self.job.name
        last_status_update = 0.0
        while True:
            item_id = f"pull-{uuid.uuid4().hex}"
            request_envelope = encrypt_secure_caption_json(
                self.remote_token,
                "request",
                job_id,
                item_id,
                {"model": model},
            )
            response_envelope = self._post_secure_pull(request_envelope)
            response_payload = decrypt_secure_caption_json(self.remote_token, "response", response_envelope)
            status = str(response_payload.get("status", "")).strip().lower()
            if status == "ready":
                self.print_and_status_update("Remote Ollama model is ready")
                return
            if status == "error":
                message = str(response_payload.get("error") or "remote Ollama model pull failed")
                raise RuntimeError(f"Remote Ollama model prepare failed: {message}")
            if status != "pulling":
                raise RuntimeError("Remote Ollama model prepare returned an unknown status")
            now = time.time()
            if now - last_status_update > 30:
                phase = str(response_payload.get("phase") or "").strip().lower()
                if phase == "warming":
                    self.print_and_status_update("Remote Ollama model is warming up")
                elif phase == "pulling":
                    self.print_and_status_update("Remote Ollama model is downloading")
                else:
                    self.print_and_status_update("Remote Ollama model is preparing")
                last_status_update = now
            time.sleep(5)

    def unload_remote_model(self):
        if not self.remote_base_url or not self.remote_token:
            return
        model = self.caption_config.model_name_or_path
        if not model:
            return
        item_id = f"unload-{uuid.uuid4().hex}"
        job_id = self.job_id or self.job.name
        payload = {"model": model}
        request_envelope = encrypt_secure_caption_json(self.remote_token, "request", job_id, item_id, payload)
        response_envelope = self._post_secure_unload(request_envelope)
        decrypt_secure_caption_json(self.remote_token, "response", response_envelope)

    def run(self):
        try:
            super(SecureRemoteOllamaCaptioner, self).run()
        finally:
            try:
                self.unload_remote_model()
                print("Remote Ollama model unload requested")
            except Exception as exc:
                print(f"Warning: remote Ollama model unload failed: {exc}")

    def get_caption_for_file(self, file_path: str) -> str:
        item_id = self._item_id_for_file(file_path)
        job_id = self.job_id or self.job.name
        payload = {
            "model": self.caption_config.model_name_or_path,
            "prompt": self.caption_config.caption_prompt,
            "systemPrompt": self.caption_config.system_prompt,
            "imageBase64": self._image_to_base64(file_path),
            "maxNewTokens": self.caption_config.max_new_tokens,
        }
        request_envelope = encrypt_secure_caption_json(self.remote_token, "request", job_id, item_id, payload)
        response_envelope = self._post_secure_caption(request_envelope)
        response_payload = decrypt_secure_caption_json(self.remote_token, "response", response_envelope)
        caption = str(response_payload.get("caption", "")).strip()
        return caption or None
