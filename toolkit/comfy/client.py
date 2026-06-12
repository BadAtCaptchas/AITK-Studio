import json
import time
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image

from toolkit.comfy.errors import ComfyError, ComfyWorkflowError


class ComfyClient:
    def __init__(self, server_url: str, timeout: float = 900, poll_interval: float = 0.5):
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.poll_interval = poll_interval

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Any]] = None,
        expect_json: bool = True,
    ):
        url = self.server_url + path
        if query:
            url += '?' + urllib.parse.urlencode(query)

        body = None
        headers = {}
        if data is not None:
            body = json.dumps(data).encode('utf-8')
            headers['Content-Type'] = 'application/json'

        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=min(self.timeout, 60)) as response:
                payload = response.read()
                if not expect_json:
                    return payload
                if not payload:
                    return None
                return json.loads(payload.decode('utf-8'))
        except urllib.error.HTTPError as e:
            detail = e.read().decode('utf-8', errors='replace')
            raise ComfyError(f"ComfyUI request failed: {method} {path} returned {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise ComfyError(f"Could not reach ComfyUI at {self.server_url}: {e.reason}") from e

    def system_stats(self):
        return self._request('GET', '/system_stats')

    def object_info(self):
        return self._request('GET', '/object_info')

    def queue_prompt(self, workflow: Dict[str, Any], client_id: str):
        response = self._request('POST', '/prompt', {'prompt': workflow, 'client_id': client_id})
        prompt_id = response.get('prompt_id') if isinstance(response, dict) else None
        if not prompt_id:
            raise ComfyError(f"ComfyUI did not return a prompt_id: {response}")
        return prompt_id

    def history(self, prompt_id: str):
        return self._request('GET', f'/history/{urllib.parse.quote(prompt_id)}')

    def view(self, filename: str, subfolder: str = '', image_type: str = 'output') -> bytes:
        return self._request(
            'GET',
            '/view',
            query={'filename': filename, 'subfolder': subfolder, 'type': image_type},
            expect_json=False,
        )

    def interrupt(self):
        return self._request('POST', '/interrupt', {})

    def free(self, unload_models: bool = True, free_memory: bool = True):
        return self._request('POST', '/free', {'unload_models': unload_models, 'free_memory': free_memory})

    def wait_for_history(self, prompt_id: str, cancel_check=None):
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if cancel_check is not None and cancel_check():
                self.interrupt()
                raise ComfyError("ComfyUI generation was cancelled")
            history = self.history(prompt_id)
            entry = self._normalize_history_entry(history, prompt_id)
            if entry is not None:
                status = entry.get('status', {}) if isinstance(entry, dict) else {}
                if status.get('status_str') == 'error':
                    raise ComfyError(f"ComfyUI prompt failed: {status}")
                return entry
            time.sleep(self.poll_interval)
        raise ComfyError(f"Timed out waiting for ComfyUI prompt {prompt_id}")

    def prompt_and_wait(self, workflow: Dict[str, Any], client_id: str, cancel_check=None):
        prompt_id = self.queue_prompt(workflow, client_id)
        return self.wait_for_history(prompt_id, cancel_check=cancel_check)

    def save_history_images(self, history_entry: Dict[str, Any], gen_config) -> List[str]:
        image_refs = list(iter_history_image_refs(history_entry))
        if not image_refs:
            raise ComfyWorkflowError("ComfyUI workflow completed without image outputs")

        saved_paths = []
        max_count = max(0, len(image_refs) - 1)
        for count, image_ref in enumerate(image_refs):
            image_bytes = self.view(
                filename=image_ref['filename'],
                subfolder=image_ref.get('subfolder', ''),
                image_type=image_ref.get('type', 'output'),
            )
            with Image.open(BytesIO(image_bytes)) as image:
                image = image.convert('RGB')
                gen_config.save_image(image, count=count, max_count=max_count)
                gen_config.log_image(image, count=count, max_count=max_count)
                saved_paths.append(gen_config.get_image_path(count=count, max_count=max_count))
        return saved_paths

    @staticmethod
    def _normalize_history_entry(history: Any, prompt_id: str):
        if not history:
            return None
        if isinstance(history, dict) and prompt_id in history:
            return history[prompt_id]
        if isinstance(history, dict) and ('outputs' in history or 'status' in history):
            return history
        return None


def iter_history_image_refs(history_entry: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    outputs = history_entry.get('outputs', {}) if isinstance(history_entry, dict) else {}
    for output in outputs.values():
        for image in output.get('images', []) or []:
            if isinstance(image, dict) and image.get('filename'):
                yield image
