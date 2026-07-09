from toolkit.paths import MODELS_PATH
import requests
import os
import json
import hashlib
import tempfile
import tqdm
from urllib.parse import parse_qs, urlparse
from toolkit.network_policy import assert_url_allowed


REQUEST_TIMEOUT = (10, 120)
DEFAULT_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024 * 1024


def _max_download_bytes():
    raw_value = os.environ.get('AITK_CIVITAI_MAX_DOWNLOAD_GB', '').strip()
    if not raw_value:
        return DEFAULT_MAX_DOWNLOAD_BYTES
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError('AITK_CIVITAI_MAX_DOWNLOAD_GB must be a positive number') from exc
    if value <= 0:
        raise ValueError('AITK_CIVITAI_MAX_DOWNLOAD_GB must be a positive number')
    return int(value * 1024 * 1024 * 1024)


def _safe_filename(value):
    filename = os.path.basename(str(value or '').replace('\\', '/')).strip()
    if not filename or filename in {'.', '..'}:
        raise ValueError('Civitai returned an invalid model filename')
    return filename[:240]


def _expected_sha256(file_info):
    hashes = file_info.get('hashes') or {}
    for key, value in hashes.items():
        if str(key).lower().replace('-', '') == 'sha256' and isinstance(value, str):
            normalized = value.strip().lower()
            if len(normalized) == 64 and all(char in '0123456789abcdef' for char in normalized):
                return normalized
    return None


class ModelCache:
    def __init__(self):
        self.raw_cache = {}
        self.cache_path = os.path.join(MODELS_PATH, '.ai_toolkit_cache.json')
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    all_cache = json.load(f)
                    if 'models' in all_cache:
                        self.raw_cache = all_cache['models']
                    else:
                        self.raw_cache = all_cache
            except (OSError, ValueError):
                self.raw_cache = {}

    def get_model_path(self, model_id: int, model_version_id: int = None):
        if str(model_id) not in self.raw_cache:
            return None
        if model_version_id is None:
            # get latest version
            model_version_id = max([int(x) for x in self.raw_cache[str(model_id)].keys()])
            if model_version_id is None:
                return None
            model_path = self.raw_cache[str(model_id)][str(model_version_id)]['model_path']
            # check if model path exists
            if not os.path.exists(model_path):
                # remove version from cache
                del self.raw_cache[str(model_id)][str(model_version_id)]
                self.save()
                return None
            return model_path
        else:
            if str(model_version_id) not in self.raw_cache[str(model_id)]:
                return None
            model_path = self.raw_cache[str(model_id)][str(model_version_id)]['model_path']
            # check if model path exists
            if not os.path.exists(model_path):
                # remove version from cache
                del self.raw_cache[str(model_id)][str(model_version_id)]
                self.save()
                return None
            return model_path

    def update_cache(self, model_id: int, model_version_id: int, model_path: str):
        if str(model_id) not in self.raw_cache:
            self.raw_cache[str(model_id)] = {}
        if str(model_version_id) not in self.raw_cache[str(model_id)]:
            self.raw_cache[str(model_id)][str(model_version_id)] = {}
        self.raw_cache[str(model_id)][str(model_version_id)] = {
            'model_path': model_path
        }
        self.save()

    def save(self):
        if not os.path.exists(os.path.dirname(self.cache_path)):
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        all_cache = {'models': {}}
        if os.path.exists(self.cache_path):
            # load it first
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    all_cache = json.load(f)
            except (OSError, ValueError):
                all_cache = {'models': {}}

        all_cache['models'] = self.raw_cache

        fd, tmp_path = tempfile.mkstemp(prefix='.aitk-cache-', suffix='.tmp', dir=os.path.dirname(self.cache_path))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(all_cache, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.cache_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise


def get_model_download_info(model_id: int, model_version_id: int = None):
    # curl https://civitai.com/api/v1/models?limit=3&types=TextualInversion \
    # -H "Content-Type: application/json" \
    # -X GET
    print(
        f"Getting model info for model id: {model_id}{f' and version id: {model_version_id}' if model_version_id is not None else ''}")
    endpoint = f"https://civitai.com/api/v1/models/{model_id}"

    # get the json
    assert_url_allowed(endpoint, "Civitai model metadata request")
    response = requests.get(endpoint, timeout=REQUEST_TIMEOUT)
    try:
        response.raise_for_status()
        model_data = response.json()
    finally:
        response.close()

    model_version = None

    # go through versions and get the top one if one is not set
    for version in model_data['modelVersions']:
        if model_version_id is not None:
            if str(version['id']) == str(model_version_id):
                model_version = version
                break
        else:
            # get first version
            model_version = version
            break

    if model_version is None:
        raise ValueError(
            f"Could not find a model version for model id: {model_id}{f' and version id: {model_version_id}' if model_version_id is not None else ''}")

    model_file = None
    # Prefer safetensors over arbitrary primary artifacts.
    # "metadata": {
    #   "fp": "fp16",
    #   "size": "pruned",
    #   "format": "SafeTensor"
    # },
    # todo check pickle scans and skip if not good
    # try to get fp16 safetensor
    for file in model_version.get('files', []):
        metadata = file.get('metadata') or {}
        if metadata.get('fp') == 'fp16' and metadata.get('format') == 'SafeTensor':
            model_file = file
            break

    if model_file is None:
        # try to get any safetensor
        for file in model_version.get('files', []):
            if (file.get('metadata') or {}).get('format') == 'SafeTensor':
                model_file = file
                break

    if model_file is None:
        # then try the API's primary artifact
        for file in model_version.get('files', []):
            if file.get('primary'):
                model_file = file
                break

    if model_file is None:
        # try to get any fp16
        for file in model_version.get('files', []):
            if (file.get('metadata') or {}).get('fp') == 'fp16':
                model_file = file
                break

    if model_file is None:
        # try to get any
        for file in model_version.get('files', []):
            model_file = file
            break

    if model_file is None:
        raise ValueError(f"Could not find a model file to download for model id: {model_id}")

    return model_file, model_version['id']


def get_model_path_from_url(url: str):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    model_version_id = (query_params.get('modelVersionId') or [None])[0]
    model_id = parsed_url.path.rstrip('/').split('/')[-1]
    if model_id.isdigit():
        model_id = int(model_id)
    else:
        raise ValueError(f"Invalid model id: {model_id}")

    model_cache = ModelCache()
    model_path = model_cache.get_model_path(model_id, model_version_id)
    if model_path is not None:
        return model_path
    else:
        # download model
        file_info, model_version_id = get_model_download_info(model_id, model_version_id)

        download_url = file_info['downloadUrl']
        filename = _safe_filename(file_info.get('name'))
        destination_root = os.path.realpath(
            os.path.join(MODELS_PATH, 'civitai', str(model_id), str(model_version_id))
        )
        models_root = os.path.realpath(MODELS_PATH)
        if os.path.commonpath([models_root, destination_root]) != models_root:
            raise ValueError('Invalid Civitai model destination')
        model_path = os.path.join(destination_root, filename)
        max_download_bytes = _max_download_bytes()
        advertised_size = int(float(file_info.get('sizeKB') or 0) * 1024)
        if advertised_size > max_download_bytes:
            raise ValueError('Civitai model exceeds the configured download size limit')

        # download model
        print(f"Did not find model locally, downloading from model from: {download_url}")

        # use tqdm to show status of downlod
        assert_url_allowed(download_url, "Civitai model download")
        response = requests.get(download_url, stream=True, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        if total_size_in_bytes > max_download_bytes:
            response.close()
            raise ValueError('Civitai model exceeds the configured download size limit')
        block_size = 1024 * 1024
        progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        os.makedirs(destination_root, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix='.download-', suffix='.tmp', dir=destination_root)
        os.close(fd)

        try:
            digest = hashlib.sha256()
            downloaded_bytes = 0
            with open(tmp_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    if not data:
                        continue
                    downloaded_bytes += len(data)
                    if downloaded_bytes > max_download_bytes:
                        raise ValueError('Civitai model exceeds the configured download size limit')
                    progress_bar.update(len(data))
                    f.write(data)
                    digest.update(data)
                f.flush()
                os.fsync(f.fileno())
            if total_size_in_bytes and downloaded_bytes != total_size_in_bytes:
                raise ValueError('Civitai model download ended before the advertised size')
            expected_hash = _expected_sha256(file_info)
            if expected_hash and digest.hexdigest().lower() != expected_hash:
                raise ValueError('Civitai model SHA-256 verification failed')
            os.replace(tmp_path, model_path)
            model_cache.update_cache(model_id, model_version_id, model_path)
            return model_path
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        finally:
            progress_bar.close()
            response.close()


# if is main
if __name__ == '__main__':
    model_path = get_model_path_from_url("https://civitai.com/models/25694?modelVersionId=127742")
    print(model_path)
