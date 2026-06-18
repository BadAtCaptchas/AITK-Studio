import os
import re

from huggingface_hub import hf_hub_download
from toolkit.paths import MODELS_PATH


_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_LOCAL_PATH_PREFIXES = ("/", "./", "../", "~/")


def _print_download_status(message: str):
    try:
        from toolkit.print import print_acc

        print_acc(message)
    except Exception:
        print(message)


def resolve_hub_single_file_path(model_path: str, description: str = "model") -> str:
    if not isinstance(model_path, str) or os.path.exists(model_path):
        return model_path
    if not model_path.endswith(".safetensors"):
        return model_path
    if "://" in model_path or _WINDOWS_DRIVE_PATH.match(model_path):
        return model_path
    if model_path.startswith(_LOCAL_PATH_PREFIXES):
        return model_path

    path_parts = model_path.split("/")
    if len(path_parts) < 3:
        return model_path
    if any(part in ("", ".", "..", "~") for part in path_parts[:2]):
        return model_path

    repo_id = "/".join(path_parts[:2])
    filename = "/".join(path_parts[2:])
    target_dir = os.path.join(MODELS_PATH, "diffusion_models")
    target_path = os.path.join(target_dir, *filename.split("/"))
    if os.path.isfile(target_path):
        return target_path

    try:
        _print_download_status(f"Downloading {description} from Hugging Face Hub: {model_path}")
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
            token=os.getenv("HF_TOKEN") or None,
        )
    except Exception as e:
        raise ValueError(
            f"Failed to download {description} from Hugging Face Hub ({model_path}): {e}"
        )
