import os

import huggingface_hub


def resolve_single_file_model_path(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path

    path_parts = model_path.split("/")
    if len(path_parts) < 3:
        raise ValueError(
            f"Single-file Z-Image model path {model_path} is not a valid local path or hub path."
        )

    repo_id = "/".join(path_parts[:2])
    filename = "/".join(path_parts[2:])
    try:
        return huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
    except Exception as e:
        raise ValueError(
            f"Failed to download single-file Z-Image model from {model_path}: {e}"
        )
