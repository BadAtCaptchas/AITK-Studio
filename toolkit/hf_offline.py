import os


HF_OFFLINE_ENV_VARS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)


def hf_env_flag(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().upper() in {"1", "ON", "TRUE", "YES"}


def is_hf_offline_mode() -> bool:
    return any(hf_env_flag(name) for name in HF_OFFLINE_ENV_VARS)
