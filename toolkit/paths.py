import os

TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_ROOT = os.path.join(TOOLKIT_ROOT, 'config')
KEYMAPS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "keymaps")
ORIG_CONFIGS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "orig_configs")
DIFFUSERS_CONFIGS_ROOT = os.path.join(TOOLKIT_ROOT, "toolkit", "diffusers_configs")
COMFY_PATH = os.getenv("COMFY_PATH", None)
COMFY_MODELS_PATH = None
if COMFY_PATH:
    COMFY_MODELS_PATH = os.path.join(COMFY_PATH, "models")

# Only a nonblank environment value overrides the repository default. The UI
# resolves its authenticated setting into this environment variable before a
# job starts, so CLI and Studio launches share the same final precedence.
_models_path_env = os.environ.get("MODELS_PATH", "").strip()
if _models_path_env:
    MODELS_PATH = _models_path_env
else:
    MODELS_PATH = os.path.join(TOOLKIT_ROOT, "models")


def get_path(path):
    # we allow absolute paths, but if it is not absolute, we assume it is relative to the toolkit root
    if not os.path.isabs(path):
        path = os.path.join(TOOLKIT_ROOT, path)
    return path
