from toolkit.comfy.config import is_comfy_backend
from toolkit.comfy.errors import ComfyConfigError, ComfyError, ComfyWorkflowError

__all__ = [
    'ComfyConfigError',
    'ComfyError',
    'ComfyWorkflowError',
    'generate_images_with_comfy',
    'is_comfy_backend',
]


def __getattr__(name):
    if name == 'generate_images_with_comfy':
        from toolkit.comfy.generator import generate_images_with_comfy
        return generate_images_with_comfy
    raise AttributeError(name)
