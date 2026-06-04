def is_comfy_backend(config) -> bool:
    return getattr(config, 'backend', 'native') == 'comfy'
