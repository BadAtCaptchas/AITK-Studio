"""Lazy registry; importing this package never imports model implementations."""
from toolkit.model_registry import MODEL_DESCRIPTORS

AI_TOOLKIT_MODELS = [model for model in MODEL_DESCRIPTORS
                     if model.module.startswith(__name__ + '.')]
_BY_NAME = {model.class_name: model for model in AI_TOOLKIT_MODELS}


def __getattr__(name):
    descriptor = _BY_NAME.get(name)
    if descriptor is None:
        raise AttributeError(name)
    cls = descriptor.load()
    globals()[name] = cls
    return cls


def __dir__():
    return sorted(set(globals()) | set(_BY_NAME))
