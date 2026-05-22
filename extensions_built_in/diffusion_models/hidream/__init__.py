from .hidream_model import HidreamModel
from .hidream_e1_model import HidreamE1Model


def __getattr__(name):
    if name == "HidreamO1Model":
        from .hidream_o1_model import HidreamO1Model

        return HidreamO1Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
