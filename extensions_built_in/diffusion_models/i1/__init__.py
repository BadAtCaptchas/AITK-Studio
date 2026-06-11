__all__ = ["I1Model"]


def __getattr__(name):
    if name == "I1Model":
        from .i1_model import I1Model

        return I1Model
    raise AttributeError(name)
