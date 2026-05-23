from .z_image import ZImageModel


def __getattr__(name):
    if name == "ZImageL2PModel":
        from .z_image_l2p_model import ZImageL2PModel

        return ZImageL2PModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
