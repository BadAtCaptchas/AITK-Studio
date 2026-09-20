"""Ordered sample references shared by training and inference."""


def validate_control_paths(value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > 10 or any(
        not isinstance(path, str) or not path.strip() for path in value
    ):
        raise ValueError("ctrl_imgs must be a list of at most 10 non-empty paths")
    return list(value)


def sample_control_paths(config):
    paths = getattr(config, "ctrl_imgs", None)
    if paths is not None:
        return validate_control_paths(paths)
    first = getattr(config, "ctrl_img", None)
    alias = getattr(config, "ctrl_img_1", None)
    return [path for path in (
        first, alias if alias != first else None,
        getattr(config, "ctrl_img_2", None), getattr(config, "ctrl_img_3", None),
    ) if path]
