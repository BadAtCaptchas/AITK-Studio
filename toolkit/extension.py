import os
import importlib
import pkgutil
from typing import List

from toolkit.paths import TOOLKIT_ROOT


class Extension(object):
    """Base class for extensions.

    Extensions are registered with the ExtensionManager, which is
    responsible for calling the extension's load() and unload()
    methods at the appropriate times.

    """

    name: str = None
    uid: str = None

    @classmethod
    def get_process(cls):
        # extend in subclass
        pass


def _unavailable_process_class(extension: Extension, import_error: ImportError):
    uid = getattr(extension, "uid", "unknown")
    name = getattr(extension, "name", uid)

    class UnavailableExtensionProcess:
        pass

    def __init__(self, *args, **kwargs):
        raise ImportError(
            f"{name} ({uid}) requires optional dependencies that are not available: {import_error}"
        ) from import_error

    UnavailableExtensionProcess.__name__ = f"Unavailable{uid}Process"
    UnavailableExtensionProcess.__qualname__ = UnavailableExtensionProcess.__name__
    UnavailableExtensionProcess.__init__ = __init__
    return UnavailableExtensionProcess


def get_all_extensions() -> List[Extension]:
    extension_folders = ['extensions', 'extensions_built_in']

    # This will hold the classes from all extension modules
    all_extension_classes: List[Extension] = []

    # Iterate over all directories (i.e., packages) in the "extensions" directory
    for sub_dir in extension_folders:
        extensions_dir = os.path.join(TOOLKIT_ROOT, sub_dir)
        for (_, name, _) in pkgutil.iter_modules([extensions_dir]):
            try:
                module = importlib.import_module(f"{sub_dir}.{name}")
            except ImportError:
                continue

            extensions = getattr(module, "AI_TOOLKIT_EXTENSIONS", None)
            if isinstance(extensions, list):
                all_extension_classes.extend(extensions)

    return all_extension_classes


def get_all_extensions_process_dict():
    all_extensions = get_all_extensions()
    process_dict = {}
    for extension in all_extensions:
        try:
            process_dict[extension.uid] = extension.get_process()
        except ImportError as e:
            process_dict[extension.uid] = _unavailable_process_class(extension, e)
    return process_dict
