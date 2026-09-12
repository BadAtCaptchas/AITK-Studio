"""Teach Astroid about public functions exported by installed native libraries.

Only real, inspected exports are added. Unknown names still produce lint errors.
Compiled functions often lack Python signatures and infer as None through Torch's
docstring registration; their return types are deliberately left uninferred.
"""

from functools import partial
import importlib
import inspect

import astroid
from astroid.brain.helpers import register_module_extender


def _native_exports(module_name, names=None):
    """Describe installed native exports without executing any model code."""
    module = importlib.import_module(module_name)
    declarations = []
    for name in names or dir(module):
        if name.startswith('_') or not name.isidentifier():
            continue
        value = getattr(module, name)
        if inspect.isbuiltin(value):
            declarations.append(
                f'def {name}(*args, **kwargs):\n    return _compiled_result\n'
            )
        elif module_name == 'cv2' and isinstance(value, (int, float, str)):
            declarations.append(f'{name} = {value!r}\n')
    return astroid.parse('\n'.join(declarations))


def register(_linter):
    """Register precise native-library extensions for the project lint run."""
    for module_name, names in (
        ('torch.nn.functional', None),
        ('torch.linalg', None),
        ('torch.fft', None),
        ('torch.special', None),
        ('torch', ('istft',)),
        ('torch.overrides', ('has_torch_function_unary',)),
        ('cv2', None),
    ):
        register_module_extender(
            astroid.MANAGER, module_name, partial(_native_exports, module_name, names)
        )
