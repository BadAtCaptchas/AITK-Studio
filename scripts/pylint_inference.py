"""Teach Astroid about public functions exported by installed native libraries.

Only real, inspected exports are added. Unknown names still produce lint errors.
Compiled functions often lack Python signatures and infer as None through Torch's
docstring registration; their return types are deliberately left uninferred.
"""

from functools import partial
import importlib
import inspect
import os
import sys

import astroid
from astroid.brain.helpers import register_module_extender


def _normalize_transformers_import_path():
    """Astroid 4 expects strings; Transformers processors install Path entries."""
    module = sys.modules.get('transformers')
    spec = getattr(module, '__spec__', None)
    locations = getattr(spec, 'submodule_search_locations', None)
    if locations and any(isinstance(location, os.PathLike) for location in locations):
        spec.submodule_search_locations = [os.fspath(location) for location in locations]


def _native_exports(module_name, names=None):
    """Describe installed native exports without executing any model code."""
    module = importlib.import_module(module_name)
    declarations = []
    for name in names or dir(module):
        if (name.startswith('_') and not (module_name == 'torch' and name in ('_int_mm', '_scaled_mm'))) or not name.isidentifier():
            continue
        value = getattr(module, name)
        if inspect.isbuiltin(value):
            declarations.append(
                f'def {name}(*args, **kwargs):\n    return _compiled_result\n'
            )
        elif module_name == 'torch' and inspect.ismodule(value):
            declarations.append(f'import {value.__name__} as {name}\n')
        elif module_name == 'cv2' and inspect.isclass(value) and value.__module__.startswith('cv2'):
            # OpenCV's extension types are public classes, not builtin
            # functions. Preserve their instance shape so misspelled methods
            # still fail instead of treating the constructor result as Any.
            members = ['    def __init__(self, *args, **kwargs):\n        pass\n']
            for member_name in dir(value):
                if member_name.startswith('_') or not member_name.isidentifier():
                    continue
                member = getattr(value, member_name)
                if inspect.ismethoddescriptor(member) or inspect.isbuiltin(member):
                    members.append(
                        f'    def {member_name}(self, *args, **kwargs):\n        return _compiled_result\n'
                    )
                elif isinstance(member, (int, float, str)):
                    members.append(f'    {member_name} = {member!r}\n')
                elif inspect.isdatadescriptor(member):
                    members.append(
                        f'    @property\n    def {member_name}(self):\n        return _compiled_result\n'
                    )
            declarations.append(f'class {name}:\n' + ''.join(members))
        elif module_name == 'cv2' and isinstance(value, (int, float, str)):
            declarations.append(f'{name} = {value!r}\n')
    return astroid.parse('\n'.join(declarations))


def _replace_method(node, name, declaration):
    method = astroid.parse(declaration).body[0]
    method.parent = node
    node.locals[name] = [method]
    node.instance_attrs.pop(name, None)


def _callable_declaration(name, value):
    """Keep the installed callable's real argument names and keyword policy."""
    signature = inspect.signature(value, follow_wrapped=False)
    parameters = []
    for parameter in signature.parameters.values():
        default = parameter.default
        if default is not inspect.Parameter.empty and not isinstance(default, (type(None), bool, int, float, str)):
            default = None
        parameters.append(parameter.replace(annotation=inspect.Parameter.empty, default=default))
    signature = signature.replace(parameters=parameters, return_annotation=inspect.Signature.empty)
    return f'def {name}{signature}:\n    return _compiled_result\n'


def _torch_tensor_methods(node):
    """Docstring registration preserves native Tensor methods at runtime."""
    tensor = importlib.import_module('torch').Tensor
    for name in dir(tensor):
        if name.startswith('_') or not name.isidentifier():
            continue
        if inspect.ismethoddescriptor(getattr(tensor, name)):
            _replace_method(node, name, f'def {name}(self, *args, **kwargs):\n    return _compiled_result\n')


def _bounded_torch_getattr(node):
    # Torch's __getattr__ only accepts the installed deprecated/lazy names;
    # treating it as an arbitrary dynamic lookup hides misspelled exports.
    torch = importlib.import_module('torch')
    for name in torch._lazy_modules:
        if not name.startswith('_'):
            stub = astroid.parse(f'import torch.{name} as {name}')
            imported = stub.body[0]
            imported.parent = node
            node.locals[name] = [imported]
    node.locals.pop('__getattr__', None)
    return node


_CONFIG_CLASSES = {
    'transformers.models.clip.configuration_clip.CLIPVisionConfig',
    'transformers.models.qwen2.configuration_qwen2.Qwen2Config',
    'transformers.models.qwen3.configuration_qwen3.Qwen3Config',
    'transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5TextConfig',
    'transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig',
    'transformers.models.gemma3.configuration_gemma3.Gemma3Config',
}


def _config_constructor(node):
    """Transformers wraps generated config initializers to accept legacy kwargs."""
    module_name, class_name = node.qname().rsplit('.', 1)
    config = getattr(importlib.import_module(module_name), class_name)
    _normalize_transformers_import_path()
    _replace_method(node, '__init__', _callable_declaration('__init__', config.__init__))


def _checkpoint_callable(node):
    """ModelMixin installs this callable when checkpointing is enabled."""
    from diffusers.models.modeling_utils import ModelMixin
    _normalize_transformers_import_path()
    source = inspect.getsource(ModelMixin.enable_gradient_checkpointing)
    if 'gradient_checkpointing_func' in source and callable(ModelMixin._set_gradient_checkpointing):
        _replace_method(node, '_gradient_checkpointing_func',
                        'def _gradient_checkpointing_func(self, module, *args):\n    return _compiled_result\n')


_GRADIO_EVENTS = {
    'gradio.components.file.File': ('upload', 'delete', 'clear'),
    'gradio.components.button.Button': ('click',),
}


def _gradio_events(node):
    module_name, class_name = node.qname().rsplit('.', 1)
    component = getattr(importlib.import_module(module_name), class_name)
    for name in _GRADIO_EVENTS[node.qname()]:
        event = getattr(component, name)
        if callable(event):
            _replace_method(node, name, _callable_declaration(name, event))


def _lazy_exports(module_name, names):
    """Resolve public aliases to their installed definition, without constructing them."""
    module = importlib.import_module(module_name)
    declarations = []
    for name in names:
        value = getattr(module, name)
        declarations.append(f'from {value.__module__} import {value.__name__} as {name}')
    _normalize_transformers_import_path()
    return astroid.parse('\n'.join(declarations))


def _model_registry_exports():
    """Use the same declarations as the lazy runtime registry, without loading models."""
    from toolkit.model_registry import MODEL_DESCRIPTORS
    return astroid.parse('\n'.join(
        f'from {model.module} import {model.class_name}' for model in MODEL_DESCRIPTORS
        if model.module.startswith('extensions_built_in.diffusion_models.')
    ))


def _torch_context_manager():
    """Describe the installed native context manager used by tensor DTOs."""
    value = importlib.import_module('torch')._C.DisableTorchFunctionSubclass
    if not all(callable(getattr(value, name, None)) for name in ('__enter__', '__exit__')):
        return astroid.parse('')
    return astroid.parse(
        'class DisableTorchFunctionSubclass:\n'
        '    def __enter__(self):\n        return self\n'
        '    def __exit__(self, exc_type, exc_value, traceback):\n        return None\n'
    )


def _triton_rint(module_name):
    """The installed extern decorator returns a Triton value, not None."""
    try:
        value = importlib.import_module(module_name).rint
    except (ImportError, OSError):
        # Triton is optional, including platforms without its native runtime.
        return astroid.parse('')
    return astroid.parse(_callable_declaration('rint', inspect.unwrap(value)))


def register(_linter):
    """Register precise native-library extensions for the project lint run."""
    _normalize_transformers_import_path()
    astroid.MANAGER.register_transform(astroid.nodes.Module, _bounded_torch_getattr, lambda node: node.name == 'torch')
    for module_name, names in (
        ('torch.nn.functional', None),
        ('torch.linalg', None),
        ('torch.fft', None),
        ('torch.special', None),
        ('torch', None),
        ('torch.overrides', ('has_torch_function_unary',)),
        ('cv2', None),
    ):
        register_module_extender(
            astroid.MANAGER, module_name, partial(_native_exports, module_name, names)
        )
    for module_name, names in (
        ('transformers', ('T5TokenizerFast', 'Qwen2TokenizerFast')),
        ('diffusers.pipelines.stable_diffusion_xl', ('StableDiffusionXLPipelineOutput',)),
    ):
        register_module_extender(astroid.MANAGER, module_name, partial(_lazy_exports, module_name, names))
    register_module_extender(astroid.MANAGER, 'extensions_built_in.diffusion_models', _model_registry_exports)
    register_module_extender(astroid.MANAGER, 'torch._C', _torch_context_manager)
    for module_name in ('triton.language.extra.libdevice', 'triton.language.extra.cuda.libdevice'):
        register_module_extender(astroid.MANAGER, module_name, partial(_triton_rint, module_name))
    astroid.MANAGER.register_transform(astroid.nodes.ClassDef, _torch_tensor_methods,
                                       lambda node: node.qname() in ('torch.Tensor', 'torch._tensor.Tensor'))
    astroid.MANAGER.register_transform(astroid.nodes.ClassDef, _config_constructor,
                                       lambda node: node.qname() in _CONFIG_CLASSES)
    astroid.MANAGER.register_transform(astroid.nodes.ClassDef, _checkpoint_callable,
                                       lambda node: node.qname() == 'diffusers.models.modeling_utils.ModelMixin')
    astroid.MANAGER.register_transform(astroid.nodes.ClassDef, _gradio_events,
                                       lambda node: node.qname() in _GRADIO_EVENTS)
