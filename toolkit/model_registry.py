"""Weight-free model discovery shared with Studio's versioned capability contract."""
from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path

CONTRACT = json.loads((Path(__file__).resolve().parents[1] / 'ui/src/domain/modelCapabilities.json').read_text(encoding='utf-8'))
if CONTRACT['version'] != 1:
    raise RuntimeError('Unsupported model capability contract')


class MissingModelDependency(ImportError):
    """A declared optional extra is absent; installing its profile may resolve it."""


class BrokenModelIntegration(ImportError):
    """A model's implementation or installed dependency API failed to import."""


@dataclass(frozen=True)
class ModelDescriptor:
    """Describe a model's public class without importing its training dependencies."""
    arch: str
    module: str
    class_name: str
    optional_dependencies: tuple[str, ...] = ()

    def load(self):
        """Import the public class and distinguish missing extras from broken code."""
        try:
            cls = getattr(import_module(self.module), self.class_name)
            if not isinstance(cls, type):
                raise TypeError(f'{self.class_name} is not a model class')
            return cls
        except ModuleNotFoundError as error:
            if error.name in self.optional_dependencies:
                raise MissingModelDependency(
                    f'{self.arch} requires the declared optional dependency {error.name}. '
                    'Install the matching runtime profile and run the environment doctor.'
                ) from error
            raise BrokenModelIntegration(f'{self.arch}: missing undeclared module {error.name}') from error
        except (ImportError, AttributeError, TypeError) as error:
            raise BrokenModelIntegration(f'{self.arch}: broken integration in {self.module}: {error}') from error


MODEL_DESCRIPTORS = tuple(ModelDescriptor(row['arch'], row['module'], row['className'],
    tuple(row['optionalDependencies'])) for row in CONTRACT['models'])
BY_ARCH = {model.arch: model for model in MODEL_DESCRIPTORS}


def resolve_model(arch: str):
    """Resolve a registered architecture or compatibility alias to its model class."""
    arch = arch.split(':', 1)[0]
    arch = {'sd15': 'sd1', 'flex1': 'flux'}.get(arch, arch)
    descriptor = BY_ARCH.get(arch)
    if descriptor is None:
        raise ValueError(f'Unsupported model architecture: {arch}')
    return descriptor.load()


def validate_model_imports(arches=None):
    """Strict CI/preflight entry point. Never constructs a model or downloads weights."""
    failures = {}
    for arch in arches or BY_ARCH:
        try:
            resolve_model(arch)
        except Exception as error:
            failures[arch] = f'{type(error).__name__}: {error}'
    return failures
