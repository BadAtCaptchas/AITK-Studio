import os
from typing import Iterable, List, Optional, Tuple
from toolkit.models.base_model import BaseModel
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.config_modules import ModelConfig
from toolkit.paths import TOOLKIT_ROOT
import importlib
import pkgutil

from toolkit.models.wan21 import Wan21, Wan21I2V
from toolkit.models.cogview4 import CogView4

BUILT_IN_MODELS = [
    Wan21,
    Wan21I2V,
    CogView4,
]

# Static registry for built-in model plugin packages. Keeping this list explicit avoids
# importing every package in extensions/ during normal model resolution.
BUILT_IN_MODEL_REGISTRY: Tuple[Tuple[str, str, str], ...] = (
    ("flex2", "extensions_built_in.flex2", "Flex2"),
    ("ace_step_15", "extensions_built_in.audio_models", "AceStep15Model"),
    ("ace_step_15_xl", "extensions_built_in.audio_models", "AceStep15XLModel"),
    ("chroma", "extensions_built_in.diffusion_models", "ChromaModel"),
    ("chroma_radiance", "extensions_built_in.diffusion_models", "ChromaRadianceModel"),
    ("hidream", "extensions_built_in.diffusion_models", "HidreamModel"),
    ("hidream_e1", "extensions_built_in.diffusion_models", "HidreamE1Model"),
    ("f-lite", "extensions_built_in.diffusion_models", "FLiteModel"),
    ("omnigen2", "extensions_built_in.diffusion_models", "OmniGen2Model"),
    ("flux_kontext", "extensions_built_in.diffusion_models", "FluxKontextModel"),
    ("wan22_5b", "extensions_built_in.diffusion_models", "Wan225bModel"),
    ("wan22_14b", "extensions_built_in.diffusion_models", "Wan2214bModel"),
    ("wan22_14b_i2v", "extensions_built_in.diffusion_models", "Wan2214bI2VModel"),
    ("qwen_image", "extensions_built_in.diffusion_models", "QwenImageModel"),
    ("qwen_image_edit", "extensions_built_in.diffusion_models", "QwenImageEditModel"),
    ("qwen_image_edit_plus", "extensions_built_in.diffusion_models", "QwenImageEditPlusModel"),
    ("flux2", "extensions_built_in.diffusion_models", "Flux2Model"),
    ("zimage", "extensions_built_in.diffusion_models", "ZImageModel"),
    ("zimage_l2p", "extensions_built_in.diffusion_models", "ZImageL2PModel"),
    ("ltx2", "extensions_built_in.diffusion_models", "LTX2Model"),
    ("ltx2.3", "extensions_built_in.diffusion_models", "LTX23Model"),
    ("flux2_klein_4b", "extensions_built_in.diffusion_models", "Flux2Klein4BModel"),
    ("flux2_klein_9b", "extensions_built_in.diffusion_models", "Flux2Klein9BModel"),
    ("asymflux2_klein_9b", "extensions_built_in.diffusion_models", "AsymFlux2Klein9BModel"),
    ("zeta_chroma", "extensions_built_in.diffusion_models", "ZetaChromaModel"),
    ("ernie_image", "extensions_built_in.diffusion_models", "ErnieImageModel"),
    ("ideogram4", "extensions_built_in.diffusion_models", "Ideogram4Model"),
    ("nucleus_image", "extensions_built_in.diffusion_models", "NucleusImageModel"),
    ("hidream_o1", "extensions_built_in.diffusion_models", "HidreamO1Model"),
    ("glm_image", "extensions_built_in.diffusion_models", "GlmImageModel"),
)


def _external_model_plugins_enabled() -> bool:
    return os.environ.get("AI_TOOLKIT_ENABLE_EXTENSION_MODEL_PLUGINS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _load_model_class(module_name: str, class_name: str) -> Optional[BaseModel]:
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Failed to import the {module_name} module. Error: {str(e)}")
        return None
    return getattr(module, class_name, None)


def _iter_external_extension_models() -> Iterable[BaseModel]:
    if not _external_model_plugins_enabled():
        return

    extensions_dir = os.path.join(TOOLKIT_ROOT, 'extensions')
    for (_, name, _) in pkgutil.iter_modules([extensions_dir]):
        try:
            module = importlib.import_module(f"extensions.{name}")
        except ImportError as e:
            print(f"Failed to import the {name} module. Error: {str(e)}")
            continue

        models = getattr(module, "AI_TOOLKIT_MODELS", None)
        if isinstance(models, list):
            yield from models


def get_all_models() -> List[BaseModel]:
    # Copy the built-in list so callers cannot mutate the module-level registry.
    all_model_classes: List[BaseModel] = list(BUILT_IN_MODELS)

    for _, module_name, class_name in BUILT_IN_MODEL_REGISTRY:
        model_class = _load_model_class(module_name, class_name)
        if model_class is not None:
            all_model_classes.append(model_class)

    all_model_classes.extend(_iter_external_extension_models())
    return all_model_classes


def get_model_class(config: ModelConfig):
    for ModelClass in BUILT_IN_MODELS:
        if ModelClass.arch == config.arch:
            return ModelClass

    for arch, module_name, class_name in BUILT_IN_MODEL_REGISTRY:
        if arch != config.arch:
            continue
        model_class = _load_model_class(module_name, class_name)
        if model_class is not None:
            return model_class

    for ModelClass in _iter_external_extension_models():
        if ModelClass.arch == config.arch:
            return ModelClass

    # default to the legacy model
    return StableDiffusion
