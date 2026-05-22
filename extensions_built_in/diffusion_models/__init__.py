from .chroma import ChromaModel, ChromaRadianceModel
from .hidream import HidreamModel, HidreamE1Model
from .f_light import FLiteModel
from .omnigen2 import OmniGen2Model
from .flux_kontext import FluxKontextModel
from .wan22 import Wan225bModel, Wan2214bModel, Wan2214bI2VModel
from .qwen_image import QwenImageModel, QwenImageEditModel, QwenImageEditPlusModel
from .flux2 import Flux2Model, Flux2Klein4BModel, Flux2Klein9BModel, AsymFlux2Klein9BModel
from .z_image import ZImageModel
from .ltx2 import LTX2Model, LTX23Model
from .zeta_chroma import ZetaChromaModel
from .ernie_image import ErnieImageModel
from .nucleus_image import NucleusImageModel


def _unavailable_model_class(class_name: str, arch: str, import_error: ImportError):
    class UnavailableModel:
        pass

    def __init__(self, *args, **kwargs):
        raise ImportError(
            f"{class_name} requires optional dependencies that are not available: {import_error}"
        ) from import_error

    UnavailableModel.__name__ = class_name
    UnavailableModel.__qualname__ = class_name
    UnavailableModel.arch = arch
    UnavailableModel.__init__ = __init__
    return UnavailableModel


try:
    from .hidream.hidream_o1_model import HidreamO1Model
except ImportError as e:
    HidreamO1Model = _unavailable_model_class("HidreamO1Model", "hidream_o1", e)

try:
    from .glm_image import GlmImageModel
except ImportError as e:
    GlmImageModel = _unavailable_model_class("GlmImageModel", "glm_image", e)

AI_TOOLKIT_MODELS = [
    # put a list of models here
    ChromaModel,
    ChromaRadianceModel,
    HidreamModel,
    HidreamE1Model,
    FLiteModel,
    OmniGen2Model,
    FluxKontextModel,
    Wan225bModel,
    Wan2214bI2VModel,
    Wan2214bModel,
    QwenImageModel,
    QwenImageEditModel,
    QwenImageEditPlusModel,
    Flux2Model,
    ZImageModel,
    LTX2Model,
    LTX23Model,
    Flux2Klein4BModel,
    Flux2Klein9BModel,
    AsymFlux2Klein9BModel,
    ZetaChromaModel,
    ErnieImageModel,
    NucleusImageModel,
    HidreamO1Model,
    GlmImageModel,
]
