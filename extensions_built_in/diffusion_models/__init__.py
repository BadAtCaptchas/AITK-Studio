from importlib import import_module


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


def _optional_models(module_name: str, model_specs):
    try:
        module = import_module(module_name, __name__)
    except ImportError as e:
        return tuple(
            _unavailable_model_class(class_name, arch, e)
            for class_name, arch in model_specs
        )

    return tuple(getattr(module, class_name) for class_name, _ in model_specs)


ChromaModel, ChromaRadianceModel = _optional_models(
    ".chroma",
    (
        ("ChromaModel", "chroma"),
        ("ChromaRadianceModel", "chroma_radiance"),
    ),
)
HidreamModel, HidreamE1Model = _optional_models(
    ".hidream",
    (
        ("HidreamModel", "hidream"),
        ("HidreamE1Model", "hidream_e1"),
    ),
)
FLiteModel, = _optional_models(".f_light", (("FLiteModel", "f-lite"),))
OmniGen2Model, = _optional_models(".omnigen2", (("OmniGen2Model", "omnigen2"),))
FluxKontextModel, = _optional_models(
    ".flux_kontext", (("FluxKontextModel", "flux_kontext"),)
)
Wan225bModel, Wan2214bModel, Wan2214bI2VModel = _optional_models(
    ".wan22",
    (
        ("Wan225bModel", "wan22_5b"),
        ("Wan2214bModel", "wan22_14b"),
        ("Wan2214bI2VModel", "wan22_14b_i2v"),
    ),
)
QwenImageModel, QwenImageEditModel, QwenImageEditPlusModel = _optional_models(
    ".qwen_image",
    (
        ("QwenImageModel", "qwen_image"),
        ("QwenImageEditModel", "qwen_image_edit"),
        ("QwenImageEditPlusModel", "qwen_image_edit_plus"),
    ),
)
Flux2Model, Flux2Klein4BModel, Flux2Klein9BModel, AsymFlux2Klein9BModel = _optional_models(
    ".flux2",
    (
        ("Flux2Model", "flux2"),
        ("Flux2Klein4BModel", "flux2_klein_4b"),
        ("Flux2Klein9BModel", "flux2_klein_9b"),
        ("AsymFlux2Klein9BModel", "asymflux2_klein_9b"),
    ),
)
ZImageModel, = _optional_models(".z_image.z_image", (("ZImageModel", "zimage"),))
ZImageL2PModel, = _optional_models(
    ".z_image.z_image_l2p_model", (("ZImageL2PModel", "zimage_l2p"),)
)
LTX2Model, LTX23Model = _optional_models(
    ".ltx2",
    (
        ("LTX2Model", "ltx2"),
        ("LTX23Model", "ltx2.3"),
    ),
)
ZetaChromaModel, = _optional_models(
    ".zeta_chroma", (("ZetaChromaModel", "zeta_chroma"),)
)
ErnieImageModel, = _optional_models(
    ".ernie_image", (("ErnieImageModel", "ernie_image"),)
)
NucleusImageModel, = _optional_models(
    ".nucleus_image", (("NucleusImageModel", "nucleus_image"),)
)
HidreamO1Model, = _optional_models(
    ".hidream.hidream_o1_model", (("HidreamO1Model", "hidream_o1"),)
)
GlmImageModel, = _optional_models(".glm_image", (("GlmImageModel", "glm_image"),))


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
    ZImageL2PModel,
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
