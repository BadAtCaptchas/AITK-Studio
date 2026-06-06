import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class ModelPluginDiscoverySecurityTest(unittest.TestCase):
    def import_get_model_with_stubs(self, toolkit_root: Path):
        class BaseModel:
            pass

        class StableDiffusion(BaseModel):
            arch = "stable_diffusion"

        class Wan21(BaseModel):
            arch = "wan21"

        class Wan21I2V(BaseModel):
            arch = "wan21_i2v"

        class CogView4(BaseModel):
            arch = "cogview4"

        class ModelConfig:
            def __init__(self, arch):
                self.arch = arch

        stub_modules = {
            "toolkit.models.base_model": types.ModuleType("toolkit.models.base_model"),
            "toolkit.stable_diffusion_model": types.ModuleType("toolkit.stable_diffusion_model"),
            "toolkit.config_modules": types.ModuleType("toolkit.config_modules"),
            "toolkit.paths": types.ModuleType("toolkit.paths"),
            "toolkit.models.wan21": types.ModuleType("toolkit.models.wan21"),
            "toolkit.models.cogview4": types.ModuleType("toolkit.models.cogview4"),
        }
        stub_modules["toolkit.models.base_model"].BaseModel = BaseModel
        stub_modules["toolkit.stable_diffusion_model"].StableDiffusion = StableDiffusion
        stub_modules["toolkit.config_modules"].ModelConfig = ModelConfig
        stub_modules["toolkit.paths"].TOOLKIT_ROOT = str(toolkit_root)
        stub_modules["toolkit.models.wan21"].Wan21 = Wan21
        stub_modules["toolkit.models.wan21"].Wan21I2V = Wan21I2V
        stub_modules["toolkit.models.cogview4"].CogView4 = CogView4

        for module_name in ["toolkit.util.get_model", *stub_modules]:
            sys.modules.pop(module_name, None)

        with mock.patch.dict(sys.modules, stub_modules):
            module = importlib.import_module("toolkit.util.get_model")

        return module, ModelConfig, StableDiffusion

    def test_get_model_class_does_not_import_user_extensions_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            toolkit_root = Path(tmpdir)
            package_dir = toolkit_root / "extensions" / "pwned_model"
            package_dir.mkdir(parents=True)
            marker = toolkit_root / "imported_marker"
            package_dir.joinpath("__init__.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('imported')\n"
                "class PwnedModel:\n"
                "    arch = 'pwned'\n"
                "AI_TOOLKIT_MODELS = [PwnedModel]\n",
                encoding="utf-8",
            )

            with mock.patch.dict(sys.modules), mock.patch.dict("os.environ", {}, clear=True):
                sys.path.insert(0, str(toolkit_root))
                try:
                    get_model, ModelConfig, StableDiffusion = self.import_get_model_with_stubs(toolkit_root)
                    self.assertIs(get_model.get_model_class(ModelConfig("pwned")), StableDiffusion)
                    self.assertFalse(marker.exists())
                finally:
                    sys.path.remove(str(toolkit_root))
                    sys.modules.pop("extensions.pwned_model", None)

    def test_get_model_class_imports_matching_static_built_in_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            class Flux2Klein4BModel:
                arch = "flux2_klein_4b"

            diffusion_models = types.ModuleType("extensions_built_in.diffusion_models")
            diffusion_models.Flux2Klein4BModel = Flux2Klein4BModel

            with mock.patch.dict(sys.modules, {"extensions_built_in.diffusion_models": diffusion_models}):
                get_model, ModelConfig, _ = self.import_get_model_with_stubs(Path(tmpdir))
                self.assertIs(get_model.get_model_class(ModelConfig("flux2_klein_4b")), Flux2Klein4BModel)


if __name__ == "__main__":
    unittest.main()
