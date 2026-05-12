import unittest
import types
from unittest import mock

torch_module = types.ModuleType("torch")
torch_nn_module = types.ModuleType("torch.nn")
torch_nn_functional_module = types.ModuleType("torch.nn.functional")


class TorchModule:
    pass


torch_module.Tensor = object
torch_module.nn = torch_nn_module
torch_nn_module.Module = TorchModule
torch_nn_module.ModuleList = list
torch_nn_module.Linear = mock.Mock
torch_nn_module.functional = torch_nn_functional_module

diffusers_module = types.ModuleType("diffusers")
diffusers_module.Transformer2DModel = object
transformer_flux_module = types.ModuleType("diffusers.models.transformers.transformer_flux")
transformer_flux_module.FluxTransformerBlock = object
transformers_module = types.ModuleType("transformers")
for class_name in [
    "AutoModel",
    "AutoTokenizer",
    "Qwen2Model",
    "LlamaModel",
    "Qwen2Tokenizer",
    "LlamaTokenizer",
]:
    setattr(transformers_module, class_name, object)
dequantize_module = types.ModuleType("toolkit.dequantize")
dequantize_module.patch_dequantization_on_save = mock.Mock()
prompt_utils_module = types.ModuleType("toolkit.prompt_utils")
prompt_utils_module.PromptEmbeds = object

with mock.patch.dict(
    "sys.modules",
    {
        "torch": torch_module,
        "torch.nn": torch_nn_module,
        "torch.nn.functional": torch_nn_functional_module,
        "diffusers": diffusers_module,
        "diffusers.models": types.ModuleType("diffusers.models"),
        "diffusers.models.transformers": types.ModuleType("diffusers.models.transformers"),
        "diffusers.models.transformers.transformer_flux": transformer_flux_module,
        "transformers": transformers_module,
        "toolkit.dequantize": dequantize_module,
        "toolkit.prompt_utils": prompt_utils_module,
        "toolkit.train_tools": types.ModuleType("toolkit.train_tools"),
    },
):
    from toolkit.models.llm_adapter import _validate_num_cloned_blocks


class LLMAdapterNumClonedBlocksValidationTest(unittest.TestCase):
    def test_accepts_integer_within_available_blocks(self):
        self.assertEqual(_validate_num_cloned_blocks(0, 2), 0)
        self.assertEqual(_validate_num_cloned_blocks(2, 2), 2)

    def test_rejects_non_integer_values(self):
        for value in (True, False, 1.0, 1.9, "1", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "num_cloned_blocks must be an integer"):
                    _validate_num_cloned_blocks(value, 2)

    def test_rejects_integer_outside_available_blocks(self):
        for value in (-1, 3):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "num_cloned_blocks must be between 0 and 2"):
                    _validate_num_cloned_blocks(value, 2)


if __name__ == "__main__":
    unittest.main()
