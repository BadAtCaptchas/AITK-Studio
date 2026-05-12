import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIDREAM_O1_MODEL_PATH = (
    PROJECT_ROOT
    / "extensions_built_in"
    / "diffusion_models"
    / "hidream"
    / "hidream_o1_model.py"
)


def load_hidream_o1_lora_converter():
    # Avoid importing the full model stack just to exercise these pure key mappers.
    source = HIDREAM_O1_MODEL_PATH.read_text()
    module = ast.parse(source, filename=str(HIDREAM_O1_MODEL_PATH))
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "HidreamO1Model"
    )
    methods = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"convert_lora_weights_before_save", "convert_lora_weights_before_load"}
    ]
    test_class = ast.ClassDef(
        name="HidreamO1LoraConverter",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    test_module = ast.Module(body=[test_class], type_ignores=[])
    ast.fix_missing_locations(test_module)

    namespace = {}
    exec(compile(test_module, str(HIDREAM_O1_MODEL_PATH), "exec"), namespace)
    return namespace["HidreamO1LoraConverter"]()


class HidreamO1LoraKeyConversionTest(unittest.TestCase):
    def setUp(self):
        self.model = load_hidream_o1_lora_converter()

    def test_save_preserves_model_language_model_path(self):
        state_dict = {
            "transformer.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": 1
        }

        converted = self.model.convert_lora_weights_before_save(state_dict)

        self.assertIn(
            "diffusion_model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
            converted,
        )
        self.assertNotIn(
            "diffusion_model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
            converted,
        )

    def test_save_preserves_model_final_layer_path(self):
        state_dict = {"transformer.model.final_layer2.linear.lora_B.weight": 1}

        converted = self.model.convert_lora_weights_before_save(state_dict)

        self.assertIn(
            "diffusion_model.model.final_layer2.linear.lora_B.weight",
            converted,
        )
        self.assertNotIn(
            "diffusion_model.final_layer2.linear.lora_B.weight",
            converted,
        )

    def test_load_accepts_model_prefixed_saved_keys(self):
        state_dict = {
            "diffusion_model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": 1
        }

        converted = self.model.convert_lora_weights_before_load(state_dict)

        self.assertIn(
            "transformer.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
            converted,
        )

    def test_load_accepts_legacy_stripped_saved_keys(self):
        state_dict = {
            "diffusion_model.language_model.layers.0.self_attn.q_proj.lora_A.weight": 1
        }

        converted = self.model.convert_lora_weights_before_load(state_dict)

        self.assertIn(
            "transformer.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
            converted,
        )


if __name__ == "__main__":
    unittest.main()
