import ast
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import load_file, save_file

from toolkit.prompt_utils import PromptEmbeds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANIMA_PATH = (
    PROJECT_ROOT
    / "extensions_built_in"
    / "diffusion_models"
    / "anima"
    / "anima.py"
)
REGISTRY_PATH = (
    PROJECT_ROOT / "extensions_built_in" / "diffusion_models" / "__init__.py"
)
CONFIG_PATH = PROJECT_ROOT / "toolkit" / "config_modules.py"
OFFLOAD_PATH = PROJECT_ROOT / "toolkit" / "memory_management" / "offload.py"
PROMPT_UTILS_PATH = PROJECT_ROOT / "toolkit" / "prompt_utils.py"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements_base.txt"
README_PATH = PROJECT_ROOT / "README.md"
UI_OPTIONS_PATH = (
    PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "options.ts"
)
UI_PROFILES_PATH = (
    PROJECT_ROOT
    / "ui"
    / "src"
    / "app"
    / "jobs"
    / "new"
    / "autoTrainingProfiles.ts"
)
UI_MEMORY_PATH = PROJECT_ROOT / "ui" / "src" / "utils" / "memoryProfiles.ts"


def _class_node(name: str) -> ast.ClassDef:
    parsed = ast.parse(ANIMA_PATH.read_text(encoding="utf-8"), str(ANIMA_PATH))
    return next(
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def load_prompt_embeds_class():
    test_module = ast.Module(body=[_class_node("AnimaPromptEmbeds")], type_ignores=[])
    ast.fix_missing_locations(test_module)
    namespace = {
        "List": list,
        "PromptEmbeds": PromptEmbeds,
        "load_file": load_file,
        "os": __import__("os"),
        "save_file": save_file,
        "torch": torch,
    }
    exec(compile(test_module, str(ANIMA_PATH), "exec"), namespace)
    return namespace["AnimaPromptEmbeds"]


def load_conversion_class():
    source_class = _class_node("AnimaModel")
    method_names = {
        "_strip_ai_toolkit_wrapper_prefix",
        "_add_ai_toolkit_wrapper_prefix",
        "_convert_diffusers_lora_key_to_comfy",
        "convert_lora_weights_before_save",
        "convert_lora_weights_before_load",
    }
    methods = [
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    test_class = ast.ClassDef(
        name="AnimaModel",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    test_module = ast.Module(body=[test_class], type_ignores=[])
    ast.fix_missing_locations(test_module)
    namespace = {}
    exec(compile(test_module, str(ANIMA_PATH), "exec"), namespace)
    return namespace["AnimaModel"]


def load_generation_pipeline_method(factory, unwrap_model):
    source_class = _class_node("AnimaModel")
    method = next(
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_generation_pipeline"
    )
    test_class = ast.ClassDef(
        name="AnimaModel",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    test_module = ast.Module(body=[test_class], type_ignores=[])
    ast.fix_missing_locations(test_module)
    namespace = {
        "AnimaEmbedsToImageBlocks": factory,
        "unwrap_model": unwrap_model,
    }
    exec(compile(test_module, str(ANIMA_PATH), "exec"), namespace)
    return namespace["AnimaModel"].get_generation_pipeline


class AnimaStaticSupportTest(unittest.TestCase):
    def test_registry_config_docs_and_dependency_pin_include_anima(self):
        registry = REGISTRY_PATH.read_text(encoding="utf-8")
        config = CONFIG_PATH.read_text(encoding="utf-8")
        offload = OFFLOAD_PATH.read_text(encoding="utf-8")
        requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8")
        readme = README_PATH.read_text(encoding="utf-8")

        self.assertIn('AnimaModel, = _optional_models(".anima"', registry)
        self.assertIn('("AnimaModel", "anima")', registry)
        self.assertIn("AnimaModel,", registry)
        self.assertIn("'anima'", config)
        self.assertIn('"anima"', offload)
        self.assertIn("c943837899b16cbae2f619b8dd4f7bb6f07dd81a", requirements)
        self.assertIn("circlestone-labs/Anima-Base-v1.0-Diffusers", readme)

    def test_ui_defaults_profiles_and_memory_policy_include_anima(self):
        options = UI_OPTIONS_PATH.read_text(encoding="utf-8")
        profiles = UI_PROFILES_PATH.read_text(encoding="utf-8")
        memory = UI_MEMORY_PATH.read_text(encoding="utf-8")
        start = options.index("name: 'anima'")
        end = options.index("disableSections", start)
        block = options[start:end]

        self.assertIn("label: 'Anima'", block)
        self.assertIn("circlestone-labs/Anima-Base-v1.0-Diffusers", block)
        self.assertIn("'config.process[0].model.quantize': [false, false]", block)
        self.assertIn("'config.process[0].model.quantize_te': [false, false]", block)
        self.assertIn("'config.process[0].train.timestep_type': ['weighted', 'sigmoid']", block)
        self.assertIn("'config.process[0].sample.neg'", block)
        self.assertIn("'anima'", profiles)
        self.assertEqual(memory.count("'anima'"), 2)

    def test_prompt_cache_dispatch_and_nested_progress_bar_fix_are_present(self):
        prompt_utils = PROMPT_UTILS_PATH.read_text(encoding="utf-8")
        anima = ANIMA_PATH.read_text(encoding="utf-8")

        self.assertIn('metadata.get("class_name", "") == "AnimaPromptEmbeds"', prompt_utils)
        self.assertIn("disable_progress_bars(pipeline._blocks)", anima)
        self.assertIn("disable_progress_bars(sub_block)", anima)
        self.assertIn("attach_layer_offloading(", anima)
        self.assertIn("quantize_device=self.quantize_device", anima)


class AnimaBehaviorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Embeds = load_prompt_embeds_class()
        cls.ConversionModel = load_conversion_class()

    def _embeds(self, qwen_length: int, t5_length: int, value: float = 1.0):
        return self.Embeds(
            qwen_prompt_embeds=torch.full((1, qwen_length, 4), value),
            qwen_attention_mask=torch.ones((1, qwen_length), dtype=torch.long),
            t5_input_ids=torch.arange(t5_length, dtype=torch.long).unsqueeze(0),
            t5_attention_mask=torch.ones((1, t5_length), dtype=torch.long),
        )

    def test_prompt_embeds_preserve_integer_tokens_when_casting_and_batching(self):
        embeds = self._embeds(2, 3)
        embeds.to(dtype=torch.float16)

        self.assertEqual(embeds.text_embeds.dtype, torch.float16)
        self.assertEqual(embeds.t5_input_ids.dtype, torch.long)
        expanded = embeds.expand_to_batch(3)
        self.assertEqual(expanded.text_embeds.shape, (3, 2, 4))
        self.assertEqual(expanded.t5_input_ids.shape, (3, 3))
        self.assertIsNot(expanded.text_embeds, embeds.text_embeds)

    def test_prompt_embeds_concat_pads_each_encoder_independently(self):
        first = self._embeds(2, 4, 1.0)
        second = self._embeds(5, 1, 2.0)

        combined = self.Embeds.concat_prompt_embeds([first, second])

        self.assertEqual(combined.text_embeds.shape, (2, 5, 4))
        self.assertEqual(combined.t5_input_ids.shape, (2, 4))
        self.assertTrue(torch.equal(combined.attention_mask[0, 2:], torch.zeros(3, dtype=torch.long)))
        self.assertTrue(torch.equal(combined.t5_attention_mask[1, 1:], torch.zeros(3, dtype=torch.long)))

    def test_prompt_embeds_cache_round_trip_and_dispatch(self):
        embeds = self._embeds(3, 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "anima.safetensors")
            embeds.save(path)
            loaded = self.Embeds.load(path)
            self.assertTrue(torch.equal(loaded.text_embeds, embeds.text_embeds))
            self.assertTrue(torch.equal(loaded.t5_input_ids, embeds.t5_input_ids))

            fake_module = types.ModuleType(
                "extensions_built_in.diffusion_models.anima"
            )
            fake_module.AnimaPromptEmbeds = self.Embeds
            with mock.patch.dict(
                sys.modules,
                {"extensions_built_in.diffusion_models.anima": fake_module},
            ):
                dispatched = PromptEmbeds.load(path)
            self.assertIsInstance(dispatched, self.Embeds)

    def test_lora_export_conversion_covers_transformer_and_conditioner(self):
        model = self.ConversionModel()
        weights = {
            "transformer.transformer.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.ones(1),
            "transformer.text_conditioner.adapter.lora_A.weight": torch.ones(1),
        }

        converted = model.convert_lora_weights_before_save(weights)

        self.assertIn(
            "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight",
            converted,
        )
        self.assertIn(
            "diffusion_model.llm_adapter.adapter.lora_A.weight",
            converted,
        )
        restored = model.convert_lora_weights_before_load(
            {"transformer.blocks.0.attn.to_q.lora_A.weight": torch.ones(1)}
        )
        self.assertIn(
            "transformer.transformer.blocks.0.attn.to_q.lora_A.weight",
            restored,
        )

    def test_generation_pipeline_disables_every_nested_progress_bar(self):
        class Block:
            def __init__(self, *children):
                self.sub_blocks = {
                    str(index): child for index, child in enumerate(children)
                }
                self.disabled = False

            def set_progress_bar_config(self, *, disable):
                self.disabled = disable

        leaf = Block()
        middle = Block(leaf)
        root = Block(middle)

        class Pipeline:
            def __init__(self):
                self._blocks = root
                self.components = None

            def update_components(self, **components):
                self.components = components

            def to(self, device):
                self.device = device
                return self

        pipeline = Pipeline()

        class Factory:
            def init_pipeline(self):
                return pipeline

        get_pipeline = load_generation_pipeline_method(
            Factory, lambda component: component
        )
        model = types.SimpleNamespace(
            trainable_model=types.SimpleNamespace(
                transformer="transformer",
                text_conditioner="conditioner",
            ),
            vae="vae",
            device_torch=torch.device("cpu"),
            get_train_scheduler=lambda: "scheduler",
        )

        result = get_pipeline(model)

        self.assertIs(result, pipeline)
        self.assertTrue(middle.disabled)
        self.assertTrue(leaf.disabled)
        self.assertEqual(pipeline.components["scheduler"], "scheduler")


if __name__ == "__main__":
    unittest.main()
