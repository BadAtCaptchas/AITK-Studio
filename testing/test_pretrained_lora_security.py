import os
import unittest


class PretrainedLoraSecurityTest(unittest.TestCase):
    def read_source(self, *parts):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, *parts)
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()

    def test_pretrained_lora_guard_matches_case_sensitive_loader(self):
        source = self.read_source("jobs", "process", "BaseSDTrainProcess.py")

        self.assertIn(
            'os.path.splitext(str(pretrained_path))[1] != ".safetensors"',
            source,
        )
        self.assertNotIn(
            'str(pretrained_path).lower().endswith(".safetensors")',
            source,
        )

    def test_merge_network_on_save_excludes_pretrained_lora_from_full_model_resume(self):
        source = self.read_source("jobs", "process", "BaseSDTrainProcess.py")

        self.assertIn(
            "def get_latest_save_path(self, name=None, post='', include_pretrained_lora=True)",
            source,
        )
        self.assertGreaterEqual(
            source.count("get_latest_save_path(include_pretrained_lora=False)"),
            2,
        )
        self.assertIn(
            "elif self.train_config.merge_network_on_save and self.network_config.pretrained_lora_path is not None:",
            source,
        )
        self.assertIn("Loading initial lora from pretrained lora path", source)

    def test_save_now_path_saves_once(self):
        source = self.read_source(
            "extensions_built_in", "sd_trainer", "DiffusionTrainer.py"
        )
        start = source.index("    def maybe_save(self):")
        end = source.index("    async def _update_key", start)
        maybe_save_source = source[start:end]

        self.assertEqual(maybe_save_source.count("self.save(self.step_num)"), 1)


if __name__ == "__main__":
    unittest.main()
