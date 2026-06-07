import os
import unittest


class PretrainedLoraSecurityTest(unittest.TestCase):
    def test_pretrained_lora_guard_matches_case_sensitive_loader(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_process_path = os.path.join(
            project_root, "jobs", "process", "BaseSDTrainProcess.py"
        )

        with open(train_process_path, "r", encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn(
            'os.path.splitext(str(pretrained_path))[1] != ".safetensors"',
            source,
        )
        self.assertNotIn(
            'str(pretrained_path).lower().endswith(".safetensors")',
            source,
        )


if __name__ == "__main__":
    unittest.main()
