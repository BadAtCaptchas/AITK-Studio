from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIONS_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "options.ts"
SIMPLE_JOB_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "SimpleJob.tsx"


class ZImageUiConfigTest(unittest.TestCase):
    def test_turbo_defaults_set_safe_lora_network(self):
        source = OPTIONS_PATH.read_text(encoding="utf-8")
        start = source.index("name: 'zimage:turbo'")
        end = source.index("name: 'zimage'", start)
        block = source[start:end]

        self.assertIn("'config.process[0].network.type': ['lora', 'lora']", block)
        self.assertIn("'config.process[0].network.linear': [32, 32]", block)
        self.assertIn("'config.process[0].network.linear_alpha': [32, 32]", block)
        self.assertIn("'config.process[0].network.conv': [undefined, 16]", block)
        self.assertIn("'config.process[0].network.conv_alpha': [undefined, 16]", block)
        self.assertIn("'config.process[0].network.network_kwargs.only_if_contains': [[], []]", block)

    def test_simple_lora_rank_input_rejects_zero(self):
        source = SIMPLE_JOB_PATH.read_text(encoding="utf-8")
        start = source.index('label="Linear Rank"')
        end = source.index("max={1024}", start)
        block = source[start:end]

        self.assertIn("min={1}", block)


if __name__ == "__main__":
    unittest.main()
