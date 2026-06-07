import os
import unittest
from unittest import mock

from toolkit.hf_offline import is_hf_offline_mode


class HuggingFaceOfflineTest(unittest.TestCase):
    def test_offline_mode_accepts_standard_truthy_env_values(self):
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            with self.subTest(name=name):
                with mock.patch.dict(os.environ, {name: "1"}, clear=True):
                    self.assertTrue(is_hf_offline_mode())

    def test_offline_mode_ignores_falsey_env_values(self):
        with mock.patch.dict(
            os.environ,
            {
                "HF_HUB_OFFLINE": "0",
                "TRANSFORMERS_OFFLINE": "false",
                "HF_DATASETS_OFFLINE": "",
            },
            clear=True,
        ):
            self.assertFalse(is_hf_offline_mode())


if __name__ == "__main__":
    unittest.main()
