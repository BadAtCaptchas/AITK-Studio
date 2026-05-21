import io
import json
import os
import tempfile
import unittest
from collections import OrderedDict
from contextlib import redirect_stdout
from unittest import mock

from jobs.process.BaseProcess import BaseProcess
from toolkit.config import get_config
from toolkit.secrets import REDACTED_VALUE, redact_secrets


class ConfigSecretHandlingTest(unittest.TestCase):
    def write_config(self, payload):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        with handle:
            json.dump(payload, handle)
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        return handle.name

    def test_hf_token_cannot_be_interpolated_from_config(self):
        secret = "hf_secret_should_not_appear"
        config_path = self.write_config(
            {
                "job": "generate",
                "config": {
                    "name": "blocked_secret",
                    "process": [{"type": "noop", "leak": "${HF_TOKEN}"}],
                },
            }
        )

        with mock.patch.dict(os.environ, {"HF_TOKEN": secret}, clear=False):
            with self.assertRaises(ValueError) as raised:
                get_config(config_path)

        message = str(raised.exception)
        self.assertIn("HF_TOKEN", message)
        self.assertNotIn(secret, message)

    def test_non_sensitive_environment_interpolation_still_works(self):
        config_path = self.write_config(
            {
                "job": "generate",
                "config": {
                    "name": "allowed_env",
                    "process": [{"type": "noop", "folder": "${AITK_TEST_FOLDER}"}],
                },
            }
        )

        with mock.patch.dict(os.environ, {"AITK_TEST_FOLDER": "expanded-folder"}, clear=False):
            config = get_config(config_path)

        self.assertEqual(config["config"]["process"][0]["folder"], "expanded-folder")

    def test_redact_secrets_redacts_nested_sensitive_keys(self):
        config = OrderedDict(
            [
                ("token", "top-secret"),
                ("nested", OrderedDict([("api_key", "nested-secret"), ("safe", "visible")])),
                ("items", [{"password": "item-secret"}]),
            ]
        )

        redacted = redact_secrets(config)

        self.assertEqual(redacted["token"], REDACTED_VALUE)
        self.assertEqual(redacted["nested"]["api_key"], REDACTED_VALUE)
        self.assertEqual(redacted["nested"]["safe"], "visible")
        self.assertEqual(redacted["items"][0]["password"], REDACTED_VALUE)

    def test_base_process_prints_redacted_config(self):
        secret = "hf_secret_should_not_be_logged"

        class FakeJob:
            name = "redacted_job"
            meta = OrderedDict()

        config = OrderedDict(
            [
                ("type", "noop"),
                ("hf_token", secret),
                ("nested", OrderedDict([("safe", "visible")])),
            ]
        )

        output = io.StringIO()
        with redirect_stdout(output):
            BaseProcess(0, FakeJob(), config)

        rendered = output.getvalue()
        self.assertIn(REDACTED_VALUE, rendered)
        self.assertIn("visible", rendered)
        self.assertNotIn(secret, rendered)


if __name__ == "__main__":
    unittest.main()
