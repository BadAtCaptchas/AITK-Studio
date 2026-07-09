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

    def test_secret_environment_variables_cannot_be_interpolated(self):
        secret_variables = (
            "HF_TOKEN",
            "OPENROUTER_API_KEY",
            "AITK_ENCRYPTED_DATASET_KEYS_B64",
            "AI_TOOLKIT_AUTH",
            "AITK_MONGODB_URI",
        )

        for variable in secret_variables:
            with self.subTest(variable=variable):
                secret = f"{variable.lower()}_should_not_appear"
                config_path = self.write_config(
                    {
                        "job": "generate",
                        "config": {
                            "name": "blocked_secret",
                            # The destination key is deliberately innocuous.
                            "process": [{"type": "noop", "folder": f"${{{variable}}}"}],
                        },
                    }
                )

                with mock.patch.dict(os.environ, {variable: secret}, clear=False):
                    with self.assertRaises(ValueError) as raised:
                        get_config(config_path)

                message = str(raised.exception)
                self.assertIn(variable, message)
                self.assertNotIn(secret, message)

    def test_unknown_environment_variables_are_denied_by_default(self):
        config_path = self.write_config(
            {
                "job": "generate",
                "config": {
                    "name": "blocked_unknown",
                    "process": [{"type": "noop", "folder": "${HARMLESS_LOOKING_VALUE}"}],
                },
            }
        )

        with mock.patch.dict(os.environ, {"HARMLESS_LOOKING_VALUE": "not-secret"}, clear=False):
            with self.assertRaisesRegex(ValueError, "HARMLESS_LOOKING_VALUE"):
                get_config(config_path)

    def test_non_sensitive_environment_interpolation_still_works(self):
        config_path = self.write_config(
            {
                "job": "generate",
                "config": {
                    "name": "allowed_env",
                    "process": [{"type": "noop", "folder": "${MODELS_PATH}/base"}],
                },
            }
        )

        with mock.patch.dict(os.environ, {"MODELS_PATH": "expanded-models"}, clear=False):
            config = get_config(config_path)

        self.assertEqual(config["config"]["process"][0]["folder"], "expanded-models/base")

    def test_shell_style_environment_expansion_is_rejected(self):
        config_path = self.write_config(
            {
                "job": "generate",
                "config": {
                    "name": "invalid_placeholder",
                    "process": [{"type": "noop", "folder": "${MODELS_PATH:-models}"}],
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "Invalid config environment placeholder"):
            get_config(config_path)

    def test_environment_value_cannot_inject_config_syntax(self):
        config_path = self.write_config(
            {
                "job": "generate",
                "config": {
                    "name": "literal_value",
                    "process": [{"type": "noop", "folder": "${MODELS_PATH}"}],
                },
            }
        )
        path_with_json_syntax = 'models", "injected": "value'

        with mock.patch.dict(os.environ, {"MODELS_PATH": path_with_json_syntax}, clear=False):
            config = get_config(config_path)

        process = config["config"]["process"][0]
        self.assertEqual(process["folder"], path_with_json_syntax)
        self.assertNotIn("injected", process)

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
