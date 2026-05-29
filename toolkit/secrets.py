import re
from collections import OrderedDict
from typing import Any


SENSITIVE_CONFIG_KEY_RE = re.compile(
    r"(token|secret|password|api[_-]?key|access[_-]?key|private[_-]?key|credential|prompt)",
    re.IGNORECASE,
)
REDACTED_VALUE = "[REDACTED]"


def is_sensitive_config_key(key: Any) -> bool:
    return isinstance(key, str) and SENSITIVE_CONFIG_KEY_RE.search(key) is not None


def redact_secrets(value: Any):
    if isinstance(value, OrderedDict):
        return OrderedDict(
            (key, REDACTED_VALUE if is_sensitive_config_key(key) else redact_secrets(item))
            for key, item in value.items()
        )

    if isinstance(value, dict):
        return {
            key: REDACTED_VALUE if is_sensitive_config_key(key) else redact_secrets(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [redact_secrets(item) for item in value]

    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)

    return value
