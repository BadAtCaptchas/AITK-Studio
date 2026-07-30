import os
from collections.abc import MutableMapping


TELEMETRY_ENABLED_ENV = "AITK_TELEMETRY_ENABLED"
_TRUTHY_VALUES = {"1", "ON", "TRUE", "YES", "ENABLED"}


def telemetry_enabled(env: MutableMapping[str, str] | None = None) -> bool:
    target = os.environ if env is None else env
    value = target.get(TELEMETRY_ENABLED_ENV, "")
    return value.strip().upper() in _TRUTHY_VALUES


def configure_telemetry_environment(
    env: MutableMapping[str, str] | None = None,
) -> bool:
    target = os.environ if env is None else env
    enabled = telemetry_enabled(target)
    target["DISABLE_TELEMETRY"] = "NO" if enabled else "YES"
    target["HF_HUB_DISABLE_TELEMETRY"] = "0" if enabled else "1"
    return enabled
