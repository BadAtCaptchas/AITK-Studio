import os
import socket
from ipaddress import ip_address, ip_network
from typing import Iterable
from urllib.parse import urlparse


OFFLINE_MODE_ENV_VARS = ("AITK_OFFLINE_MODE", "AI_TOOLKIT_OFFLINE_MODE")
OFFLINE_ALLOWED_HOSTS_ENV = "AITK_OFFLINE_ALLOWED_HOSTS"
HF_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")

_TRUTHY_VALUES = {"1", "ON", "TRUE", "YES"}
_LOCAL_PRIVATE_NETWORKS = tuple(
    ip_network(value)
    for value in (
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    )
)

_installed = False
_allowed_resolved_addresses: set[str] = set()
_original_getaddrinfo = socket.getaddrinfo
_original_create_connection = socket.create_connection
_original_socket_connect = socket.socket.connect


class OfflineNetworkError(RuntimeError):
    pass


def env_flag(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().upper() in _TRUTHY_VALUES


def is_offline_mode_enabled() -> bool:
    return any(env_flag(name) for name in OFFLINE_MODE_ENV_VARS)


def configure_offline_environment() -> bool:
    if not is_offline_mode_enabled():
        return False

    for name in HF_OFFLINE_ENV_VARS:
        os.environ[name] = os.environ.get(name) or "1"
    os.environ["DISABLE_TELEMETRY"] = os.environ.get("DISABLE_TELEMETRY") or "YES"
    return True


def _normalize_hostname(value: object) -> str:
    hostname = str(value or "").strip().lower().strip("[]")
    return hostname[:-1] if hostname.endswith(".") else hostname


def _allowed_hosts() -> set[str]:
    raw = os.getenv(OFFLINE_ALLOWED_HOSTS_ENV, "")
    return {_normalize_hostname(item) for item in raw.split(",") if _normalize_hostname(item)}


def _is_allowed_host(hostname: str) -> bool:
    return _normalize_hostname(hostname) in _allowed_hosts()


def _record_allowed_addresses(addresses: Iterable[str]):
    for address in addresses:
        normalized = str(address or "").strip()
        if normalized:
            _allowed_resolved_addresses.add(normalized)


def is_local_private_ip(value: object) -> bool:
    try:
        parsed = ip_address(str(value).strip())
    except ValueError:
        return False

    if parsed.is_unspecified or parsed.is_multicast:
        return False

    if parsed.version == 6 and parsed.ipv4_mapped is not None:
        parsed = parsed.ipv4_mapped

    return any(parsed in network for network in _LOCAL_PRIVATE_NETWORKS)


def is_allowed_ip(value: object) -> bool:
    address = str(value or "").strip()
    return address in _allowed_resolved_addresses or is_local_private_ip(address)


def _resolve_addresses(hostname: str, port: int | None = None) -> list[str]:
    host = _normalize_hostname(hostname)
    if not host:
        return []

    try:
        ip_address(host)
        return [host]
    except ValueError:
        pass

    infos = _original_getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addresses: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr:
            addresses.append(str(sockaddr[0]))
    return list(dict.fromkeys(addresses))


def assert_host_allowed(hostname: str, port: int | None = None, feature: str = "request") -> None:
    if not is_offline_mode_enabled():
        return

    host = _normalize_hostname(hostname)
    if not host:
        raise OfflineNetworkError(f"Offline mode blocked {feature}: missing hostname")

    if _is_allowed_host(host):
        addresses = _resolve_addresses(host, port)
        _record_allowed_addresses(addresses)
        return

    try:
        addresses = _resolve_addresses(host, port)
    except OSError as exc:
        raise OfflineNetworkError(f"Offline mode blocked {feature} to {host}: DNS lookup failed: {exc}") from exc

    if not addresses:
        raise OfflineNetworkError(f"Offline mode blocked {feature} to {host}: DNS lookup returned no addresses")

    blocked = [address for address in addresses if not is_allowed_ip(address)]
    if blocked:
        raise OfflineNetworkError(
            f"Offline mode blocked {feature} to {host}: resolved outside local/private IP space ({', '.join(blocked)})"
        )


def assert_url_allowed(url: str, feature: str = "request") -> None:
    if not is_offline_mode_enabled():
        return

    parsed = urlparse(str(url))
    if parsed.scheme and parsed.scheme.lower() not in {"http", "https", "ws", "wss"}:
        raise OfflineNetworkError(f"Offline mode blocked {feature}: unsupported URL protocol {parsed.scheme}")

    assert_host_allowed(parsed.hostname or "", parsed.port, feature)


def _guarded_getaddrinfo(host, port, *args, **kwargs):
    if is_offline_mode_enabled() and host is not None:
        assert_host_allowed(str(host), int(port) if port is not None else None, "DNS lookup")
    return _original_getaddrinfo(host, port, *args, **kwargs)


def _guarded_socket_connect(self, address):
    if is_offline_mode_enabled() and isinstance(address, tuple) and len(address) >= 2:
        host = str(address[0])
        port = int(address[1])
        if is_allowed_ip(host):
            return _original_socket_connect(self, address)
        assert_host_allowed(host, port, "socket connection")
    return _original_socket_connect(self, address)


def _guarded_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, *args, **kwargs):
    if is_offline_mode_enabled() and isinstance(address, tuple) and len(address) >= 2:
        assert_host_allowed(str(address[0]), int(address[1]), "socket connection")
    return _original_create_connection(address, timeout, source_address, *args, **kwargs)


def install_offline_network_guard() -> bool:
    global _installed

    configure_offline_environment()
    if _installed or not is_offline_mode_enabled():
        return _installed

    socket.getaddrinfo = _guarded_getaddrinfo
    socket.socket.connect = _guarded_socket_connect
    socket.create_connection = _guarded_create_connection
    _installed = True
    return True
