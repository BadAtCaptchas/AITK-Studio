import os
import socket
import unittest
from unittest import mock

from toolkit import network_policy


def fake_addrinfo(*addresses):
    return [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))
        for address in addresses
    ]


class NetworkPolicyTest(unittest.TestCase):
    def setUp(self):
        network_policy._allowed_resolved_addresses.clear()

    def test_offline_mode_configures_library_offline_env(self):
        with mock.patch.dict(os.environ, {"AITK_OFFLINE_MODE": "1"}, clear=True):
            self.assertTrue(network_policy.configure_offline_environment())
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_DATASETS_OFFLINE"], "1")

    def test_public_ip_is_blocked_when_offline(self):
        with mock.patch.dict(os.environ, {"AITK_OFFLINE_MODE": "1"}, clear=True):
            with self.assertRaisesRegex(network_policy.OfflineNetworkError, "outside local/private IP space"):
                network_policy.assert_url_allowed("https://8.8.8.8/api", "test request")

    def test_local_and_private_ips_are_allowed_when_offline(self):
        with mock.patch.dict(os.environ, {"AITK_OFFLINE_MODE": "1"}, clear=True):
            for url in (
                "http://127.0.0.1:8675/api",
                "http://10.1.2.3/api",
                "http://172.16.0.1/api",
                "http://192.168.1.10/api",
                "http://[::1]/api",
            ):
                with self.subTest(url=url):
                    network_policy.assert_url_allowed(url, "test request")

    def test_hostname_must_resolve_only_to_local_private_addresses(self):
        with mock.patch.dict(os.environ, {"AITK_OFFLINE_MODE": "1"}, clear=True):
            with mock.patch.object(
                network_policy,
                "_original_getaddrinfo",
                return_value=fake_addrinfo("192.168.1.20", "8.8.8.8"),
            ):
                with self.assertRaisesRegex(network_policy.OfflineNetworkError, "8.8.8.8"):
                    network_policy.assert_host_allowed("mixed.example", 443, "test request")

    def test_allowed_host_bypasses_offline_mode_and_records_resolved_ip(self):
        with mock.patch.dict(
            os.environ,
            {
                "AITK_OFFLINE_MODE": "1",
                "AITK_OFFLINE_ALLOWED_HOSTS": "worker.example",
            },
            clear=True,
        ):
            with mock.patch.object(
                network_policy,
                "_original_getaddrinfo",
                return_value=fake_addrinfo("203.0.113.10"),
            ):
                network_policy.assert_host_allowed("worker.example", 443, "worker")
                self.assertTrue(network_policy.is_allowed_ip("203.0.113.10"))


if __name__ == "__main__":
    unittest.main()
