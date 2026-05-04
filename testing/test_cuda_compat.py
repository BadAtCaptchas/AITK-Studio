import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.cuda_compat import (
    check_blackwell_cuda_compatibility,
    get_cuda_compatibility_report,
    parse_cuda_version,
)


class _FakeVersion:
    def __init__(self, cuda):
        self.cuda = cuda


class _FakeCuda:
    def __init__(self, capability=(12, 0), cuda_available=True, arch_list=None):
        self._capability = capability
        self._cuda_available = cuda_available
        self._arch_list = arch_list or []

    def is_available(self):
        return self._cuda_available

    def device_count(self):
        return 1 if self._cuda_available else 0

    def get_device_name(self, index):
        return "NVIDIA GeForce RTX 5080"

    def get_device_capability(self, index):
        return self._capability

    def get_arch_list(self):
        return self._arch_list


class _FakeTorch:
    def __init__(self, cuda_version, capability=(12, 0), arch_list=None, cuda_available=True):
        self.__version__ = f"test+cu{cuda_version}"
        self.version = _FakeVersion(cuda_version)
        self.cuda = _FakeCuda(
            capability=capability,
            cuda_available=cuda_available,
            arch_list=arch_list,
        )


class CudaCompatTest(unittest.TestCase):
    def test_parse_cuda_version(self):
        self.assertEqual(parse_cuda_version("12.8"), (12, 8))
        self.assertEqual(parse_cuda_version("13"), (13, 0))
        self.assertIsNone(parse_cuda_version(None))

    def test_blackwell_with_old_cuda_fails(self):
        torch_module = _FakeTorch("12.6", arch_list=["sm_90"])

        with self.assertRaises(RuntimeError) as ctx:
            check_blackwell_cuda_compatibility(torch_module)

        message = str(ctx.exception)
        self.assertIn("Blackwell", message)
        self.assertIn("torch==2.9.1", message)
        self.assertIn("cu128", message)

    def test_blackwell_with_cuda_128_and_sm120_passes(self):
        torch_module = _FakeTorch("12.8", arch_list=["sm_90", "sm_120"])
        report = check_blackwell_cuda_compatibility(torch_module)

        self.assertEqual(report["problems"], [])

    def test_non_blackwell_and_no_cuda_do_not_fail(self):
        non_blackwell = _FakeTorch("12.6", capability=(9, 0), arch_list=["sm_90"])
        no_cuda = _FakeTorch("12.6", cuda_available=False)

        self.assertEqual(get_cuda_compatibility_report(non_blackwell)["problems"], [])
        self.assertEqual(get_cuda_compatibility_report(no_cuda)["problems"], [])


if __name__ == "__main__":
    unittest.main()

