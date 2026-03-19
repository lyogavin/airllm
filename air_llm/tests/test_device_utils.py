import unittest
import torch

from airllm.device_utils import (
    get_device_type,
    is_cuda_device,
    is_directml_device,
    can_pin_memory,
    supports_bitsandbytes,
    is_directml_available,
    get_directml_device,
    empty_cache,
    get_free_memory_bytes,
)


class TestGetDeviceType(unittest.TestCase):

    def test_cuda_variants(self):
        self.assertEqual(get_device_type("cuda"), "cuda")
        self.assertEqual(get_device_type("cuda:0"), "cuda")
        self.assertEqual(get_device_type("cuda:1"), "cuda")
        self.assertEqual(get_device_type("CUDA:0"), "cuda")

    def test_directml_variants(self):
        self.assertEqual(get_device_type("privateuseone:0"), "directml")
        self.assertEqual(get_device_type("privateuseone:1"), "directml")
        self.assertEqual(get_device_type("dml:0"), "directml")
        self.assertEqual(get_device_type("directml:0"), "directml")

    def test_mps(self):
        self.assertEqual(get_device_type("mps"), "mps")
        self.assertEqual(get_device_type("mps:0"), "mps")

    def test_cpu(self):
        self.assertEqual(get_device_type("cpu"), "cpu")


class TestDeviceBoolHelpers(unittest.TestCase):

    def test_is_cuda_device(self):
        self.assertTrue(is_cuda_device("cuda:0"))
        self.assertFalse(is_cuda_device("privateuseone:0"))
        self.assertFalse(is_cuda_device("cpu"))
        self.assertFalse(is_cuda_device("mps"))

    def test_is_directml_device(self):
        self.assertTrue(is_directml_device("privateuseone:0"))
        self.assertTrue(is_directml_device("dml:0"))
        self.assertFalse(is_directml_device("cuda:0"))
        self.assertFalse(is_directml_device("cpu"))

    def test_can_pin_memory(self):
        # pin_memory is only useful when copying to CUDA
        self.assertTrue(can_pin_memory("cuda:0"))
        self.assertFalse(can_pin_memory("privateuseone:0"))
        self.assertFalse(can_pin_memory("mps"))
        self.assertFalse(can_pin_memory("cpu"))

    def test_supports_bitsandbytes(self):
        # bitsandbytes is NVIDIA-only
        self.assertTrue(supports_bitsandbytes("cuda:0"))
        self.assertFalse(supports_bitsandbytes("privateuseone:0"))
        self.assertFalse(supports_bitsandbytes("mps"))
        self.assertFalse(supports_bitsandbytes("cpu"))


class TestEmptyCache(unittest.TestCase):

    def test_empty_cache_cpu_does_not_crash(self):
        empty_cache("cpu")

    def test_empty_cache_directml_does_not_crash(self):
        empty_cache("privateuseone:0")

    @unittest.skipUnless(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "MPS not available on this machine"
    )
    def test_empty_cache_mps_does_not_crash(self):
        empty_cache("mps")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_empty_cache_cuda(self):
        empty_cache("cuda:0")


class TestGetFreeMemoryBytes(unittest.TestCase):

    def test_returns_minus_one_for_cpu(self):
        self.assertEqual(get_free_memory_bytes("cpu"), -1)

    def test_returns_minus_one_for_directml(self):
        self.assertEqual(get_free_memory_bytes("privateuseone:0"), -1)

    def test_returns_minus_one_for_mps(self):
        self.assertEqual(get_free_memory_bytes("mps"), -1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_returns_positive_for_cuda(self):
        free = get_free_memory_bytes("cuda:0")
        self.assertGreater(free, 0)


class TestDirectMLDetection(unittest.TestCase):

    def test_is_directml_available_returns_bool(self):
        result = is_directml_available()
        self.assertIsInstance(result, bool)

    @unittest.skipUnless(is_directml_available(), "torch-directml not installed")
    def test_get_directml_device_returns_valid_device(self):
        dev = get_directml_device(0)
        self.assertIsNotNone(dev)
        self.assertIn("privateuseone", str(dev))

    def test_get_directml_device_returns_none_when_unavailable(self):
        if is_directml_available():
            self.skipTest("torch-directml is installed on this machine")
        result = get_directml_device(0)
        self.assertIsNone(result)
