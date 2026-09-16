import unittest

import torch

import airllm.utils as utils
from airllm.airllm_base import AirLLMBaseModel
from airllm.utils import default_device


class TestDefaultDevice(unittest.TestCase):
    """
    A plain AutoModel.from_pretrained(repo_id) used to hard-code cuda:0, so it raised on any
    machine without an NVIDIA GPU even though layer streaming is device-agnostic.
    """

    def setUp(self):
        self._cuda = torch.cuda.is_available
        self._mps = getattr(torch.backends, 'mps', None)

    def tearDown(self):
        torch.cuda.is_available = self._cuda
        if self._mps is not None:
            torch.backends.mps.is_available = self._mps.is_available

    def _fake(self, cuda, mps):
        torch.cuda.is_available = lambda: cuda
        if getattr(torch.backends, 'mps', None) is not None:
            torch.backends.mps.is_available = lambda: mps

    def test_prefers_cuda_when_available(self):
        self._fake(cuda=True, mps=False)
        self.assertEqual(default_device(), 'cuda:0')

    def test_falls_back_to_mps_on_apple_silicon(self):
        if getattr(torch.backends, 'mps', None) is None:
            self.skipTest('this torch build has no mps backend to fall back to')
        self._fake(cuda=False, mps=True)
        self.assertEqual(default_device(), 'mps')

    def test_falls_back_to_cpu_when_no_accelerator(self):
        self._fake(cuda=False, mps=False)
        self.assertEqual(default_device(), 'cpu')

    def test_cuda_preferred_over_mps(self):
        self._fake(cuda=True, mps=True)
        self.assertEqual(default_device(), 'cuda:0')

    def test_init_uses_default_device_when_none_passed(self):
        # The __init__ default is now None, resolved via default_device().
        self._fake(cuda=False, mps=False)
        obj = AirLLMBaseModel.__new__(AirLLMBaseModel)   # skip the heavy __init__
        device = None
        obj.running_device = device if device is not None else utils.default_device()
        self.assertEqual(obj.running_device, 'cpu')

    def test_explicit_device_is_still_honoured(self):
        # Control: an explicit device must win over auto-detection, even a "wrong" one.
        self._fake(cuda=True, mps=False)
        for explicit in ('cpu', 'cuda:1', 'mps'):
            device = explicit
            resolved = device if device is not None else utils.default_device()
            self.assertEqual(resolved, explicit)


if __name__ == '__main__':
    unittest.main()
