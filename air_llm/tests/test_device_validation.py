import unittest
from unittest.mock import patch

import torch

from ..airllm.airllm_base import AirLLMBaseModel


class TestDeviceValidation(unittest.TestCase):
    def test_cpu_device_is_always_allowed(self):
        model = AirLLMBaseModel.__new__(AirLLMBaseModel)
        model.running_device = "cpu"
        model.device = torch.device("cpu")
        model._validate_running_device()

    @patch("torch.cuda.is_available", return_value=False)
    def test_unavailable_gpu_fails_before_checkpoint_work(self, is_available):
        model = AirLLMBaseModel.__new__(AirLLMBaseModel)
        model.running_device = "cuda:0"
        model.device = torch.device("cuda:0")
        with self.assertRaises(EnvironmentError):
            model._validate_running_device()
