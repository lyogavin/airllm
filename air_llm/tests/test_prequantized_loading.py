import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from ..airllm.airllm_base import AirLLMBaseModel


class _PackedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_packed = nn.Parameter(
            torch.empty(4, dtype=torch.int32, device="meta"), requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(2, dtype=torch.float32, device="meta"), requires_grad=False
        )


class TestPrequantizedLoading(unittest.TestCase):
    def test_packed_weights_and_scales_keep_checkpoint_dtypes(self):
        streamed = AirLLMBaseModel.__new__(AirLLMBaseModel)
        streamed.model = _PackedModule()
        streamed.running_device = "cpu"
        streamed.running_dtype = torch.bfloat16
        streamed.hf_quantizer = SimpleNamespace(
            pre_quantized=True,
            param_needs_quantization=lambda model, name: False,
        )
        state = {
            "weight_packed": torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            "weight_scale": torch.tensor([0.25, 0.5], dtype=torch.float32),
        }

        streamed.move_layer_to_device(state)

        self.assertEqual(streamed.model.weight_packed.dtype, torch.int32)
        self.assertEqual(streamed.model.weight_scale.dtype, torch.float32)
