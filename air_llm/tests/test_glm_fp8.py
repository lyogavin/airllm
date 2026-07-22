import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from ..airllm.airllm_fp8 import AirLLMFP8Experts, AirLLMFP8Linear, replace_with_airllm_fp8


class _OriginalExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.hidden_dim = 4
        self.intermediate_dim = 4
        self.gate_up_proj = nn.Parameter(torch.empty(2, 8, 4, device="meta"))
        self.down_proj = nn.Parameter(torch.empty(2, 4, 4, device="meta"))
        self.act_fn = torch.nn.functional.silu


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False, device="meta")
        self.experts = _OriginalExperts()


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _TinyLayer()


class TestGlmFp8(unittest.TestCase):
    def test_block_scaled_linear_matches_explicit_dequantization(self):
        module = AirLLMFP8Linear(4, 4, (2, 2), device="cpu")
        weight = torch.tensor(
            [[1, 2, 3, 4], [2, 1, 0, -1], [1, -1, 2, -2], [4, 3, 2, 1]],
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn)
        scales = torch.tensor([[0.5, 1.5], [2.0, 0.25]])
        module.weight.data.copy_(weight)
        module.weight_scale_inv.data.copy_(scales)
        inputs = torch.tensor([[1.0, 2.0, -1.0, 0.5]])

        expanded = scales.repeat_interleave(2, 0).repeat_interleave(2, 1)
        expected = torch.nn.functional.linear(inputs, weight.float() * expanded)
        torch.testing.assert_close(module(inputs), expected)

    def test_expert_parameter_names_match_checkpoint_layout(self):
        experts = AirLLMFP8Experts(_OriginalExperts(), (2, 2))
        names = set(experts.state_dict())
        self.assertIn("0.gate_proj.weight", names)
        self.assertIn("0.gate_proj.weight_scale_inv", names)
        self.assertIn("1.down_proj.weight", names)
        self.assertNotIn("experts.0.gate_proj.weight", names)

    def test_replacement_uses_only_scaled_checkpoint_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            weight_names = {
                "layer.proj.weight": "model.safetensors",
                "layer.proj.weight_scale_inv": "model.safetensors",
                "layer.experts.0.gate_proj.weight": "model.safetensors",
                "layer.experts.0.gate_proj.weight_scale_inv": "model.safetensors",
            }
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_names}), encoding="utf-8"
            )
            model = _TinyModel()
            expert_count, linear_count = replace_with_airllm_fp8(
                model,
                root,
                {"weight_block_size": [2, 2]},
            )

        self.assertEqual(expert_count, 1)
        self.assertEqual(linear_count, 1)
        self.assertIsInstance(model.layer.proj, AirLLMFP8Linear)
        self.assertIsInstance(model.layer.experts, AirLLMFP8Experts)
        self.assertIn("layer.experts.0.gate_proj.weight", model.state_dict())

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA/ROCm device")
    def test_block_scaled_linear_runs_on_gpu(self):
        module = AirLLMFP8Linear(128, 128, (128, 128), device="cuda")
        module.weight.data.copy_(torch.randn(128, 128, device="cuda").to(torch.float8_e4m3fn))
        module.weight_scale_inv.data.fill_(0.125)
        inputs = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
        output = module(inputs)
        self.assertEqual(output.shape, (2, 128))
        self.assertTrue(torch.isfinite(output).all())
