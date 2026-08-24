import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from airllm.airllm_base import AirLLMBaseModel
from airllm.airllm_phimoe import AirLLMPhiMoE, PhiMoEStreamedExperts
from airllm.auto_model import AutoModel


def _config():
    return types.SimpleNamespace(
        hidden_size=4,
        intermediate_size=3,
        num_local_experts=3,
        hidden_act="silu",
    )


class TestPhiMoEStreamedExperts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.experts = PhiMoEStreamedExperts(_config()).to_empty(device="cpu")
        with torch.no_grad():
            for parameter in self.experts.parameters():
                parameter.uniform_(-0.25, 0.25)

    def test_numeric_child_keys_match_published_checkpoint(self):
        self.assertEqual(
            set(self.experts.state_dict()),
            {
                f"{expert}.{projection}.weight"
                for expert in range(3)
                for projection in ("w1", "w2", "w3")
            },
        )

    def test_output_matches_packed_bank_equations(self):
        hidden = torch.randn(5, 4)
        top_k_index = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 1], [2, 0]])
        top_k_weights = torch.softmax(torch.randn(5, 2), dim=-1)

        actual = self.experts(hidden, top_k_index, top_k_weights)

        gate_up = torch.stack([
            torch.cat((self.experts[idx].w1.weight, self.experts[idx].w3.weight), dim=0)
            for idx in range(3)
        ])
        down = torch.stack([self.experts[idx].w2.weight for idx in range(3)])
        expected = torch.zeros_like(hidden)
        expert_mask = F.one_hot(top_k_index, num_classes=3).permute(2, 1, 0)
        for expert_idx in range(3):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current = hidden[token_idx]
            gate, up = F.linear(current, gate_up[expert_idx]).chunk(2, dim=-1)
            current = F.linear(F.silu(gate) * up, down[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            expected.index_add_(0, token_idx, current)

        torch.testing.assert_close(actual, expected)

    def test_only_routed_expert_modules_execute(self):
        calls = [0, 0, 0]
        hooks = []
        for idx in range(3):
            hooks.append(self.experts[idx].register_forward_hook(
                lambda _module, _args, _out, expert_idx=idx: calls.__setitem__(
                    expert_idx, calls[expert_idx] + 1
                )
            ))
        try:
            self.experts(
                torch.randn(3, 4),
                torch.tensor([[0, 2], [2, 0], [0, 2]]),
                torch.full((3, 2), 0.5),
            )
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(calls, [1, 0, 1])

    def test_slot_runtime_uses_fixed_weight_views_without_calling_expert_modules(self):
        class FakeSlotRuntime:
            def __init__(self, experts):
                self.experts = experts
                self.calls = []

            def execute(self, layer_idx, expert_indices, *, admit, callback):
                self.calls.append((layer_idx, tuple(expert_indices), admit))
                for expert_idx in expert_indices:
                    expert = self.experts[expert_idx]
                    callback(expert_idx, (
                        expert.w1.weight,
                        expert.w2.weight,
                        expert.w3.weight,
                    ))

        hidden = torch.randn(1, 4)
        indices = torch.tensor([[0, 2]])
        weights = torch.tensor([[0.4, 0.6]])
        expected = self.experts(hidden, indices, weights)

        calls = [0, 0, 0]
        hooks = [
            self.experts[idx].register_forward_hook(
                lambda _module, _args, _out, expert_idx=idx: calls.__setitem__(
                    expert_idx, calls[expert_idx] + 1
                )
            )
            for idx in range(3)
        ]
        runtime = FakeSlotRuntime(self.experts)
        self.experts._airllm_slot_runtime = runtime
        self.experts._airllm_layer_idx = 7
        try:
            actual = self.experts(hidden, indices, weights)
        finally:
            for hook in hooks:
                hook.remove()

        torch.testing.assert_close(actual, expected)
        self.assertEqual(calls, [0, 0, 0])
        self.assertEqual(runtime.calls, [(7, (0, 2), True)])

    def test_automodel_selects_phimoe_adapter(self):
        config = types.SimpleNamespace(architectures=["PhiMoEForCausalLM"])
        with patch("airllm.auto_model.AutoConfig.from_pretrained", return_value=config):
            module, class_name = AutoModel.get_module_class("unused")
        self.assertEqual(module, "airllm")
        self.assertEqual(class_name, "AirLLMPhiMoE")

    def test_transformers_5_mlp_layout_gets_checkpoint_aliases(self):
        class PackedExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = nn.Parameter(torch.empty(3, 6, 4, device="meta"))
                self.down_proj = nn.Parameter(torch.empty(3, 4, 3, device="meta"))

        sparse_moe = nn.Module()
        sparse_moe.router = nn.Linear(4, 3, bias=False, device="meta")
        sparse_moe.experts = PackedExperts()
        layer = nn.Module()
        layer.mlp = sparse_moe
        decoder = nn.Module()
        decoder.layers = nn.ModuleList([layer])
        root = nn.Module()
        root.model = decoder

        wrapper = object.__new__(AirLLMPhiMoE)
        wrapper.model = root
        wrapper.config = _config()
        with patch.object(AirLLMBaseModel, "init_model"):
            wrapper.init_model()

        self.assertIs(layer.mlp, layer.block_sparse_moe)
        self.assertIs(layer.mlp.router, layer.block_sparse_moe.gate)
        self.assertIsInstance(layer.mlp.experts, PhiMoEStreamedExperts)
        self.assertEqual(wrapper.config._experts_implementation, "eager")
        self.assertIs(
            root.get_submodule("model.layers.0.block_sparse_moe.experts.0"),
            layer.mlp.experts[0],
        )


if __name__ == "__main__":
    unittest.main()
