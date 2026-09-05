"""Unit tests for streamed LoRA pieces that do not need a GPU or Qwen weights."""

import math
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_AIRLLM_DIR = Path(__file__).resolve().parents[1] / "airllm"

if "airllm" not in sys.modules:
    _pkg = types.ModuleType("airllm")
    _pkg.__path__ = [str(_AIRLLM_DIR)]
    sys.modules["airllm"] = _pkg

from airllm.chunked_ce import chunked_linear_cross_entropy
from airllm.lora_linear import (
    LoRALinear,
    LoRAPackedExperts,
    inject_lora,
    inject_packed_expert_lora,
    lora_parameters,
    lora_state_dict,
    load_lora_state_dict,
)


class TestLoRALinear(unittest.TestCase):
    def test_base_is_frozen_adapters_train(self):
        torch.manual_seed(0)
        linear = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.randn(4, 8))
        frozen = linear.weight.detach().clone()
        wrap = LoRALinear(linear, r=2, alpha=4, device=torch.device("cpu"), dtype=torch.float32)
        x = torch.randn(3, 8)
        y = torch.randn(3, 4)
        opt = torch.optim.Adam(lora_parameters(wrap), lr=0.05)
        before = F.mse_loss(wrap(x), y).item()
        for _ in range(30):
            opt.zero_grad()
            F.mse_loss(wrap(x), y).backward()
            opt.step()
        after = F.mse_loss(wrap(x), y).item()
        self.assertTrue(math.isfinite(after))
        self.assertLess(after, before)
        self.assertTrue(torch.equal(wrap.weight, frozen))
        self.assertFalse(wrap.weight.requires_grad)
        self.assertTrue(wrap.lora_A.requires_grad)

    def test_inject_and_roundtrip_state(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(4, 4, bias=False)
                self.other = nn.Linear(4, 4, bias=False)

        block = Block()
        n = inject_lora(block, target_modules=("q_proj",), r=2, alpha=2,
                        device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(n, 1)
        self.assertIsInstance(block.q_proj, LoRALinear)
        self.assertIsInstance(block.other, nn.Linear)
        state = lora_state_dict(block)
        self.assertTrue(any("lora_A" in k for k in state))
        block.q_proj.lora_A.data.zero_()
        load_lora_state_dict(block, state)
        self.assertTrue(torch.allclose(block.q_proj.lora_A.cpu(), state["q_proj.lora_A"]))


class TestChunkedCE(unittest.TestCase):
    def test_matches_full_cross_entropy(self):
        torch.manual_seed(1)
        n, h, v = 6, 8, 21
        hidden = torch.randn(n, h, requires_grad=True)
        weight = torch.randn(v, h)
        labels = torch.tensor([0, 3, 20, -100, 7, 1])

        loss_ref = F.cross_entropy(hidden @ weight.t(), labels, ignore_index=-100)
        loss = chunked_linear_cross_entropy(
            hidden, weight, labels, chunk_size=5, ignore_index=-100)

        self.assertTrue(torch.allclose(loss, loss_ref, rtol=1e-4, atol=1e-5))

        loss_ref.backward()
        g_ref = hidden.grad.clone()
        hidden.grad = None
        hidden2 = hidden.detach().clone().requires_grad_(True)
        loss2 = chunked_linear_cross_entropy(
            hidden2, weight, labels, chunk_size=5, ignore_index=-100)
        loss2.backward()
        self.assertTrue(torch.allclose(hidden2.grad, g_ref, rtol=1e-4, atol=1e-5))

    def test_batched_shape_and_cpu_weight(self):
        torch.manual_seed(2)
        hidden = torch.randn(2, 4, 8, requires_grad=True)
        weight = torch.randn(11, 8)
        labels = torch.randint(0, 11, (2, 4))
        loss = chunked_linear_cross_entropy(hidden, weight, labels, chunk_size=3)
        loss.backward()
        self.assertEqual(tuple(hidden.grad.shape), (2, 4, 8))
        self.assertTrue(math.isfinite(loss.item()))


class _OffloadPack:
    """Stand-in for disk offload: weight lives in a CPU dict between calls."""

    def __init__(self, module, key="weight"):
        self.module = module
        self.key = key
        self.store = {key: module.weight.detach().clone()}

    def load(self):
        with torch.no_grad():
            self.module.weight.copy_(self.store[self.key])

    def evict(self):
        with torch.no_grad():
            self.module.weight.zero_()


class _TinyStream(torch.autograd.Function):
    pack = None

    @staticmethod
    def forward(ctx, hidden):
        pack = _TinyStream.pack
        ctx.pack = pack
        h = hidden.detach()
        pack.load()
        with torch.no_grad():
            out = pack.module(h)
        pack.evict()
        ctx.save_for_backward(h)
        return out.detach()

    @staticmethod
    def backward(ctx, grad):
        pack = ctx.pack
        h = ctx.saved_tensors[0].detach().requires_grad_(True)
        pack.load()
        with torch.enable_grad():
            out = pack.module(h)
            out.backward(grad)
        pack.evict()
        return h.grad


class TestStreamedLoRA(unittest.TestCase):
    def test_streamed_lora_matches_resident_grads(self):
        torch.manual_seed(3)
        linear = nn.Linear(8, 8, bias=False)
        nn.init.normal_(linear.weight)
        wrap = LoRALinear(linear, r=2, alpha=4, device=torch.device("cpu"), dtype=torch.float32)
        x = torch.randn(4, 8)
        go = torch.randn(4, 8)

        y = wrap(x)
        y.backward(go)
        gA = wrap.lora_A.grad.clone()
        gB = wrap.lora_B.grad.clone()
        wrap.lora_A.grad = None
        wrap.lora_B.grad = None

        pack = _OffloadPack(wrap)
        pack.evict()
        _TinyStream.pack = pack
        try:
            x2 = x.detach().requires_grad_(True)
            y2 = _TinyStream.apply(x2)
            y2.backward(go)
        finally:
            _TinyStream.pack = None
        pack.load()

        self.assertTrue(torch.allclose(wrap.lora_A.grad, gA, rtol=1e-4, atol=1e-5))
        self.assertTrue(torch.allclose(wrap.lora_B.grad, gB, rtol=1e-4, atol=1e-5))


class _FakePackedExperts(nn.Module):
    """Stand-in for transformers' Qwen4ExpTextExperts (3D Parameters, not Linear)."""

    def __init__(self, n_exp=4, hidden=8, intermediate=6):
        super().__init__()
        self.num_experts = n_exp
        self.hidden_dim = hidden
        self.intermediate_dim = intermediate
        self.gate_up_proj = nn.Parameter(torch.randn(n_exp, 2 * intermediate, hidden))
        self.down_proj = nn.Parameter(torch.randn(n_exp, hidden, intermediate))
        self.act_fn = F.silu


class TestPackedExpertLoRA(unittest.TestCase):
    def test_base_frozen_and_loss_drops(self):
        torch.manual_seed(4)
        experts = _FakePackedExperts()
        frozen_gu = experts.gate_up_proj.detach().clone()
        wrap = LoRAPackedExperts(experts, r=2, alpha=4, device=torch.device("cpu"), dtype=torch.float32)
        n_tok = 5
        hidden = torch.randn(n_tok, 8)
        top_k_index = torch.tensor([[0, 1], [1, 2], [0, 3], [2, 3], [1, 0]])
        top_k_weights = torch.full((n_tok, 2), 0.5)
        target = torch.randn(n_tok, 8)
        opt = torch.optim.Adam(lora_parameters(wrap), lr=0.05)
        before = F.mse_loss(wrap(hidden, top_k_index, top_k_weights), target).item()
        for _ in range(40):
            opt.zero_grad()
            F.mse_loss(wrap(hidden, top_k_index, top_k_weights), target).backward()
            opt.step()
        after = F.mse_loss(wrap(hidden, top_k_index, top_k_weights), target).item()
        self.assertTrue(math.isfinite(after))
        self.assertLess(after, before)
        self.assertTrue(torch.equal(wrap.gate_up_proj, frozen_gu))
        self.assertFalse(wrap.gate_up_proj.requires_grad)
        self.assertTrue(wrap.lora_A_gate_up.requires_grad)

    def test_inject_by_class_name(self):
        class Qwen4ExpTextExperts(_FakePackedExperts):
            pass

        class MoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = Qwen4ExpTextExperts()

        moe = MoE()
        n = inject_packed_expert_lora(moe, r=2, alpha=2, device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(n, 1)
        self.assertIsInstance(moe.experts, LoRAPackedExperts)


if __name__ == "__main__":
    unittest.main()
