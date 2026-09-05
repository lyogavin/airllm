"""Minimal LoRA wrapper that keeps the original ``weight`` Parameter in place.

PEFT is not used: it expects materialised base weights and will try to move the
whole model. AirLLM training swaps frozen ``weight`` tensors from disk onto a
``meta`` module; the adapter matrices stay on the GPU for the whole run.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Flash-Next extra Linears: hyper-connections, PLE, shared-expert gate.
FLASH_NEXT_LORA_TARGETS = DEFAULT_LORA_TARGETS + (
    "shared_expert_gate",
    "input_mix_weight_down",
    "input_mix_weight_up",
    "block_inject_weight",
    "key_proj",
    "value_proj",
)


class LoRALinear(nn.Module):
    """``y = x W^T + (x A^T) B^T * (alpha / r)``. Base ``W`` is frozen."""

    def __init__(self, linear, r=16, alpha=32, dropout=0.0, device=None, dtype=None):
        super().__init__()
        if r < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {r}")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Keep the same Parameter object so AirLLM can still load shard keys
        # named ``<module>.weight`` / ``<module>.bias``.
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        if device is None:
            device = self.weight.device if self.weight.device.type != "meta" else torch.device("cpu")
        if dtype is None:
            dtype = self.weight.dtype if self.weight.device.type != "meta" else torch.float32

        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.lora_dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        z = self.lora_dropout(x)
        a = self.lora_A.to(dtype=z.dtype)
        b = self.lora_B.to(dtype=z.dtype)
        return base + (z @ a.t() @ b.t()) * self.scaling


def inject_lora(module, target_modules=DEFAULT_LORA_TARGETS, r=16, alpha=32,
                dropout=0.0, device=None, dtype=None):
    """Replace matching ``nn.Linear`` children with ``LoRALinear``. Returns count."""
    targets = set(target_modules)
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(
                child, r=r, alpha=alpha, dropout=dropout, device=device, dtype=dtype))
            n += 1
        else:
            n += inject_lora(child, targets, r, alpha, dropout, device, dtype)
    return n


class LoRAPackedExperts(nn.Module):
    """LoRA on Qwen4Exp's 3D packed expert tensors (not ``nn.Linear``).

    Base ``gate_up_proj`` is ``[E, 2I, H]`` and ``down_proj`` is ``[E, H, I]``.
    The original forward only touches the routed experts, so the extra matmuls
    stay on those rows too.
    """

    def __init__(self, experts, r=16, alpha=32, device=None, dtype=None):
        super().__init__()
        self.num_experts = experts.num_experts
        self.hidden_dim = experts.hidden_dim
        self.intermediate_dim = experts.intermediate_dim
        self.act_fn = experts.act_fn
        self.r = r
        self.scaling = alpha / r
        self.gate_up_proj = experts.gate_up_proj
        self.down_proj = experts.down_proj
        self.gate_up_proj.requires_grad_(False)
        self.down_proj.requires_grad_(False)

        if device is None:
            device = torch.device("cpu") if self.gate_up_proj.device.type == "meta" else self.gate_up_proj.device
        if dtype is None:
            dtype = torch.bfloat16 if self.gate_up_proj.device.type == "meta" else self.gate_up_proj.dtype

        self.lora_A_gate_up = nn.Parameter(
            torch.empty(self.num_experts, r, self.hidden_dim, device=device, dtype=dtype))
        self.lora_B_gate_up = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, r, device=device, dtype=dtype))
        self.lora_A_down = nn.Parameter(
            torch.empty(self.num_experts, r, self.intermediate_dim, device=device, dtype=dtype))
        self.lora_B_down = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A_gate_up, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_down, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_gate_up)
        nn.init.zeros_(self.lora_B_down)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            e = int(expert_idx)
            gate_up = F.linear(current_state, self.gate_up_proj[e])
            a_gu = self.lora_A_gate_up[e].to(dtype=current_state.dtype)
            b_gu = self.lora_B_gate_up[e].to(dtype=current_state.dtype)
            gate_up = gate_up + (current_state @ a_gu.t() @ b_gu.t()) * self.scaling
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            down = F.linear(current_hidden_states, self.down_proj[e])
            a_d = self.lora_A_down[e].to(dtype=current_hidden_states.dtype)
            b_d = self.lora_B_down[e].to(dtype=current_hidden_states.dtype)
            down = down + (current_hidden_states @ a_d.t() @ b_d.t()) * self.scaling
            current_hidden_states = down * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


def inject_packed_expert_lora(module, r=16, alpha=32, device=None, dtype=None):
    """Wrap ``Qwen4ExpTextExperts`` modules. Returns count."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRAPackedExperts):
            continue
        if type(child).__name__ == "Qwen4ExpTextExperts":
            setattr(module, name, LoRAPackedExperts(
                child, r=r, alpha=alpha, device=device, dtype=dtype))
            n += 1
        else:
            n += inject_packed_expert_lora(child, r, alpha, device, dtype)
    return n


def lora_parameters(module):
    return [p for n, p in module.named_parameters() if "lora_" in n]


def lora_state_dict(module):
    return {n: p.detach().cpu() for n, p in module.named_parameters() if "lora_" in n}


def load_lora_state_dict(module, state, strict=True):
    missing = []
    owned = dict(module.named_parameters())
    for name, value in state.items():
        param = owned.get(name)
        if param is None:
            missing.append(name)
            continue
        param.data.copy_(value.to(device=param.device, dtype=param.dtype))
    if strict and missing:
        raise KeyError(f"LoRA keys not found in module: {missing[:8]}")
    return missing
