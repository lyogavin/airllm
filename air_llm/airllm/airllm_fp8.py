"""AirLLM-native FP8 modules that preserve Hugging Face checkpoint names.

Transformers' fine-grained FP8 loader may fuse a checkpoint's per-expert
parameters into large 3-D tensors. AirLLM streams the original state dict a
layer at a time and therefore cannot use that conversion pipeline. These
modules keep the checkpoint paths unchanged and dequantize only the projection
currently being executed. The fallback uses ordinary PyTorch operations and is
intended for correctness on ROCm/Windows where the optional CUDA/Triton FP8
kernels are unavailable.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def checkpoint_weight_names(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint_path / index_name
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                return set(json.load(f).get("weight_map", {}))

    try:
        from safetensors import safe_open

        with safe_open(str(checkpoint_path / "model.safetensors"), framework="pt") as f:
            return set(f.keys())
    except FileNotFoundError:
        pass

    torch_path = checkpoint_path / "pytorch_model.bin"
    if torch_path.exists():
        return set(torch.load(torch_path, map_location="cpu").keys())
    return set()


def _quantization_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


class AirLLMFP8Linear(nn.Module):
    """Block-scaled FP8 linear that dequantizes one projection at a time."""

    def __init__(self, in_features, out_features, block_size, bias=False, device="meta"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = tuple(block_size)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        )
        scale_shape = (
            math.ceil(out_features / self.block_size[0]),
            math.ceil(in_features / self.block_size[1]),
        )
        self.weight_scale_inv = nn.Parameter(
            torch.empty(scale_shape, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)

    def _dequantized_weight(self, dtype):
        out_block, in_block = self.block_size
        scales = self.weight_scale_inv.to(dtype=dtype)
        scales = scales.repeat_interleave(out_block, dim=0).repeat_interleave(in_block, dim=1)
        scales = scales[: self.out_features, : self.in_features]
        return self.weight.to(dtype=dtype) * scales

    def forward(self, inputs):
        return F.linear(inputs, self._dequantized_weight(inputs.dtype), self.bias)


class _AirLLMFP8Expert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, block_size, device):
        super().__init__()
        self.gate_proj = AirLLMFP8Linear(hidden_size, intermediate_size, block_size, device=device)
        self.up_proj = AirLLMFP8Linear(hidden_size, intermediate_size, block_size, device=device)
        self.down_proj = AirLLMFP8Linear(intermediate_size, hidden_size, block_size, device=device)


class AirLLMFP8Experts(nn.ModuleList):
    """Per-expert layout matching GLM-5.2's published FP8 state dict."""

    def __init__(self, original, block_size):
        num_experts = original.num_experts
        hidden_dim = original.hidden_dim
        intermediate_dim = original.intermediate_dim
        device = original.gate_up_proj.device
        super().__init__(
            [
                _AirLLMFP8Expert(hidden_dim, intermediate_dim, block_size, device)
                for _ in range(num_experts)
            ]
        )
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = original.act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor)
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert = self[expert_idx]
            current_state = self.act_fn(expert.gate_proj(current_state)) * expert.up_proj(current_state)
            current_state = expert.down_proj(current_state)
            current_state = current_state * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_state.to(final_hidden_states.dtype))

        return final_hidden_states.to(hidden_states.dtype)


def _set_module(model, module_name, replacement):
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def replace_with_airllm_fp8(model, checkpoint_path, quantization_config):
    """Replace only modules whose checkpoint weight has an FP8 scale tensor."""
    weight_names = checkpoint_weight_names(checkpoint_path)
    block_size = _quantization_value(quantization_config, "weight_block_size", (128, 128))
    if not weight_names:
        raise FileNotFoundError(f"Cannot inspect checkpoint weight names under {checkpoint_path}")

    replaced_experts = 0
    for module_name, module in list(model.named_modules()):
        expert_scale = f"{module_name}.0.gate_proj.weight_scale_inv"
        if hasattr(module, "gate_up_proj") and expert_scale in weight_names:
            _set_module(model, module_name, AirLLMFP8Experts(module, block_size))
            replaced_experts += 1

    replaced_linears = 0
    for module_name, module in list(model.named_modules()):
        if type(module) is not nn.Linear:
            continue
        if f"{module_name}.weight_scale_inv" not in weight_names:
            continue
        replacement = AirLLMFP8Linear(
            module.in_features,
            module.out_features,
            block_size,
            bias=module.bias is not None,
            device=module.weight.device,
        )
        _set_module(model, module_name, replacement)
        replaced_linears += 1

    if replaced_experts == 0 and replaced_linears == 0:
        raise ValueError("FP8 checkpoint metadata did not match any model modules.")
    return replaced_experts, replaced_linears
