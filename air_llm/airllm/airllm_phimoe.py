"""PhiMoE adapter exposing checkpoint experts as independently streamable modules."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

from .airllm_base import AirLLMBaseModel


class PhiMoEStreamedExpert(nn.Module):
    """Checkpoint-compatible PhiMoE gated MLP allocated entirely on ``meta``."""

    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        self.w1 = nn.Linear(hidden, intermediate, bias=False, device='meta')
        self.w2 = nn.Linear(intermediate, hidden, bias=False, device='meta')
        self.w3 = nn.Linear(hidden, intermediate, bias=False, device='meta')
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.w2(self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states))


class PhiMoEStreamedExperts(nn.Module):
    """Indexed expert collection with Transformers' packed-bank calling convention.

    Numeric child names intentionally produce state-dict paths such as
    ``block_sparse_moe.experts.3.w1.weight``, matching the published checkpoint and AirLLM's
    existing per-expert discovery logic.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self._airllm_slot_runtime = None
        self._airllm_layer_idx = None
        for expert_idx in range(self.num_experts):
            self.add_module(str(expert_idx), PhiMoEStreamedExpert(config))

    def __len__(self):
        return self.num_experts

    def __getitem__(self, expert_idx):
        return self._modules[str(int(expert_idx))]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            ).permute(2, 1, 0)
            # One device-to-host synchronization per MoE layer is required because Python module
            # hooks decide which individual expert weights to materialize.
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()

        if self._airllm_slot_runtime is None:
            for expert_idx in expert_hit:
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                current_hidden_states = self[expert_idx](current_state)
                current_hidden_states = current_hidden_states * top_k_weights[
                    token_idx, top_k_pos, None
                ]
                final_hidden_states.index_add_(
                    0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
                )
            return final_hidden_states

        # The native PhiMoE implementation flattens batch and sequence before expert dispatch.
        # The validation workload is batch one, so one flattened token is decode; larger routed
        # sets are prefill and use transient scratch slots to avoid polluting the global cache.
        admit = hidden_states.ndim == 2 and hidden_states.shape[0] == 1

        def execute_one(expert_idx, weights):
            w1, w2, w3 = weights
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = F.linear(current_state, w1)
            current_hidden_states = self[expert_idx].act_fn(current_hidden_states)
            current_hidden_states = current_hidden_states * F.linear(current_state, w3)
            current_hidden_states = F.linear(current_hidden_states, w2)
            current_hidden_states = current_hidden_states * top_k_weights[
                token_idx, top_k_pos, None
            ]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        self._airllm_slot_runtime.execute(
            self._airllm_layer_idx,
            expert_hit,
            admit=admit,
            callback=execute_one,
        )
        return final_hidden_states


class AirLLMPhiMoE(AirLLMBaseModel):
    """AirLLM adapter for ``PhiMoEForCausalLM`` checkpoints.

    Current Transformers packs every layer's experts into two bank parameters. AirLLM instead
    needs the checkpoint's original per-expert module layout so each routed expert can be loaded
    independently. Attention, routing, KV caching, and generation remain owned by Transformers.
    """

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
            'expert_prefix': 'block_sparse_moe.experts',
        }

    def supports_expert_slot_backend(self):
        return True

    def init_model(self):
        super().init_model()
        # The replacement below owns expert dispatch, so Transformers must not swap its packed
        # expert bank between grouped_mm and batched_mm around decoding. Besides being inapplicable
        # to independently hooked modules, that automatic restore rejects AirLLM's runtime class.
        self.config._experts_implementation = 'eager'
        layers = self.model.model.layers
        for layer in layers:
            # Transformers 5 renamed the checkpoint's ``block_sparse_moe`` module to ``mlp`` and
            # its ``gate`` child to ``router``.  AirLLM deliberately streams the original shard
            # keys, so retain aliases for those published names while the native forward keeps
            # calling ``layer.mlp``.  Older Transformers releases already expose the old names.
            sparse_moe = getattr(layer, 'block_sparse_moe', None)
            if sparse_moe is None:
                sparse_moe = getattr(layer, 'mlp', None)
                if sparse_moe is None:
                    raise RuntimeError(
                        f"Unsupported PhiMoE decoder layout: {type(layer).__name__}"
                    )
                layer.block_sparse_moe = sparse_moe
            if not hasattr(sparse_moe, 'gate') and hasattr(sparse_moe, 'router'):
                sparse_moe.gate = sparse_moe.router

            packed = sparse_moe.experts
            if self._is_checkpoint_indexed(packed):
                continue
            if not (hasattr(packed, 'gate_up_proj') and hasattr(packed, 'down_proj')):
                raise RuntimeError(
                    f"Unsupported PhiMoE expert representation: {type(packed).__name__}"
                )
            replacement = PhiMoEStreamedExperts(self.config)
            replacement.train(packed.training)
            sparse_moe.experts = replacement

    @staticmethod
    def _is_checkpoint_indexed(experts):
        try:
            first = experts[0]
        except (KeyError, TypeError, IndexError):
            return False
        return all(hasattr(first, name) for name in ('w1', 'w2', 'w3'))
