import math
import numbers
import re
from collections import OrderedDict
from types import MethodType

import torch
import torch.nn.functional as F

from .airllm_base import AirLLMBaseModel
from .utils import layer_tensor_sizes, load_layer_subset


def _streamed_experts_forward(module, hidden_states, top_k_index, top_k_weights):
    return module._airllm_owner._run_streamed_experts(
        module, hidden_states, top_k_index, top_k_weights)


class AirLLMDeepseekV4(AirLLMBaseModel):
    """AirLLM adapter for the native DeepSeek-V4-Flash checkpoint.

    V4's published weights use a flat DeepSeek namespace and store each routed expert separately,
    while Transformers builds Hugging Face names and stacks all experts into two giant tensors.
    The ordinary layer hooks still stream attention, routing, and shared-expert weights. This
    adapter translates those keys and replaces only the stacked expert forward with safetensors
    reads for the experts selected by the router.
    """

    _EXPERT_KEY = re.compile(
        r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
    # V4 releases each streamed layer before advancing. Keeping the allocator's blocks cached
    # avoids 43 full GC / CUDA cache purges per generated token without retaining live weights.
    clean_memory_after_layer = False

    def __init__(
        self,
        *args,
        max_vram_gb=None,
        **kwargs,
    ):
        """Create a V4 adapter within an optional per-process CUDA memory budget."""
        if max_vram_gb is not None:
            if (
                isinstance(max_vram_gb, bool)
                or not isinstance(max_vram_gb, numbers.Real)
                or not math.isfinite(max_vram_gb)
                or max_vram_gb <= 0
            ):
                raise ValueError('max_vram_gb must be a positive finite number')
        self.max_vram_gb = float(max_vram_gb) if max_vram_gb is not None else None
        self.expert_cache_size = 0
        self._expert_cache = {}
        self._layer_tensor_sizes = {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def _choose_vram_policy(
        budget_bytes,
        allocated_bytes,
        resident_bytes,
        largest_streamed_bytes,
        expert_working_bytes,
        expert_cache_unit_bytes,
        max_cache_size,
    ):
        """Choose residency and a per-layer cache size within a byte budget."""
        headroom = max(1024**3, int(budget_bytes * 0.10))
        resident_working = max(headroom, expert_working_bytes)
        keep_resident = allocated_bytes + resident_bytes + resident_working <= budget_bytes

        if keep_resident:
            fixed = allocated_bytes + resident_bytes
            working = resident_working
        else:
            fixed = allocated_bytes
            # Streaming briefly needs the largest ordinary module (usually the language head)
            # alongside routed-expert workspace and allocator/KV headroom.
            working = headroom + largest_streamed_bytes + expert_working_bytes

        required = fixed + working
        if required > budget_bytes:
            return False, 0, working, required

        cache_bytes = budget_bytes - required
        cache_size = (
            min(max_cache_size, cache_bytes // expert_cache_unit_bytes)
            if expert_cache_unit_bytes else 0
        )
        return keep_resident, int(cache_size), working, required

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
            'resident': ['model.hc_head'],
            'checkpoint': {
                'embed': 'embed',
                'layer_prefix': 'layers',
                'norm': 'norm',
                'lm_head': 'head',
                'groups': {
                    'hc_head': ['hc_head_fn', 'hc_head_base', 'hc_head_scale'],
                },
                'aliases': {
                    'model.embed_tokens': 'embed',
                    'model.norm': 'norm',
                    'lm_head': 'head',
                    'model.hc_head': 'hc_head',
                },
            },
        }

    @staticmethod
    def _translate_key(key):
        is_quantization_scale = key.endswith('.scale')
        top_level = {
            'embed.weight': 'model.embed_tokens.weight',
            'head.weight': 'lm_head.weight',
            'norm.weight': 'model.norm.weight',
            'hc_head_fn': 'model.hc_head.hc_fn',
            'hc_head_base': 'model.hc_head.hc_base',
            'hc_head_scale': 'model.hc_head.hc_scale',
        }
        if key in top_level:
            return top_level[key]
        if key.startswith('mtp.'):
            return None
        if AirLLMDeepseekV4._EXPERT_KEY.match(key):
            # Routed experts never become stacked Transformers parameters. Their native keys are
            # retained so the replacement expert forward can seek to each tensor independently.
            return None

        replacements = (
            (r'^layers\.(\d+)\.attn_norm\.', r'model.layers.\1.input_layernorm.'),
            (r'^layers\.(\d+)\.ffn_norm\.', r'model.layers.\1.post_attention_layernorm.'),
            (r'^layers\.(\d+)\.hc_attn_fn$', r'model.layers.\1.attn_hc.fn'),
            (r'^layers\.(\d+)\.hc_attn_base$', r'model.layers.\1.attn_hc.base'),
            (r'^layers\.(\d+)\.hc_attn_scale$', r'model.layers.\1.attn_hc.scale'),
            (r'^layers\.(\d+)\.hc_ffn_fn$', r'model.layers.\1.ffn_hc.fn'),
            (r'^layers\.(\d+)\.hc_ffn_base$', r'model.layers.\1.ffn_hc.base'),
            (r'^layers\.(\d+)\.hc_ffn_scale$', r'model.layers.\1.ffn_hc.scale'),
            (r'^layers\.(\d+)\.attn\.', r'model.layers.\1.self_attn.'),
            (r'^layers\.(\d+)\.ffn\.', r'model.layers.\1.mlp.'),
            (r'^(model\.layers\.\d+\.self_attn)\.attn_sink$', r'\1.sinks'),
            (r'^(model\.layers\.\d+\.self_attn)\.indexer\.compressor\.norm\.',
             r'\1.compressor.indexer.kv_norm.'),
            (r'^(model\.layers\.\d+\.self_attn)\.indexer\.compressor\.ape$',
             r'\1.compressor.indexer.position_bias'),
            (r'^(model\.layers\.\d+\.self_attn)\.indexer\.compressor\.',
             r'\1.compressor.indexer.'),
            (r'^(model\.layers\.\d+\.self_attn)\.indexer\.', r'\1.compressor.indexer.'),
            (r'^(model\.layers\.\d+\.self_attn\.compressor\.indexer)\.weights_proj\.',
             r'\1.scorer.weights_proj.'),
            (r'^(model\.layers\.\d+\.self_attn\.compressor)\.norm\.', r'\1.kv_norm.'),
            (r'^(model\.layers\.\d+\.self_attn\.compressor)\.ape$', r'\1.position_bias'),
            (r'^(model\.layers\.\d+\.self_attn)\.(.*?)\.wq_a\.', r'\1.\2.q_a_proj.'),
            (r'^(model\.layers\.\d+\.self_attn)\.(.*?)\.wq_b\.', r'\1.\2.q_b_proj.'),
            (r'^(model\.layers\.\d+\.self_attn)\.(.*?)\.wkv\.', r'\1.\2.kv_proj.'),
            (r'^(model\.layers\.\d+\.self_attn)\.(.*?)\.wgate\.', r'\1.\2.gate_proj.'),
            (r'^(model\.layers\.\d+\.self_attn)\.(.*?)\.wo_a\.', r'\1.\2.o_a_proj.'),
            (r'^(model\.layers\.\d+\.self_attn)\.(.*?)\.wo_b\.', r'\1.\2.o_b_proj.'),
            (r'^(model\.layers\.\d+\.self_attn)\.wq_a\.', r'\1.q_a_proj.'),
            (r'^(model\.layers\.\d+\.self_attn)\.wq_b\.', r'\1.q_b_proj.'),
            (r'^(model\.layers\.\d+\.self_attn)\.wkv\.', r'\1.kv_proj.'),
            (r'^(model\.layers\.\d+\.self_attn)\.wo_a\.', r'\1.o_a_proj.'),
            (r'^(model\.layers\.\d+\.self_attn)\.wo_b\.', r'\1.o_b_proj.'),
            (r'^(model\.layers\.\d+\.self_attn)\.q_norm\.', r'\1.q_a_norm.'),
            (r'^(model\.layers\.\d+\.mlp\.gate)\.bias$',
             r'\1.e_score_correction_bias'),
            (r'^(model\.layers\.\d+\.mlp\.shared_experts)\.w1\.', r'\1.gate_proj.'),
            (r'^(model\.layers\.\d+\.mlp\.shared_experts)\.w2\.', r'\1.down_proj.'),
            (r'^(model\.layers\.\d+\.mlp\.shared_experts)\.w3\.', r'\1.up_proj.'),
        )
        translated = key
        for source, target in replacements:
            translated = re.sub(source, target, translated)
        if translated == key:
            return None
        if is_quantization_scale and translated.endswith('.scale'):
            translated = translated[:-len('.scale')] + '.weight_scale_inv'
        return translated

    def _translate_state_dict(self, state_dict):
        translated = {}
        for key, value in state_dict.items():
            target = self._translate_key(key)
            if target is not None:
                translated[target] = value
        return translated

    def load_layer_to_cpu(self, layer_name):
        state_dict = super().load_layer_to_cpu(layer_name)
        return self._translate_state_dict(state_dict)

    def _setup_expert_streaming(self):
        self._expert_streaming = False
        self._expert_keys = {}
        self._non_expert_keys = {}

        layer_prefix = self.layer_names_dict['layer_prefix']
        hooked = 0
        for idx in self._streamed_indices:
            layer_name = self.layer_names[idx]
            if not layer_name.startswith(layer_prefix + '.'):
                continue
            checkpoint_name = self._checkpoint_layer_name(layer_name)
            tensor_sizes = layer_tensor_sizes(self.checkpoint_path, checkpoint_name)

            experts = {}
            others = []
            for key in tensor_sizes:
                match = self._EXPERT_KEY.match(key)
                if match is None:
                    others.append(key)
                    continue
                expert_idx = int(match.group(2))
                projection = match.group(3)
                kind = match.group(4)
                experts.setdefault(expert_idx, {}).setdefault(projection, {})[kind] = key

            if not experts:
                continue

            expert_module = self.layers[idx].mlp.experts
            expert_module._airllm_owner = self
            expert_module._airllm_layer_idx = idx
            expert_module.forward = MethodType(_streamed_experts_forward, expert_module)
            self._expert_keys[idx] = experts
            self._non_expert_keys[idx] = others
            self._layer_tensor_sizes[idx] = tensor_sizes
            hooked += len(experts)

        if hooked:
            self._expert_streaming = True
            # GenerationMixin otherwise sees Transformers' default ``grouped_mm`` setting and
            # temporarily swaps the model to ``batched_mm`` during decode. V4's expert forward is
            # replaced above and owns its FP4 dispatch, so that swap is both irrelevant and fails
            # when Transformers tries to restore grouped_mm after generation.
            self.model.config._experts_implementation_internal = 'eager'
            print(f"DeepSeek V4 selective expert streaming enabled: {hooked} experts across "
                  f"{len(self._expert_keys)} layers; only routed experts are read from disk.")

    def _configure_streaming_policy(self):
        if self.max_vram_gb is None:
            return ()
        if self.device.type != 'cuda':
            raise ValueError('max_vram_gb requires a CUDA device')

        budget_bytes = int(self.max_vram_gb * 1024**3)
        total_bytes = torch.cuda.get_device_properties(self.device).total_memory
        if budget_bytes > total_bytes:
            raise ValueError(
                f"max_vram_gb={self.max_vram_gb:g} exceeds this device's "
                f"{total_bytes / 1024**3:.2f} GiB of VRAM"
            )

        allocated_bytes = torch.cuda.memory_allocated(self.device)
        resident_bytes = 0
        largest_streamed_bytes = 0
        for idx in self._streamed_indices:
            sizes = self._layer_tensor_sizes.get(idx)
            if sizes is None:
                checkpoint_name = self._checkpoint_layer_name(self.layer_names[idx])
                sizes = layer_tensor_sizes(self.checkpoint_path, checkpoint_name)
                self._layer_tensor_sizes[idx] = sizes
            keys = self._non_expert_keys.get(idx, sizes.keys())
            module_bytes = sum(
                sizes[key] for key in keys if self._translate_key(key) is not None
            )
            resident_bytes += module_bytes
            largest_streamed_bytes = max(largest_streamed_bytes, module_bytes)

        expert_cache_unit_bytes = 0
        expert_working_bytes = 0
        max_cache_size = None
        for idx, experts in self._expert_keys.items():
            sizes = self._layer_tensor_sizes[idx]
            expert_sizes = [
                sum(
                    sizes[key]
                    for projection in parts.values()
                    for key in projection.values()
                )
                for parts in experts.values()
            ]
            largest_expert = max(expert_sizes)
            expert_cache_unit_bytes += largest_expert
            expert_working_bytes = max(
                expert_working_bytes,
                largest_expert * self.config.num_experts_per_tok,
            )
            layer_cache_limit = len(experts)
            max_cache_size = (
                layer_cache_limit
                if max_cache_size is None
                else min(max_cache_size, layer_cache_limit)
            )

        keep_resident, cache_size, working_bytes, required_bytes = self._choose_vram_policy(
            budget_bytes=budget_bytes,
            allocated_bytes=allocated_bytes,
            resident_bytes=resident_bytes,
            largest_streamed_bytes=largest_streamed_bytes,
            expert_working_bytes=expert_working_bytes,
            expert_cache_unit_bytes=expert_cache_unit_bytes,
            max_cache_size=max_cache_size or 0,
        )
        if required_bytes > budget_bytes:
            raise ValueError(
                f"max_vram_gb={self.max_vram_gb:g} is too small; this checkpoint needs an "
                f"estimated minimum of {required_bytes / 1024**3:.2f} GiB"
            )

        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(budget_bytes / total_bytes, device_index)

        self.expert_cache_size = cache_size
        mode = 'resident' if keep_resident else 'streamed'
        print(
            f"DeepSeek V4 {self.max_vram_gb:g} GiB policy: {mode} ordinary weights, "
            f"{cache_size} cached experts per layer, "
            f"{working_bytes / 1024**3:.2f} GiB working headroom."
        )
        return self._streamed_indices if keep_resident else ()

    def _load_streamed_layer(self, idx):
        keys = self._non_expert_keys.get(idx) if self._expert_streaming else None
        if keys is None:
            return self.load_layer_to_cpu(self.layer_names[idx])
        state_dict = load_layer_subset(
            self.checkpoint_path, self._checkpoint_layer_name(self.layer_names[idx]), keys)
        return self._translate_state_dict(state_dict)

    def _expert_linear(self, inputs, tensors, projection):
        weight = tensors[projection]['weight'].to(inputs.device)
        scale = tensors[projection].get('scale')
        if scale is None or weight.element_size() > 1:
            return F.linear(inputs, weight.to(inputs.dtype))

        scale = scale.to(inputs.device)
        if not hasattr(self, '_fp8_linear'):
            from transformers.integrations.finegrained_fp8 import (
                finegrained_fp8_linear,
                fp8_linear,
            )

            # The current DeepGEMM integration accepts SM90/SM100 but rejects SM120. Bypass its
            # exception-driven dispatcher there and call the same Triton fallback directly.
            is_sm12x = (
                inputs.device.type == 'cuda'
                and torch.cuda.get_device_capability(inputs.device)[0] == 12
            )
            self._fp8_linear = finegrained_fp8_linear if is_sm12x else fp8_linear

        if weight.dtype == torch.int8:
            block_size = None
        else:
            quantization_config = getattr(self.config, 'quantization_config', None)
            if isinstance(quantization_config, dict):
                configured = quantization_config.get('weight_block_size', (128, 128))
            else:
                configured = getattr(quantization_config, 'weight_block_size', (128, 128))
            block_size = tuple(configured)
        return self._fp8_linear(inputs, weight, scale, block_size=block_size)

    def _run_streamed_experts(self, module, hidden_states, top_k_index, top_k_weights):
        layer_idx = module._airllm_layer_idx
        final = torch.zeros_like(hidden_states)
        single_token = hidden_states.size(0) == 1
        if single_token:
            routes = sorted(
                (int(expert_idx), top_k_pos)
                for top_k_pos, expert_idx in enumerate(top_k_index[0].tolist())
                if int(expert_idx) in self._expert_keys[layer_idx]
            )
            loaded = [expert_idx for expert_idx, _ in routes]
        else:
            with torch.no_grad():
                hit = torch.unique(top_k_index).tolist()
            loaded = [int(expert_idx) for expert_idx in hit
                      if int(expert_idx) in self._expert_keys[layer_idx]]
        layer_cache = self._expert_cache.setdefault(layer_idx, OrderedDict())
        missing = [expert_idx for expert_idx in loaded if expert_idx not in layer_cache]

        # Opening a V4 layer shard requires parsing its very large tensor index. Fetch every
        # routed expert through one handle, then execute them individually to keep the simple
        # batch-1 kernel path and bounded memory use.
        keys = [
            key
            for expert_idx in missing
            for projection in self._expert_keys[layer_idx][expert_idx].values()
            for key in projection.values()
        ]
        raw = (
            load_layer_subset(
                self.checkpoint_path,
                self._checkpoint_layer_name(self.layer_names[layer_idx]),
                keys,
            )
            if keys else {}
        )

        for loaded_pos, expert_idx in enumerate(loaded):
            tensors = layer_cache.get(expert_idx)
            if tensors is None:
                tensors = {
                    projection: {
                        kind: raw[key].to(hidden_states.device)
                        for kind, key in parts.items()
                    }
                    for projection, parts in self._expert_keys[layer_idx][expert_idx].items()
                }
                if self.expert_cache_size:
                    layer_cache[expert_idx] = tensors
            else:
                layer_cache.move_to_end(expert_idx)

            if single_token:
                selected = hidden_states
            else:
                token_idx, top_k_pos = torch.where(top_k_index == expert_idx)
                selected = hidden_states[token_idx]
            gate = self._expert_linear(selected, tensors, 'w1')
            up = self._expert_linear(selected, tensors, 'w3')
            if module.limit is not None:
                gate = gate.clamp(max=module.limit)
                up = up.clamp(min=-module.limit, max=module.limit)
            current = self._expert_linear(module.act_fn(gate) * up, tensors, 'w2')
            if single_token:
                route_weight = top_k_weights[0, routes[loaded_pos][1]]
                final.add_((current * route_weight).to(final.dtype))
            else:
                current = current * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, current.to(final.dtype))
            del tensors

        while len(layer_cache) > self.expert_cache_size:
            layer_cache.popitem(last=False)

        module._airllm_last_experts = loaded
        return final
