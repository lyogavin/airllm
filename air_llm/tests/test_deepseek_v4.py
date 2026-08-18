"""DeepSeek V4 native-checkpoint and selective-expert streaming tests."""

import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
from transformers.core_model_loading import revert_weight_conversion


_AIRLLM_DIR = Path(__file__).resolve().parents[1] / 'airllm'

# Exercise the Linux/CUDA code path on macOS without importing the optional MLX backend.
if 'airllm' not in sys.modules:
    package = types.ModuleType('airllm')
    package.__path__ = [str(_AIRLLM_DIR)]
    sys.modules['airllm'] = package

from airllm import airllm_deepseek_v4 as deepseek_v4_module
from airllm.airllm_deepseek_v4 import AirLLMDeepseekV4
from airllm.persist import model_persister as persister_module
from airllm.persist.safetensor_model_persister import SafetensorModelPersister
from airllm.utils import layer_tensor_sizes


persister_module.model_persister = SafetensorModelPersister()


class _TokenizerlessV4(AirLLMDeepseekV4):
    def get_tokenizer(self, hf_token=None):
        return None


def _tiny_config():
    return DeepseekV4Config(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=16,
        moe_intermediate_size=16,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        mlp_layer_types=['moe'],
        layer_types=['heavily_compressed_attention'],
        sliding_window=8,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        max_position_embeddings=32,
        num_nextn_predict_layers=0,
        partial_rotary_factor=0.5,
        tie_word_embeddings=False,
        use_cache=False,
    )


def test_native_key_translation():
    translate = AirLLMDeepseekV4._translate_key
    assert translate('embed.weight') == 'model.embed_tokens.weight'
    assert translate('layers.3.attn.wq_a.weight') == 'model.layers.3.self_attn.q_a_proj.weight'
    assert translate('layers.3.attn.compressor.indexer.weights_proj.weight') == (
        'model.layers.3.self_attn.compressor.indexer.scorer.weights_proj.weight')
    assert translate('layers.3.ffn.gate.bias') == (
        'model.layers.3.mlp.gate.e_score_correction_bias')
    assert translate('layers.3.ffn.shared_experts.w1.scale') == (
        'model.layers.3.mlp.shared_experts.gate_proj.weight_scale_inv')
    assert translate('layers.3.ffn.experts.17.w1.weight') is None


def test_packed_fp4_expert_goes_to_native_kernel_without_cpu_dequantization(monkeypatch):
    calls = []

    def fake_fp8_linear(inputs, weight, scale, block_size=None):
        calls.append((weight.dtype, scale.dtype, block_size))
        return torch.zeros(inputs.shape[0], weight.shape[0], dtype=inputs.dtype)

    monkeypatch.setattr(
        'transformers.integrations.finegrained_fp8.fp8_linear', fake_fp8_linear)
    adapter = AirLLMDeepseekV4.__new__(AirLLMDeepseekV4)
    adapter.config = SimpleNamespace(
        quantization_config={'weight_block_size': [128, 128]})
    tensors = {
        'w1': {
            'weight': torch.zeros(3, 2, dtype=torch.int8),
            'scale': torch.ones(3, 1, dtype=torch.uint8),
        },
    }

    output = adapter._expert_linear(torch.ones(1, 4), tensors, 'w1')

    assert output.shape == (1, 3)
    assert calls == [(torch.int8, torch.uint8, None)]


def test_layer_tensor_sizes_reads_metadata_without_materializing_weights():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        save_file(
            {
                'bf16': torch.zeros(3, 5, dtype=torch.bfloat16),
                'int8': torch.zeros(7, dtype=torch.int8),
            },
            str(root / 'layer.safetensors'),
        )

        assert layer_tensor_sizes(root, 'layer') == {'bf16': 30, 'int8': 7}


def test_max_vram_policy_prefers_residency_then_spends_remainder_on_cache():
    gib = 1024**3
    resident, cache_size, working, required = AirLLMDeepseekV4._choose_vram_policy(
        budget_bytes=16 * gib,
        allocated_bytes=gib // 2,
        resident_bytes=8 * gib,
        largest_streamed_bytes=gib,
        expert_working_bytes=gib // 10,
        expert_cache_unit_bytes=gib // 2,
        max_cache_size=256,
    )

    assert resident is True
    assert cache_size == 11
    assert working == int(1.6 * gib)
    assert required < 16 * gib


def test_max_vram_policy_streams_when_residency_does_not_fit():
    gib = 1024**3
    resident, cache_size, working, required = AirLLMDeepseekV4._choose_vram_policy(
        budget_bytes=4 * gib,
        allocated_bytes=gib // 2,
        resident_bytes=8 * gib,
        largest_streamed_bytes=gib,
        expert_working_bytes=gib // 10,
        expert_cache_unit_bytes=gib // 2,
        max_cache_size=256,
    )

    assert resident is False
    assert cache_size == 2
    assert working == 2 * gib + gib // 10
    assert required < 4 * gib


def test_tiny_native_checkpoint_matches_transformers_and_batches_routed_expert_read(monkeypatch):
    torch.manual_seed(7)
    config = _tiny_config()
    reference = DeepseekV4ForCausalLM(config).eval()
    input_ids = torch.tensor([[1]])

    with torch.no_grad():
        expected = reference(input_ids, use_cache=False).logits

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw_state = revert_weight_conversion(reference.model, reference.model.state_dict())
        raw_state['head.weight'] = reference.lm_head.weight.detach().clone()
        save_file(
            {key: value.detach().contiguous() for key, value in raw_state.items()},
            str(root / 'model.safetensors'),
        )
        config.save_pretrained(root)

        model = _TokenizerlessV4(
            root,
            device='cpu',
            dtype=torch.float32,
            prefetching=False,
        )
        expert_reads = []
        original_load_subset = deepseek_v4_module.load_layer_subset

        def counted_load_subset(local_path, layer_name, keys):
            if any('.experts.' in key for key in keys):
                expert_reads.append(tuple(keys))
            return original_load_subset(local_path, layer_name, keys)

        monkeypatch.setattr(deepseek_v4_module, 'load_layer_subset', counted_load_subset)
        with torch.no_grad():
            actual = model.model(input_ids, use_cache=False).logits
            cached = model.model(input_ids, use_cache=False).logits

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(cached, expected, rtol=0, atol=0)
        loaded = model.model.model.layers[0].mlp.experts._airllm_last_experts
        assert len(loaded) == config.num_experts_per_tok
        assert len(loaded) < config.n_routed_experts
        assert len(expert_reads) == 2
        assert model.model.config._experts_implementation == 'eager'

        split = root / 'splitted_model'
        assert (split / 'hc_head.safetensors').is_file()
        with safe_open(str(split / 'hc_head.safetensors'), framework='pt') as shard:
            assert set(shard.keys()) == {'hc_head_fn', 'hc_head_base', 'hc_head_scale'}
