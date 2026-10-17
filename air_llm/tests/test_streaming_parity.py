"""Offline streaming parity tests for ROCm and CUDA. No model downloads required."""
import gc
import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from airllm import AutoModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA or ROCm GPU')


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('prefetching', [False, True])
def test_streaming_matches_resident_model(tmp_path, dtype, prefetching):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('device does not support bfloat16')
    torch.manual_seed(318)
    config = LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=128,
                         pad_token_id=0, bos_token_id=1, eos_token_id=2,
                         tie_word_embeddings=False)
    config._attn_implementation = 'sdpa'
    reference = LlamaForCausalLM(config).to(dtype=dtype).eval()
    reference.save_pretrained(tmp_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f't{i}': i for i in range(128)}, unk_token='t3')),
        pad_token='t0', bos_token='t1', eos_token='t2', unk_token='t3')
    tokenizer.save_pretrained(tmp_path)
    reference = reference.to('cuda:0')
    ids = torch.tensor([[1, 7, 11, 21, 9]], device='cuda:0')
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        expected_logits = reference(ids, attention_mask=mask, use_cache=False).logits.cpu()
        expected_tokens = reference.generate(ids, attention_mask=mask, max_new_tokens=4,
                                             min_new_tokens=4, do_sample=False).cpu()
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModel.from_pretrained(str(tmp_path), device='cuda:0', dtype=dtype,
                                     compression=None, prefetching=prefetching)
    try:
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            actual_logits = model(ids, attention_mask=mask, use_cache=False).logits.cpu()
            assert torch.isfinite(actual_logits).all()
            torch.testing.assert_close(actual_logits, expected_logits, rtol=0.01, atol=0.01)
            allocations = []
            for _ in range(2):
                actual_tokens = model.generate(ids, attention_mask=mask, max_new_tokens=4,
                                               min_new_tokens=4, do_sample=False).cpu()
                assert torch.equal(actual_tokens, expected_tokens)
                for layer in model.layers:
                    assert all(p.device.type == 'meta' for p in layer.parameters())
                torch.cuda.synchronize()
                allocations.append(torch.cuda.memory_allocated())
            assert allocations[1] <= allocations[0] + 1024 * 1024
        print(json.dumps({'torch': torch.__version__, 'hip': torch.version.hip,
                          'gpu': torch.cuda.get_device_name(0), 'dtype': str(dtype),
                          'prefetching': prefetching,
                          'max_abs_logit_error': (actual_logits.float() - expected_logits.float()).abs().max().item(),
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                          'post_generation_allocated_bytes': allocations}))
    finally:
        if model._executor is not None:
            model._executor.shutdown(wait=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
