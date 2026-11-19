"""Offline CPU end-to-end parity for checkpoint splitting and streamed generation."""
import gc
import importlib.abc
from pathlib import Path
import sys
import tempfile
import unittest


@unittest.skipIf(sys.platform == 'darwin',
                 'macOS AutoModel uses MLX instead of the PyTorch CPU backend')
class StreamingCPUParityTests(unittest.TestCase):
    def test_streaming_without_prefetch(self):
        self._check_streaming_parity(prefetching=False)

    def test_streaming_with_prefetch(self):
        self._check_streaming_parity(prefetching=True)

    def _check_streaming_parity(self, prefetching):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        attempts = []

        class BlockBitsandbytes(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'bitsandbytes' or fullname.startswith('bitsandbytes.'):
                    attempts.append(fullname)
                    raise ImportError('uncompressed inference must not import bitsandbytes')

        blocker = BlockBitsandbytes()
        sys.meta_path.insert(0, blocker)
        self.addCleanup(sys.meta_path.remove, blocker)
        from airllm import AutoModel

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            torch.manual_seed(318)
            config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                                 num_hidden_layers=2, num_attention_heads=4,
                                 num_key_value_heads=2, max_position_embeddings=64,
                                 pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                 tie_word_embeddings=False)
            config._attn_implementation = 'sdpa'
            reference = LlamaForCausalLM(config).eval()
            reference.save_pretrained(model_path)
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(WordLevel({f't{i}': i for i in range(32)}, unk_token='t3')),
                pad_token='t0', bos_token='t1', eos_token='t2', unk_token='t3')
            tokenizer.save_pretrained(model_path)
            ids = torch.tensor([[1, 7, 11, 21, 9]])
            mask = torch.ones_like(ids)
            with torch.inference_mode():
                expected_logits = reference(ids, attention_mask=mask, use_cache=False).logits
                expected_tokens = reference.generate(ids, attention_mask=mask, max_new_tokens=4,
                                                     min_new_tokens=4, do_sample=False)
            del reference
            gc.collect()
            model = AutoModel.from_pretrained(str(model_path), device='cpu', dtype=torch.float32,
                                             compression=None, prefetching=prefetching)
            try:
                self.assertEqual(model.running_device, 'cpu')
                self.assertEqual(model.prefetching, prefetching)
                with torch.inference_mode():
                    actual_logits = model(ids, attention_mask=mask, use_cache=False).logits
                    self.assertTrue(torch.isfinite(actual_logits).all())
                    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)
                    for _ in range(2):
                        actual_tokens = model.generate(ids, attention_mask=mask, max_new_tokens=4,
                                                       min_new_tokens=4, do_sample=False)
                        self.assertTrue(torch.equal(actual_tokens, expected_tokens))
                        for layer in model.layers:
                            self.assertTrue(all(p.device.type == 'meta' for p in layer.parameters()))
                self.assertEqual(attempts, [], 'uncompressed inference attempted to import bitsandbytes')
            finally:
                if model._executor is not None:
                    model._executor.shutdown(wait=True)
                del model
                gc.collect()


if __name__ == '__main__':
    unittest.main()
