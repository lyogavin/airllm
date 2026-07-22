import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ..airllm.airllm_base import AirLLMBaseModel


class TestStreamingCpuIntegration(unittest.TestCase):
    @staticmethod
    def _config():
        return LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )

    def test_tiny_llama_matches_full_model(self):
        torch.manual_seed(7)
        config = self._config()
        reference = LlamaForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.inference_mode():
            expected = reference(input_ids=input_ids, use_cache=False).logits

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            reference.save_pretrained(model_path)
            with patch.object(AirLLMBaseModel, "get_tokenizer", return_value=None):
                streamed = AirLLMBaseModel(
                    model_path,
                    device="cpu",
                    prefetching=False,
                )
            with torch.inference_mode():
                actual = streamed(input_ids=input_ids, use_cache=False).logits

        torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA/ROCm device")
    def test_tiny_llama_streams_on_gpu(self):
        torch.manual_seed(11)
        reference = LlamaForCausalLM(self._config()).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            reference.save_pretrained(model_path)
            reference = reference.to("cuda")
            with torch.inference_mode():
                expected = reference(input_ids=input_ids.to("cuda"), use_cache=False).logits.cpu()

            with patch.object(AirLLMBaseModel, "get_tokenizer", return_value=None):
                streamed = AirLLMBaseModel(
                    model_path,
                    device="cuda:0",
                    prefetching=False,
                )
            with torch.inference_mode():
                actual = streamed(input_ids=input_ids.to("cuda"), use_cache=False).logits.cpu()

        torch.testing.assert_close(actual, expected)
