from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ..airllm.auto_model import AutoModel



class TestAutoModel(unittest.TestCase):
    def test_auto_model_should_return_correct_model(self):
        # Standard Transformers layouts intentionally use the generic streamer since 8f62eec;
        # only non-standard layouts need an architecture-specific adapter. Mocking config also
        # keeps this unit test deterministic and independent of Hub/network changes.
        mapping_dict = {
            'LlamaForCausalLM': 'AirLLMBaseModel',
            'QWenLMHeadModel': 'AirLLMQWen',
            'InternLMForCausalLM': 'AirLLMInternLM',
            'ChatGLMForConditionalGeneration': 'AirLLMChatGLM',
            'BaichuanForCausalLM': 'AirLLMBaichuan',
            'MistralForCausalLM': 'AirLLMBaseModel',
            'MixtralForCausalLM': 'AirLLMBaseModel',
        }

        for architecture, expected in mapping_dict.items():
            with self.subTest(architecture=architecture), patch(
                "airllm.auto_model.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(architectures=[architecture]),
            ):
                module, cls = AutoModel.get_module_class("unused")
            self.assertEqual(module, "airllm")
            self.assertEqual(cls, expected)
