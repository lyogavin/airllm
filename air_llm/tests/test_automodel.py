import unittest
from types import SimpleNamespace
from unittest.mock import patch

from airllm.auto_model import AutoConfig, AutoModel



class TestAutoModel(unittest.TestCase):
    def test_auto_model_should_return_correct_model(self):
        # Config fixtures keep discovery offline and follow the current generic
        # streaming path for models with the standard decoder layout.
        mapping_dict = {
            'garage-bAInd/Platypus2-7B': ('LlamaForCausalLM', 'AirLLMBaseModel'),
            'Qwen/Qwen-7B': ('QWenLMHeadModel', 'AirLLMQWen'),
            'internlm/internlm-chat-7b': ('InternLMForCausalLM', 'AirLLMInternLM'),
            'THUDM/chatglm3-6b-base': ('ChatGLMModel', 'AirLLMChatGLM'),
            'baichuan-inc/Baichuan2-7B-Base': ('BaichuanForCausalLM', 'AirLLMBaichuan'),
            'mistralai/Mistral-7B-Instruct-v0.1': ('MistralForCausalLM', 'AirLLMBaseModel'),
            'mistralai/Mixtral-8x7B-v0.1': ('MixtralForCausalLM', 'AirLLMBaseModel'),
        }

        for model_id, (architecture, expected_class) in mapping_dict.items():
            with self.subTest(model_id=model_id), patch.object(
                AutoConfig, 'from_pretrained',
                return_value=SimpleNamespace(architectures=[architecture]),
            ) as load_config:
                module, cls = AutoModel.get_module_class(model_id)
                self.assertEqual(module, 'airllm')
                self.assertEqual(cls, expected_class)
                load_config.assert_called_once_with(model_id, trust_remote_code=True)

