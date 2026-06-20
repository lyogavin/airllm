import sys
import unittest
from unittest.mock import patch

#sys.path.insert(0, '../airllm')

from ..airllm import auto_model as auto_model_module
from ..airllm.auto_model import AutoModel


class _Config:
    def __init__(self, architectures, model_type="", text_model_type=""):
        self.architectures = architectures
        self.model_type = model_type
        if text_model_type:
            self.text_config = _Config([], model_type=text_model_type)


class TestAutoModel(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_auto_model_should_return_correct_model(self):
        mapping_dict = {
            'garage-bAInd/Platypus2-7B': 'AirLLMLlama2',
            'Qwen/Qwen-7B': 'AirLLMQWen',
            'internlm/internlm-chat-7b': 'AirLLMInternLM',
            'THUDM/chatglm3-6b-base': 'AirLLMChatGLM',
            'baichuan-inc/Baichuan2-7B-Base': 'AirLLMBaichuan',
            'mistralai/Mistral-7B-Instruct-v0.1': 'AirLLMMistral',
            'mistralai/Mixtral-8x7B-v0.1': 'AirLLMMixtral'
        }


        for k,v in mapping_dict.items():
            module, cls = AutoModel.get_module_class(k)
            self.assertEqual(cls, v, f"expecting {v}")

    def test_auto_model_should_detect_qwen3_5_architecture(self):
        qwen3_5_configs = [
            _Config(["Qwen3_5ForConditionalGeneration"], "qwen3_5"),
            _Config([], "qwen3_5"),
            _Config([], text_model_type="qwen3_5_text"),
        ]

        for config in qwen3_5_configs:
            with patch.object(auto_model_module.AutoConfig, "from_pretrained", return_value=config):
                module, cls = AutoModel.get_module_class("Qwen/Qwen3.6-27B-FP8")
                self.assertEqual(module, "airllm")
                self.assertEqual(cls, "AirLLMQWen3_5")
