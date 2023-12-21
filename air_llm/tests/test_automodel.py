import sys
import unittest

#sys.path.insert(0, '../airllm')

from ..airllm.auto_model import AutoModel



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

