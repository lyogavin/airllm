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
            # These architectures have dedicated subclasses in ARCH_OVERRIDES
            'Qwen/Qwen-7B': 'AirLLMQWen',
            'THUDM/chatglm3-6b-base': 'AirLLMChatGLM',
            'baichuan-inc/Baichuan2-7B-Base': 'AirLLMBaichuan',
            'internlm/internlm-chat-7b': 'AirLLMInternLM',
            # Standard architectures now use the generic streaming model
            'garage-bAInd/Platypus2-7B': 'AirLLMBaseModel',
            'mistralai/Mistral-7B-Instruct-v0.1': 'AirLLMBaseModel',
            'mistralai/Mixtral-8x7B-v0.1': 'AirLLMBaseModel',
        }


        for k,v in mapping_dict.items():
            module, cls = AutoModel.get_module_class(k)
            self.assertEqual(cls, v, f"expecting {v} for {k}, got {cls}")


