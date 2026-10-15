import os
import unittest

import airllm.auto_model as auto_model_mod
from airllm.auto_model import AutoModel


class _FakeConfig:
    def __init__(self, architectures):
        self.architectures = architectures


class TestAutoModelRouting(unittest.TestCase):
    """
    AutoModel routing, tested offline by faking the config lookup.

    Since the v3.0 rewrite only architectures with a non-standard module layout get a dedicated
    AirLLM subclass (ARCH_OVERRIDES); every other *ForCausalLM streams through the generic
    AirLLMBaseModel, which is what lets newly released models work without code changes.
    """

    def setUp(self):
        self._orig = auto_model_mod.AutoConfig.from_pretrained

    def tearDown(self):
        auto_model_mod.AutoConfig.from_pretrained = self._orig

    def _route(self, architecture):
        def fake(path, trust_remote_code=None, **kwargs):
            return _FakeConfig([architecture] if architecture else [])

        auto_model_mod.AutoConfig.from_pretrained = staticmethod(fake)
        module, cls = AutoModel.get_module_class('some/repo')
        self.assertEqual(module, 'airllm')
        return cls

    def test_custom_architectures_get_dedicated_classes(self):
        mapping = {
            'ChatGLMModel': 'AirLLMChatGLM',
            'ChatGLMForConditionalGeneration': 'AirLLMChatGLM',
            'QWenLMHeadModel': 'AirLLMQWen',
            'BaichuanForCausalLM': 'AirLLMBaichuan',
            'BaiChuanForCausalLM': 'AirLLMBaichuan',
            'InternLMForCausalLM': 'AirLLMInternLM',
        }
        for arch, expected in mapping.items():
            self.assertEqual(self._route(arch), expected, f"{arch} should route to {expected}")

    def test_standard_architectures_use_generic_streaming_model(self):
        # These used to have dedicated subclasses (AirLLMLlama2/Mistral/Mixtral) but now go through
        # the generic path, which is why the old assertions in this test were stale.
        for arch in ('LlamaForCausalLM', 'MistralForCausalLM', 'MixtralForCausalLM',
                     'Qwen2ForCausalLM', 'Qwen3ForCausalLM', 'Phi3ForCausalLM',
                     'Gemma2ForCausalLM', 'DeepseekV3ForCausalLM'):
            self.assertEqual(self._route(arch), 'AirLLMBaseModel',
                             f"{arch} should use the generic streaming model")

    def test_unknown_or_missing_architecture_falls_back_to_generic(self):
        self.assertEqual(self._route('SomeBrandNewForCausalLM'), 'AirLLMBaseModel')
        self.assertEqual(self._route(None), 'AirLLMBaseModel')


@unittest.skipUnless(os.environ.get('AIRLLM_TEST_NETWORK'),
                     "set AIRLLM_TEST_NETWORK=1 to run tests that download configs from Hugging Face")
class TestAutoModelRoutingAgainstHub(unittest.TestCase):
    """Same routing, resolved against the real Hugging Face configs. Needs network access."""

    def test_real_repo_ids_route_as_expected(self):
        mapping_dict = {
            'garage-bAInd/Platypus2-7B': 'AirLLMBaseModel',
            'mistralai/Mistral-7B-Instruct-v0.1': 'AirLLMBaseModel',
            'mistralai/Mixtral-8x7B-v0.1': 'AirLLMBaseModel',
            'Qwen/Qwen-7B': 'AirLLMQWen',
            'internlm/internlm-chat-7b': 'AirLLMInternLM',
            'THUDM/chatglm3-6b-base': 'AirLLMChatGLM',
            'baichuan-inc/Baichuan2-7B-Base': 'AirLLMBaichuan',
        }
        for repo_id, expected in mapping_dict.items():
            module, cls = AutoModel.get_module_class(repo_id)
            self.assertEqual(cls, expected, f"expecting {expected} for {repo_id}")


if __name__ == '__main__':
    unittest.main()
