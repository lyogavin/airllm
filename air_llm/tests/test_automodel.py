import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ..airllm.auto_model import AutoModel


class TestAutoModel(unittest.TestCase):
    @patch("airllm.auto_model.AutoConfig.from_pretrained")
    def test_custom_architectures_use_dedicated_streaming_layouts(self, from_pretrained):
        cases = {
            "ChatGLMForConditionalGeneration": "AirLLMChatGLM",
            "QWenLMHeadModel": "AirLLMQWen",
            "BaichuanForCausalLM": "AirLLMBaichuan",
            "InternLMForCausalLM": "AirLLMInternLM",
            "KimiK25ForConditionalGeneration": "AirLLMKimiK25",
            "GlmMoeDsaForCausalLM": "AirLLMGlmMoeDsa",
        }
        for architecture, expected in cases.items():
            with self.subTest(architecture=architecture):
                from_pretrained.return_value = SimpleNamespace(architectures=[architecture])
                module, class_name = AutoModel.get_module_class("local-model")
                self.assertEqual(module, "airllm")
                self.assertEqual(class_name, expected)

    @patch("airllm.auto_model.AutoConfig.from_pretrained")
    def test_standard_and_new_architectures_use_generic_streaming(self, from_pretrained):
        for architecture in ("LlamaForCausalLM", "UnknownForCausalLM"):
            with self.subTest(architecture=architecture):
                from_pretrained.return_value = SimpleNamespace(architectures=[architecture])
                self.assertEqual(
                    AutoModel.get_module_class("local-model"),
                    ("airllm", "AirLLMBaseModel"),
                )

    @patch("airllm.auto_model.AutoConfig.from_pretrained")
    def test_config_probe_uses_requested_cache(self, from_pretrained):
        from_pretrained.return_value = SimpleNamespace(architectures=["LlamaForCausalLM"])
        AutoModel.get_module_class("repo/model", cache_dir="D:/model-cache", hf_token="token")
        from_pretrained.assert_called_once_with(
            "repo/model",
            trust_remote_code=True,
            cache_dir="D:/model-cache",
            token="token",
        )
