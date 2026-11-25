import tempfile
import unittest

from transformers import GenerationConfig

from airllm.airllm_base import AirLLMBaseModel


class TestGenerationConfig(unittest.TestCase):
    def test_base_loads_the_models_real_generation_config(self):
        # base.get_generation_config() should read the model's generation_config.json (eos_token_id,
        # etc.) rather than returning an empty config that transformers would fall back to defaults on.
        with tempfile.TemporaryDirectory() as d:
            GenerationConfig(eos_token_id=2, bos_token_id=1, max_new_tokens=7).save_pretrained(d)

            obj = AirLLMBaseModel.__new__(AirLLMBaseModel)  # skip heavy __init__
            obj.model_local_path = d
            cfg = obj.get_generation_config()

            self.assertEqual(cfg.eos_token_id, 2)
            self.assertEqual(cfg.max_new_tokens, 7)

    def test_base_falls_back_to_empty_config_when_none_present(self):
        with tempfile.TemporaryDirectory() as d:
            obj = AirLLMBaseModel.__new__(AirLLMBaseModel)
            obj.model_local_path = d
            cfg = obj.get_generation_config()
            self.assertIsInstance(cfg, GenerationConfig)  # no crash, usable default

    def test_custom_arch_subclasses_do_not_shadow_generation_config(self):
        # The custom-architecture subclasses used to override get_generation_config() to return an
        # empty GenerationConfig(), discarding the model's real generation defaults. They must inherit
        # base's loader instead. (Baichuan pulls in an optional tokenizer dep, so import defensively.)
        checked = 0
        for module_name, cls_name in (
            ('airllm.airllm_qwen', 'AirLLMQWen'),
            ('airllm.airllm_chatglm', 'AirLLMChatGLM'),
            ('airllm.airllm_internlm', 'AirLLMInternLM'),
            ('airllm.airllm_qwen2', 'AirLLMQWen2'),
            ('airllm.airllm_mistral', 'AirLLMMistral'),
            ('airllm.airllm_mixtral', 'AirLLMMixtral'),
            ('airllm.airllm_baichuan', 'AirLLMBaichuan'),
        ):
            try:
                mod = __import__(module_name, fromlist=[cls_name])
                cls = getattr(mod, cls_name)
            except Exception:
                continue  # optional dependency missing; skip that family
            self.assertIs(cls.get_generation_config, AirLLMBaseModel.get_generation_config,
                          f"{cls_name} should inherit get_generation_config from AirLLMBaseModel")
            checked += 1
        self.assertGreater(checked, 0, "expected to check at least one subclass")


if __name__ == '__main__':
    unittest.main()
