import unittest

import airllm.auto_model as auto_model_mod
from airllm.auto_model import AutoModel
from airllm.utils import load_prefer_no_remote_code


class _FakeConfig:
    def __init__(self, architectures):
        self.architectures = architectures


class TestLoadPreferNoRemoteCode(unittest.TestCase):
    def test_uses_false_and_does_not_retry_when_native_load_succeeds(self):
        calls = []

        def loader(path, trust_remote_code=None, **kwargs):
            calls.append(trust_remote_code)
            return 'ok'

        result = load_prefer_no_remote_code(loader, 'some/repo')

        self.assertEqual(result, 'ok')
        self.assertEqual(calls, [False])  # tried False, never fell back to True

    def test_falls_back_to_true_when_native_load_raises(self):
        calls = []

        def loader(path, trust_remote_code=None, **kwargs):
            calls.append(trust_remote_code)
            if trust_remote_code is False:
                raise ValueError('requires trust_remote_code=True')
            return 'ok-remote'

        result = load_prefer_no_remote_code(loader, 'custom/repo')

        self.assertEqual(result, 'ok-remote')
        self.assertEqual(calls, [False, True])  # tried False first, then True

    def test_forwards_extra_kwargs(self):
        seen = {}

        def loader(path, trust_remote_code=None, **kwargs):
            seen.update(kwargs)
            seen['path'] = path
            return 'ok'

        load_prefer_no_remote_code(loader, 'some/repo', token='abc')

        self.assertEqual(seen['path'], 'some/repo')
        self.assertEqual(seen['token'], 'abc')


class TestGetModuleClassTrust(unittest.TestCase):
    """AutoModel.get_module_class must not enable remote code unless it is actually required."""

    def setUp(self):
        self._orig = auto_model_mod.AutoConfig.from_pretrained

    def tearDown(self):
        auto_model_mod.AutoConfig.from_pretrained = self._orig

    def test_standard_arch_loads_config_without_remote_code(self):
        calls = []

        def fake(path, trust_remote_code=None, **kwargs):
            calls.append(trust_remote_code)
            return _FakeConfig(['LlamaForCausalLM'])

        auto_model_mod.AutoConfig.from_pretrained = staticmethod(fake)

        module, cls = AutoModel.get_module_class('some/llama-repo')

        self.assertEqual((module, cls), ('airllm', 'AirLLMBaseModel'))
        self.assertEqual(calls, [False])  # native path only; remote code never enabled

    def test_custom_arch_falls_back_to_remote_code(self):
        calls = []

        def fake(path, trust_remote_code=None, **kwargs):
            calls.append(trust_remote_code)
            if trust_remote_code is False:
                raise ValueError('requires trust_remote_code=True')
            return _FakeConfig(['ChatGLMModel'])

        auto_model_mod.AutoConfig.from_pretrained = staticmethod(fake)

        module, cls = AutoModel.get_module_class('some/chatglm-repo')

        self.assertEqual((module, cls), ('airllm', 'AirLLMChatGLM'))
        self.assertEqual(calls, [False, True])  # only trusts remote code as a fallback


if __name__ == '__main__':
    unittest.main()
