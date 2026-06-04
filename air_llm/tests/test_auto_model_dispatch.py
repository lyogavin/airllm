"""Offline unit tests for AutoModel architecture detection.

These tests patch ``transformers.AutoConfig.from_pretrained`` so they can
exercise the dispatch logic without making any network calls and without
needing a real model checkpoint.
"""

import unittest
from unittest.mock import patch

from airllm.auto_model import (
    AutoModel,
    UnknownArchitectureError,
    _ARCHITECTURE_DISPATCH,
)


def _fake_config(architecture):
    """Build a minimal mock that quacks like a transformers PretrainedConfig."""

    class FakeConfig:
        architectures = [architecture] if architecture else None

    return FakeConfig()


class ArchitectureDispatchTests(unittest.TestCase):
    """Each supported architecture must map to exactly one wrapper class."""

    EXPECTED = {
        "LlamaForCausalLM":          "AirLLMLlama2",
        "MistralForCausalLM":        "AirLLMMistral",
        "MixtralForCausalLM":        "AirLLMMixtral",
        "Qwen2ForCausalLM":          "AirLLMQWen2",
        "QWenLMHeadModel":           "AirLLMQWen",
        "BaichuanForCausalLM":       "AirLLMBaichuan",
        "ChatGLMModel":              "AirLLMChatGLM",
        "InternLMForCausalLM":       "AirLLMInternLM",
    }

    def test_every_known_architecture_resolves(self):
        for arch, expected_cls in self.EXPECTED.items():
            with patch(
                "airllm.auto_model.AutoConfig.from_pretrained",
                return_value=_fake_config(arch),
            ):
                module, cls = AutoModel.get_module_class("any/repo")
                self.assertEqual(module, "airllm", f"module for {arch}")
                self.assertEqual(cls, expected_cls, f"class for {arch}")

    def test_qwen2_is_more_specific_than_qwen(self):
        """Qwen2ForCausalLM must NOT fall through to the QWen matcher."""
        with patch(
            "airllm.auto_model.AutoConfig.from_pretrained",
            return_value=_fake_config("Qwen2ForCausalLM"),
        ):
            _, cls = AutoModel.get_module_class("any/repo")
            self.assertEqual(cls, "AirLLMQWen2")

    def test_mixtral_does_not_fall_through_to_mistral(self):
        with patch(
            "airllm.auto_model.AutoConfig.from_pretrained",
            return_value=_fake_config("MixtralForCausalLM"),
        ):
            _, cls = AutoModel.get_module_class("any/repo")
            self.assertEqual(cls, "AirLLMMixtral")

    def test_unknown_architecture_raises(self):
        with patch(
            "airllm.auto_model.AutoConfig.from_pretrained",
            return_value=_fake_config("SomeFutureModelForCausalLM"),
        ):
            with self.assertRaises(UnknownArchitectureError) as ctx:
                AutoModel.get_module_class("org/some-future-model")
            msg = str(ctx.exception)
            self.assertIn("SomeFutureModelForCausalLM", msg)
            self.assertIn("Supported architectures", msg)

    def test_empty_architectures_raises(self):
        with patch(
            "airllm.auto_model.AutoConfig.from_pretrained",
            return_value=_fake_config(None),
        ):
            with self.assertRaises(UnknownArchitectureError):
                AutoModel.get_module_class("org/weird-model")

    def test_dispatch_table_has_no_duplicate_needles(self):
        """No two patterns should collide in a way that makes the table ambiguous."""
        needles = [n for n, _ in _ARCHITECTURE_DISPATCH]
        self.assertEqual(len(needles), len(set(needles)))


class HfTokenPassthroughTests(unittest.TestCase):
    """The hf_token kwarg, if provided, must reach AutoConfig.from_pretrained."""

    def test_hf_token_is_forwarded(self):
        with patch(
            "airllm.auto_model.AutoConfig.from_pretrained",
            return_value=_fake_config("LlamaForCausalLM"),
        ) as mocked:
            AutoModel.get_module_class("any/repo", hf_token="secret-token")
            kwargs = mocked.call_args.kwargs
            self.assertEqual(kwargs.get("token"), "secret-token")

    def test_no_hf_token_passes_none(self):
        with patch(
            "airllm.auto_model.AutoConfig.from_pretrained",
            return_value=_fake_config("LlamaForCausalLM"),
        ) as mocked:
            AutoModel.get_module_class("any/repo")
            kwargs = mocked.call_args.kwargs
            self.assertIsNone(kwargs.get("token"))


if __name__ == "__main__":
    unittest.main()
