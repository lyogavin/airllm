"""Tests for AutoModel.get_module_class routing.

After the v3.0 rewrite most architectures route through the generic
AirLLMBaseModel.  Only models with non-standard module layouts (custom
remote-code models) keep dedicated subclasses.
"""

import pytest


class TestAutoModel:
    # These architectures have custom remote-code overrides and should still
    # resolve to their dedicated AirLLM subclasses.
    ARCH_OVERRIDES_EXPECTED = {
        "THUDM/chatglm3-6b-base": "AirLLMChatGLM",
        "internlm/internlm-chat-7b": "AirLLMInternLM",
    }

    # Standard HuggingFace architectures should route to the generic base.
    GENERIC_ARCHITECTURES = [
        "garage-bAInd/Platypus2-7B",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-v0.1",
    ]

    @pytest.mark.parametrize("model_id,expected_cls",
                             list(ARCH_OVERRIDES_EXPECTED.items()))
    def test_override_architectures_resolve_to_dedicated_class(
            self, model_id, expected_cls):
        from airllm.auto_model import AutoModel
        module, cls = AutoModel.get_module_class(model_id)
        assert cls == expected_cls, f"{model_id}: expected {expected_cls}, got {cls}"

    @pytest.mark.parametrize("model_id", GENERIC_ARCHITECTURES)
    def test_generic_architectures_resolve_to_base_model(self, model_id):
        from airllm.auto_model import AutoModel
        module, cls = AutoModel.get_module_class(model_id)
        assert cls == "AirLLMBaseModel", (
            f"{model_id}: after v3.0 rewrite, standard architectures "
            f"should route to AirLLMBaseModel, got {cls}"
        )
