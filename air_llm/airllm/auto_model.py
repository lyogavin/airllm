import importlib
from transformers import AutoConfig
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from airllm import AirLLMLlamaMlx

# Architectures that need a dedicated AirLLM subclass because of a non-standard module layout
# (custom remote-code models). Everything else uses the generic AirLLMBaseModel, which streams any
# standard *ForCausalLM (model.model.layers + lm_head / norm) and lets transformers own the
# forward pass, so newly released architectures work without code changes.
ARCH_OVERRIDES = {
    "ChatGLMModel": "AirLLMChatGLM",
    "ChatGLMForConditionalGeneration": "AirLLMChatGLM",
    "QWenLMHeadModel": "AirLLMQWen",
    "BaichuanForCausalLM": "AirLLMBaichuan",
    "BaiChuanForCausalLM": "AirLLMBaichuan",
    "InternLMForCausalLM": "AirLLMInternLM",
    # Kimi K2.5/2.6/2.7 wrap the text decoder below ``language_model``.
    "KimiK25ForConditionalGeneration": "AirLLMKimiK25",
    # GLM-5/5.2 FP8 publishes per-expert tensors that must not be fused by the
    # normal Transformers loader before AirLLM's layer streaming hooks run.
    "GlmMoeDsaForCausalLM": "AirLLMGlmMoeDsa",
}


class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        config_kwargs = {
            "trust_remote_code": True,
        }
        if kwargs.get("hf_token") is not None:
            config_kwargs["token"] = kwargs["hf_token"]
        if kwargs.get("cache_dir") is not None:
            config_kwargs["cache_dir"] = kwargs["cache_dir"]
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)

        architectures = getattr(config, "architectures", None) or []
        arch = architectures[0] if architectures else ""

        cls_name = ARCH_OVERRIDES.get(arch)
        if cls_name is None:
            print(f"using generic AirLLM streaming model for architecture: {arch or 'unknown'}")
            cls_name = "AirLLMBaseModel"
        return "airllm", cls_name

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):

        if is_on_mac_os:
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        module, class_name = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        module = importlib.import_module(module)
        class_ = getattr(module, class_name)
        return class_(pretrained_model_name_or_path, *inputs, **kwargs)
