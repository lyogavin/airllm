from transformers import AutoConfig

from .airllm import AirLLMLlama2
from .airllm_mistral import AirLLMMistral
from .airllm_baichuan import AirLLMBaichuan
from .airllm_internlm import AirLLMInternLM
from .airllm_chatglm import AirLLMChatGLM
from .airllm_qwen import AirLLMQWen


class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        if "QWen" in config.architectures[0]:
            return AirLLMQWen(pretrained_model_name_or_path, *inputs, **kwargs)
        elif "Baichuan" in config.architectures[0]:
            return AirLLMBaichuan(pretrained_model_name_or_path, *inputs, **kwargs)
        elif "ChatGLM" in config.architectures[0]:
            return AirLLMChatGLM(pretrained_model_name_or_path, *inputs, **kwargs)
        elif "InternLM" in config.architectures[0]:
            return AirLLMInternLM(pretrained_model_name_or_path, *inputs, **kwargs)
        elif "Mistral" in config.architectures[0]:
            return AirLLMMistral(pretrained_model_name_or_path, *inputs, **kwargs)
        elif "Llama" in config.architectures[0]:
            return AirLLMLlama2(pretrained_model_name_or_path, *inputs, **kwargs)
        else:
            print(f"unknown artichitecture: {config.architectures[0]}, try to use Llama2...")
            return AirLLMLlama2(pretrained_model_name_or_path, *inputs, **kwargs)