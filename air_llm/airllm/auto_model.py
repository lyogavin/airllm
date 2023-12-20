import importlib
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
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if 'hf_token' in kwargs:
            print(f"using hf_token")
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, token=kwargs['hf_token'])
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        if "QWen" in config.architectures[0]:
            return "airllm", "AirLLMQWen"
        elif "Baichuan" in config.architectures[0]:
            return "airllm", "AirLLMBaichuan"
        elif "ChatGLM" in config.architectures[0]:
            return "airllm", "AirLLMChatGLM"
        elif "InternLM" in config.architectures[0]:
            return "airllm", "AirLLMInternLM"
        elif "Mistral" in config.architectures[0]:
            return "airllm", "AirLLMMistral"
        elif "Llama" in config.architectures[0]:
            return "airllm", "AirLLMLlama2"
        else:
            print(f"unknown artichitecture: {config.architectures[0]}, try to use Llama2...")
            return "airllm", "AirLLMLlama2"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):

        module, cls = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)

        module = importlib.import_module(module)
        class_ = getattr(module, cls)
        return class_(pretrained_model_name_or_path, *inputs, ** kwargs)