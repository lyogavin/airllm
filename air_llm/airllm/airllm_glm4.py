from transformers import GenerationConfig

from .airllm_base import AirLLMBaseModel


class AirLLMGLM4(AirLLMBaseModel):
    def __init__(self, *args, **kwargs):
        super(AirLLMGLM4, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()
