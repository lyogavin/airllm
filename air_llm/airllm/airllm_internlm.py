
from transformers import GenerationConfig

from .airllm_base import AirLLMBaseModel



class AirLLMInternLM(AirLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(AirLLMInternLM, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False
    def get_generation_config(self):
        return GenerationConfig()


