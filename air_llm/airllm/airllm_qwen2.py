
from transformers import GenerationConfig


from .airllm_base import AirLLMBaseModel



class AirLLMQWen2(AirLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(AirLLMQWen2, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False


