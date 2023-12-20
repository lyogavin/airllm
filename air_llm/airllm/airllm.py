

from .airllm_base import AirLLMBaseModel



class AirLLMLlama2(AirLLMBaseModel):
    def __init__(self, *args, **kwargs):
        super(AirLLMLlama2, self).__init__(*args, **kwargs)

