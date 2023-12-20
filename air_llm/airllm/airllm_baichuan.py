
from transformers import GenerationConfig

from .tokenization_baichuan import BaichuanTokenizer

from .airllm_base import AirLLMBaseModel



class AirLLMBaichuan(AirLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(AirLLMBaichuan, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False
    def get_tokenizer(self, hf_token=None):
        # use this hack util the bug is fixed: https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/discussions/2
        return BaichuanTokenizer.from_pretrained(self.model_local_path, use_fast=False, trust_remote_code=True)

    def get_generation_config(self):
        return GenerationConfig()


