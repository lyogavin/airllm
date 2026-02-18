
from transformers import GenerationConfig

from .airllm_base import AirLLMBaseModel


class AirLLMQwen3Moe(AirLLMBaseModel):
    """
    AirLLM handler for Qwen3.5 MoE (Mixture of Experts) models.
    
    Supports:
    - Qwen3_5MoeForConditionalGeneration (multimodal)
    - Qwen3MoeForCausalLM
    - Models with MoE architecture (512 experts, 10 experts per token typical)
    """

    def __init__(self, *args, **kwargs):
        super(AirLLMQwen3Moe, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        # BetterTransformer doesn't support MoE architecture
        return False

    def get_generation_config(self):
        return GenerationConfig()

    def set_layer_names_dict(self):
        # Qwen3.5 MoE is a multimodal model - text layers are nested under language_model
        self.layer_names_dict = {
            'embed': 'model.language_model.embed_tokens',
            'layer_prefix': 'model.language_model.layers',
            'norm': 'model.language_model.norm',
            'lm_head': 'lm_head',
        }


class AirLLMQwen3(AirLLMBaseModel):
    """
    AirLLM handler for Qwen3 dense models (non-MoE).
    
    Supports:
    - Qwen3ForCausalLM
    """

    def __init__(self, *args, **kwargs):
        super(AirLLMQwen3, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        # Use SDPA instead of BetterTransformer for Qwen3
        return False

    def get_generation_config(self):
        return GenerationConfig()

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
        }
