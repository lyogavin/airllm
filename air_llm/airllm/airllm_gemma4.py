from .airllm_base import AirLLMBaseModel


EMBED_MODULE = "model.language_model.embed_tokens"
LAYER_PREFIX = "model.language_model.layers"
NORM_MODULE = "model.language_model.norm"
LM_HEAD_MODULE = "lm_head"
PLE_EMBED_MODULE = "model.language_model.embed_tokens_per_layer"
PLE_PROJECTION_MODULE = "model.language_model.per_layer_model_projection"
PLE_NORM_MODULE = "model.language_model.per_layer_projection_norm"
VISION_EMBED_MODULE = "model.embed_vision"
VISION_TOWER_MODULE = "model.vision_tower"
RESIDENT_MODULES = [
    PLE_EMBED_MODULE,
    PLE_PROJECTION_MODULE,
    PLE_NORM_MODULE,
    VISION_EMBED_MODULE,
    VISION_TOWER_MODULE,
]


class AirLLMGemma4(AirLLMBaseModel):
    """Gemma 4 multimodal MoE model with a nested language decoder."""

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            "embed": EMBED_MODULE,
            "layer_prefix": LAYER_PREFIX,
            "norm": NORM_MODULE,
            "lm_head": LM_HEAD_MODULE,
            "resident": RESIDENT_MODULES,
        }
