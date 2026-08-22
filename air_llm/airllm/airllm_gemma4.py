from .airllm_base import AirLLMBaseModel


EMBED_MODULE = "model.language_model.embed_tokens"
LAYER_PREFIX = "model.language_model.layers"
NORM_MODULE = "model.language_model.norm"
LM_HEAD_MODULE = "lm_head"
RESIDENT_MODULES = ["model.embed_vision", "model.vision_tower"]


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
