from .airllm_base import AirLLMBaseModel
from .airllm_packed import replace_with_airllm_packed_linears


class AirLLMKimiK25(AirLLMBaseModel):
    """Text-only AirLLM streaming for Kimi K2.5/K2.6/K2.7 checkpoints.

    Kimi's multimodal wrapper nests its DeepSeek-style decoder under
    ``language_model``. The vision tower is intentionally not streamed: it is
    not needed for text requests, while silently accepting image inputs would
    execute meta-device weights.
    """

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            "embed": "language_model.model.embed_tokens",
            "layer_prefix": "language_model.model.layers",
            "norm": "language_model.model.norm",
            "lm_head": "language_model.lm_head",
        }

    def init_model(self):
        # The official multimodal config forces FlashAttention2 for MoonViT.
        # AirLLM's Kimi path is text-only and never executes the vision tower,
        # so requiring flash-attn merely to construct its meta skeleton would
        # unnecessarily block Windows/ROCm users.
        vision_config = getattr(self.config, "vision_config", None)
        if isinstance(vision_config, dict):
            vision_config["_attn_implementation"] = "eager"
        elif vision_config is not None:
            vision_config._attn_implementation = "eager"
            vision_config._attn_implementation_internal = "eager"
        super().init_model()

    def prepare_quantized_model(self, quantization_config):
        super().prepare_quantized_model(quantization_config)
        quant_method = (
            quantization_config.get("quant_method")
            if isinstance(quantization_config, dict)
            else getattr(quantization_config, "quant_method", None)
        )
        if quant_method == "compressed-tensors":
            replace_with_airllm_packed_linears(self.model)
            # The custom modules consume the published packed tensors directly.
            # They must be loaded verbatim and can use normal module.to('meta') cleanup.
            self.hf_quantizer = None

    @staticmethod
    def _reject_visual_inputs(kwargs):
        for name in ("pixel_values", "grid_thws"):
            if kwargs.get(name) is not None:
                raise NotImplementedError(
                    "AirLLM's Kimi K2.5/K2.6/K2.7 path currently supports text-only inference; "
                    "vision-tower streaming is not implemented."
                )

    def generate(self, *args, **kwargs):
        self._reject_visual_inputs(kwargs)
        return super().generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        self._reject_visual_inputs(kwargs)
        return super().forward(*args, **kwargs)
