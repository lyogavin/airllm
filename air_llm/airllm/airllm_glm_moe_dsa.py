from .airllm_base import AirLLMBaseModel
from .airllm_fp8 import replace_with_airllm_fp8


def _quantization_method(config):
    if isinstance(config, dict):
        return config.get("quant_method")
    return getattr(config, "quant_method", None)


class AirLLMGlmMoeDsa(AirLLMBaseModel):
    """GLM-5/5.2 loader with an AirLLM-native fine-grained FP8 path."""

    def prepare_quantized_model(self, quantization_config):
        if _quantization_method(quantization_config) == "fp8":
            replace_with_airllm_fp8(self.model, self.model_local_path, quantization_config)
            return
        super().prepare_quantized_model(quantization_config)
