from .airllm_base import AirLLMBaseModel


class AirLLMQwen3_5(AirLLMBaseModel):
    """Qwen3.5 / Qwen3.8 dense VL (``Qwen3_5ForConditionalGeneration``).

    Qwen3.8-27B is this architecture: a native vision-language wrapper whose decoder lives under
    ``model.language_model``, with a SigLIP-style tower at ``model.visual``. The language stack is
    a hybrid of Gated DeltaNet (linear attention) and Gated Attention, 64 layers, bf16, ~27B.

    Transformers ignores the checkpoint's ``mtp.*`` Multi-Token Prediction head
    (``_keys_to_ignore_on_load_unexpected``), so we do not stream or load it. The vision tower is
    kept resident: text-only ``generate()`` never runs it, but leaving it on meta would crash the
    moment a caller passes ``pixel_values``.

    Needs transformers 5.8+ (the class is in-tree; this repo ships no remote modeling code).
    Optional CUDA kernels ``fla`` / ``causal-conv1d`` speed up DeltaNet; transformers falls back
    to a PyTorch implementation when they are missing.
    """

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.language_model.embed_tokens',
            'layer_prefix': 'model.language_model.layers',
            'norm': 'model.language_model.norm',
            'lm_head': 'lm_head',
            'resident': [
                'model.visual',
            ],
        }
