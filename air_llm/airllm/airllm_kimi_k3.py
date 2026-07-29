from .airllm_base import AirLLMBaseModel


class AirLLMKimiK3(AirLLMBaseModel):
    """Kimi K3 (``KimiK3ForConditionalGeneration``).

    K3 is a multimodal MoE checkpoint, so the decoder lives one level deeper than usual, under
    ``language_model``, and the checkpoint carries a vision tower and projector alongside it.
    It also adds a pair of top-level Attention Residual modules that sit outside the normal
    embed -> layers -> norm -> lm_head sequence. Everything else (MXFP4 weights, per-layer
    streaming) is handled by the generic base class.

    Each layer holds 896 experts and routes a token to 16 of them, so the experts are streamed
    individually rather than by layer: expanded, a layer's experts are ~55GB but a token needs
    ~1GB of them.
    """

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'language_model.model.embed_tokens',
            'layer_prefix': 'language_model.model.layers',
            'norm': 'language_model.model.norm',
            'lm_head': 'language_model.lm_head',
            'expert_prefix': 'block_sparse_moe.experts',
            # Not streamed: loaded once and kept resident. Together these are well under 1GB.
            'resident': [
                'language_model.model.output_attn_res_norm',
                'language_model.model.output_attn_res_proj',
                'mm_projector',
                'vision_tower',
            ],
        }
