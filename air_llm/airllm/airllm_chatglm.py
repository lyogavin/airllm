
from transformers import GenerationConfig

from .airllm_base import AirLLMBaseModel



class AirLLMChatGLM(AirLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(AirLLMChatGLM, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()

    def get_sequence_len(self, seq):
        return seq.shape[0]

    def get_past_key_values_cache_seq_len(self, past_key_values):
        return past_key_values[0][0].shape[0]


    # customize layer names here
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'transformer.embedding.word_embeddings',
                       'layer_prefix': 'transformer.encoder.layers',
                       'norm': 'transformer.encoder.final_layernorm',
                       'lm_head': 'transformer.output_layer',
                       'rotary_pos_emb': 'transformer.rotary_pos_emb'}

    def get_pos_emb_args(self, len_p, len_s):
        # Rotary positional embeddings
        rotary_pos_emb = self.model.transformer.rotary_pos_emb(self.config.seq_length)
        rotary_pos_emb = rotary_pos_emb[None, : len_s]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        return {'rotary_pos_emb': rotary_pos_emb}

    def get_past_key_value_args(self, k_cache, v_cache):
        return {'kv_cache': (k_cache, v_cache)}

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        return {'attention_mask': None}

    def get_position_ids_args(self, full_position_ids, len_p, len_s):
        return {}