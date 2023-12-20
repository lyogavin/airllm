
from transformers import GenerationConfig

from .airllm_base import AirLLMBaseModel



class AirLLMQWen(AirLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(AirLLMQWen, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False
    def get_generation_config(self):
        return GenerationConfig()


    def get_past_key_values_cache_seq_len(self, past_key_values):
        return past_key_values[0][0].shape[1]


    # customize layer names here
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'transformer.wte',
                       'layer_prefix': 'transformer.h',
                       'norm': 'transformer.ln_f',
                       'lm_head': 'lm_head',}

    def get_pos_emb_args(self, len_p, len_s):
        # Rotary positional embeddings
        if self.model.transformer.use_dynamic_ntk:
            ntk_alpha_list = [1.0]
        elif len_p + len_s != len_s:
            ntk_alpha_list = self.model.transformer.rotary_emb._ntk_alpha_cached_list
        else:
            ntk_alpha_list = []
            ntk_alpha = self.model.transformer.get_ntk_alpha(len_p + len_s)
            ntk_alpha_list.append(ntk_alpha)
        self.model.transformer.rotary_emb._ntk_alpha_cached_list = ntk_alpha_list
        rotary_pos_emb_list = [
            self.model.transformer.rotary_emb(len_p + len_s, ntk_alpha=ntk_alpha) for ntk_alpha in ntk_alpha_list
        ]
        return {'rotary_pos_emb_list': rotary_pos_emb_list}

    def get_past_key_value_args(self, k_cache, v_cache):
        return {'layer_past': (k_cache, v_cache)}

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        return {'attention_mask': None}

    def  get_position_ids_args(self, full_position_ids, len_p, len_s):

        return {}