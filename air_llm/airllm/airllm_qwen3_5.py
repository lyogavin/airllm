import torch

from .airllm_base import AirLLMBaseModel


class AirLLMQWen3_5(AirLLMBaseModel):
    def __init__(self, *args, **kwargs):
        super(AirLLMQWen3_5, self).__init__(*args, **kwargs)

    def create_model_from_config(self, **kwargs):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_config(self.config, trust_remote_code=True, **kwargs)

    def get_use_better_transformer(self):
        return False

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.language_model.embed_tokens',
            'layer_prefix': 'model.language_model.layers',
            'norm': 'model.language_model.norm',
            'lm_head': 'lm_head',
        }

    def get_pos_emb_args(self, len_p, len_s):
        position_ids = torch.arange(
            len_p,
            len_p + len_s,
            dtype=torch.long,
            device=self.running_device,
        )[None, :]
        hidden_size = getattr(self.config.text_config, "hidden_size", 1)
        x = torch.empty(
            1,
            len_s,
            hidden_size,
            dtype=self.running_dtype,
            device=self.running_device,
        )
        position_embeddings = self.model.model.language_model.rotary_emb(x, position_ids)
        return {'position_embeddings': position_embeddings}
