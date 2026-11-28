

import os
from pathlib import Path
import mlx.core as mx
from .model_persister import ModelPersister
from mlx.utils import tree_unflatten
import torch

import psutil
import numpy as np
from itertools import starmap



# Ordered (old, new) substring renames from HF/torch parameter names to the MLX module names.
_TORCH_TO_MLX_RENAMES = (
    ("model.", ""),
    ("mlp", "feed_forward"),
    ("down_proj", "w2"),
    ("up_proj", "w3"),
    ("gate_proj", "w1"),
    ("input_layernorm", "attention_norm"),
    ("post_attention_layernorm", "ffn_norm"),
    ("lm_head", "output"),
    ("embed_tokens", "tok_embeddings"),
    ("self_attn", "attention"),
    ("q_proj", "wq"),
    ("k_proj", "wk"),
    ("v_proj", "wv"),
    ("o_proj", "wo"),
)


def _rename(key):
    for old, new in _TORCH_TO_MLX_RENAMES:
        key = key.replace(old, new)
    return key


def map_torch_to_mlx(model):
    # One pass over the dict instead of rebuilding it once per rename.
    return {_rename(k): v for k, v in model.items()}

class MlxModelPersister(ModelPersister):


    def __init__(self, *args, **kwargs):


        super(MlxModelPersister, self).__init__(*args, **kwargs)


    def model_persist_exist(self, layer_name, saving_path):



        safetensor_exists = os.path.exists(str(saving_path / (layer_name + 'mlx.npz')))
        done_marker_exists = os.path.exists(str(saving_path / (layer_name + 'mlx.done')))

        #print(f"checking {layer_name}, {saving_path} - {safetensor_exists},{done_marker_exists}")

        return safetensor_exists and done_marker_exists

    def persist_model(self, state_dict, layer_name, saving_path):
        #save_file(state_dict, saving_path / (layer_name + 'safetensors'))
        weights = {k: v.to(torch.float16).numpy() for k, v in state_dict.items()}
        np.savez(
            saving_path / (layer_name + 'mlx'),
            **weights#map_torch_to_mlx(state_dict)
        )

        print(f"saved as: {saving_path / (layer_name + 'mlx')}")

        # set done marker
        (saving_path / (layer_name + 'mlx.done')).touch()


    def load_model(self, layer_name, path):
        try:
            to_load_path = Path(path) / (layer_name + ".mlx.npz")
            #available = psutil.virtual_memory().available / 1024 / 1024
            #print(f"start loading: {to_load_path}, before loading: {available:.02f}")
            layer_state_dict = mx.load(str(to_load_path))
            #available = psutil.virtual_memory().available / 1024 / 1024
            #print(f"loaded {layer_name}, available mem: {available:.02f}")

            layer_state_dict = map_torch_to_mlx(layer_state_dict)

            weights = tree_unflatten(list(layer_state_dict.items()))

            #for el in layer_name.split("."):
            #    if len(el) > 0:
            #        if el.isdigit():
            #            el = int(el)
            #        weights = weights[el]

            return weights
        except Exception as ex:
            print(f"error: {layer_name}, {path}")
            raise ex
