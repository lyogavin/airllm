

import os
from pathlib import Path
import mlx.core as mx
from .model_persister import ModelPersister
from mlx.utils import tree_unflatten
import torch

import psutil
import numpy as np
from itertools import starmap



def map_torch_to_mlx(model):

    # things to change
    # 1. there's no "model." in the weight names
    model = {k.replace("model.", ""): v for k, v in model.items()}

    # 2. mlp is called feed_forward
    model = {k.replace("mlp", "feed_forward"): v for k, v in model.items()}

    # 3. up_proj, down_proj, gate_proj
    model = {k.replace("down_proj", "w2"): v for k, v in model.items()}
    model = {k.replace("up_proj", "w3"): v for k, v in model.items()}
    model = {k.replace("gate_proj", "w1"): v for k, v in model.items()}

    # 4. layernorms
    model = {
        k.replace("input_layernorm", "attention_norm"): v for k, v in model.items()
    }
    model = {
        k.replace("post_attention_layernorm", "ffn_norm"): v for k, v in model.items()
    }

    # 5. lm head
    model = {k.replace("lm_head", "output"): v for k, v in model.items()}

    # 6. token emb
    model = {k.replace("embed_tokens", "tok_embeddings"): v for k, v in model.items()}

    # 7. attention
    model = {k.replace("self_attn", "attention"): v for k, v in model.items()}
    model = {k.replace("q_proj", "wq"): v for k, v in model.items()}
    model = {k.replace("k_proj", "wk"): v for k, v in model.items()}
    model = {k.replace("v_proj", "wv"): v for k, v in model.items()}
    model = {k.replace("o_proj", "wo"): v for k, v in model.items()}


    #weights = {k: v.to(torch.float16).numpy() for k, v in model.items()}


    return model

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
            layer_state_dict = mx.load(to_load_path)
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