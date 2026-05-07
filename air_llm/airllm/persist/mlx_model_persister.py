

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

    # 0. drop legacy buffers that no longer exist in the MLX module tree.
    # rotary_emb.inv_freq is a registered buffer on older transformers Llama
    # implementations; MLX's nn.RoPE is parameter-free, so the key has nowhere
    # to land and mlx.nn.Module.update() now raises on unknown keys.
    model = {k: v for k, v in model.items() if "rotary_emb" not in k}

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

    def persist_model(self, state_dict, layer_name, saving_path, compression=None):
        # Default: dump fp16 weights as numpy. With compression='4bit', quantize
        # 2D Linear/Embedding weights via mx.quantize and store the (w_q, scales,
        # biases) triplet under <key>.weight / .scales / .biases. Loaded into
        # nn.QuantizedLinear / nn.QuantizedEmbedding by AirLLMLlamaMlx.
        weights = {}
        if compression == '4bit':
            bits = 4
            group_size = 64
            for k, v in state_dict.items():
                np_v = v.to(torch.float16).numpy()
                # quantize 2D weights whose inner dim is divisible by group_size
                if (k.endswith('.weight') and np_v.ndim == 2
                        and np_v.shape[-1] % group_size == 0):
                    mx_w = mx.array(np_v)
                    w_q, scales, biases = mx.quantize(mx_w, group_size=group_size, bits=bits)
                    base = k[:-len('.weight')]
                    weights[k] = np.array(w_q)
                    weights[base + '.scales'] = np.array(scales)
                    weights[base + '.biases'] = np.array(biases)
                else:
                    weights[k] = np_v
        else:
            weights = {k: v.to(torch.float16).numpy() for k, v in state_dict.items()}

        np.savez(saving_path / (layer_name + 'mlx'), **weights)
        print(f"saved as: {saving_path / (layer_name + 'mlx')}{' (4bit)' if compression == '4bit' else ''}")
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