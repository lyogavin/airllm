import gc
import json
import os
import ctypes
import shutil
from tqdm import tqdm
from pathlib import Path
from glob import glob
import time

from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True


import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .persist import ModelPersister


try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
except ImportError:
    bitsandbytes_installed = False


import huggingface_hub


# replacement for bnb quantstat.as_dict(True), until the bug is fixed....
def save_quant_state_to_dict(self, packed=True):
    """
    returns dict of tensors and strings to use in serialization via _save_to_state_dict()
    param: packed -- returns dict[str, torch.Tensor] for state_dict
    """
    qs_dict = {
        'quant_type': self.quant_type,
        'absmax': self.absmax,
        'blocksize': self.blocksize,
        'quant_map': self.code,
        'dtype': str(self.dtype).strip('torch.'),
        'shape': tuple(self.shape),
    }
    if self.nested:
        qs_dict.update({
            'nested_absmax': self.state2.absmax,
            'nested_blocksize': self.state2.blocksize,
            'nested_quant_map': self.state2.code,
            'nested_dtype': str(self.state2.dtype).strip('torch.'),
            'nested_offset': self.offset.item(),
        })
    if not packed:
        return qs_dict

    qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
    non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
    qs_packed_dict["quant_state." + "bitsandbytes__" + self.quant_type] = bnb.utils.pack_dict_to_tensor(non_tensor_dict)
    return qs_packed_dict



class NotEnoughSpaceException(Exception):
    pass

# Function to clean RAM & vRAM
def clean_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as ex:
        # maybe platform
        pass
    torch.cuda.empty_cache()


def calculate_n_layers_in_gpu(checkpoint_path, layer_names, device="cuda:0", vram_safety_margin=0.8,
                              layer_shards_saving_path=None):
    """
    Auto-detect how many layers can fit in currently available VRAM.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the original model directory.
    layer_names : list of str
        Ordered list of layer names (embed, layers.0..N, norm, lm_head).
    device : str
        CUDA device string, e.g. "cuda:0".
    vram_safety_margin : float
        Fraction of free VRAM to use (default 0.8 = 80%, leaving 20% for activations).
    layer_shards_saving_path : str or Path, optional
        Path where layer shards are cached. If provided, layer sizes are read from here.

    Returns
    -------
    int
        Number of layers to load into GPU simultaneously (at least 1).
    """
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return 1

    try:
        device_idx = int(device.split(":")[-1]) if ":" in device else 0
        free_bytes, _ = torch.cuda.mem_get_info(device_idx)
        usable_bytes = int(free_bytes * vram_safety_margin)

        # Determine where to look for layer shard files.
        # checkpoint_path is already the splitted_model dir (returned by find_or_create_local_splitted_path).
        # Also check layer_shards_saving_path/splitted_model as a fallback.
        search_paths = [Path(checkpoint_path)]
        if layer_shards_saving_path is not None:
            search_paths.append(Path(layer_shards_saving_path) / "splitted_model")
            search_paths.append(Path(layer_shards_saving_path))

        # Estimate average layer size from the transformer layers (skip embed/norm/lm_head)
        transformer_layer_names = [n for n in layer_names
                                   if n not in (layer_names[0], layer_names[-1], layer_names[-2])]

        MIN_REAL_LAYER_BYTES = 1024 * 1024  # 1MB — stubs are 16 bytes
        layer_sizes = []
        # Sample up to 5 transformer layers, skipping stubs
        for layer_name in transformer_layer_names:
            if len(layer_sizes) >= 5:
                break
            for search_path in search_paths:
                layer_file = search_path / (layer_name + ".safetensors")
                size = layer_file.stat().st_size if layer_file.exists() else 0
                if size >= MIN_REAL_LAYER_BYTES:
                    layer_sizes.append(size)
                    break

        if not layer_sizes:
            print(f"Auto VRAM detection: could not find layer shard files to estimate size, defaulting to 1 layer per pass.")
            return 1

        avg_layer_bytes = sum(layer_sizes) / len(layer_sizes)
        n_layers = max(1, int(usable_bytes / avg_layer_bytes))
        n_layers = min(n_layers, len(layer_names))

        print(f"Auto VRAM detection: {free_bytes / 1024**3:.1f}GB free, "
              f"avg layer size {avg_layer_bytes / 1024**3:.2f}GB → loading {n_layers} layer(s) per pass")

        return n_layers

    except Exception as e:
        print(f"VRAM auto-detection failed ({e}), defaulting to 1 layer per pass.")
        return 1


def uncompress_layer_state_dict(layer_state_dict):
    uncompressed_layer_state_dict = None
    if any(['4bit' in k for k in layer_state_dict.keys()]):
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if '4bit' not in k:
                quant_state_dict = {kk[len(k):]: kv for kk, kv in layer_state_dict.items() if kk.startswith(k) and k != kk}
                quant_state = bnb.functional.QuantState.from_dict(qs_dict=quant_state_dict, device="cuda")

                dqv = bnb.functional.dequantize_nf4(v.cuda(), quant_state)
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict
    elif any(['8bit' in k for k in layer_state_dict.keys()]):
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if '8bit' not in k:

                absmax = layer_state_dict[k + ".8bit.absmax"]
                code = layer_state_dict[k + ".8bit.code"]

                dqv = bnb.functional.dequantize_blockwise(v.cuda(),
                                                          bnb.functional.QuantState(absmax=absmax.cuda(),
                                                                                    code=code.cuda(),
                                                                                    blocksize=2048,
                                                                                    dtype=torch.float16))
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict

    return layer_state_dict if uncompressed_layer_state_dict is None else uncompressed_layer_state_dict

def load_layer(local_path, layer_name, profiling=False):
    #layer_state_dict = load_file(Path(local_path) / (layer_name + ".safetensors"), device="cpu")
    layer_state_dict = ModelPersister.get_model_persister().load_model(layer_name, local_path)

    if profiling:
        t = time.process_time()

    to_return = uncompress_layer_state_dict(layer_state_dict)

    #clean_memory()

    if profiling:
        elapsed_time = time.process_time() - t
        return to_return, elapsed_time
    else:
        return to_return



def check_space(checkpoint_path, layer_shards_saving_path=None, compression=None, splitted_model_dir_name='splitted_model'):
    # When a separate saving path is provided the split layers are a reorganization of the
    # source data (same total bytes, not additive), so a cross-path size check is meaningless.
    # Only check space when splitting in-place (no separate saving path).
    if layer_shards_saving_path is not None:
        print(f"Separate layer_shards_saving_path provided, skipping space check.")
        return

    total_shard_files_size_bytes = 0
    for model_shard_file in glob(str(checkpoint_path / '*')):
        total_shard_files_size_bytes += os.path.getsize(model_shard_file)

    total_saved_split_files_size_bytes = 0
    for saved_split_file in glob(str(checkpoint_path / splitted_model_dir_name / '*')):
        total_saved_split_files_size_bytes += os.path.getsize(saved_split_file)

    if compression == '4bit':
        total_shard_files_size_bytes = int(total_shard_files_size_bytes / 0.2813)
    elif compression == '8bit':
        total_shard_files_size_bytes = total_shard_files_size_bytes // 2

    total, used, free = shutil.disk_usage(checkpoint_path)

    if free + total_saved_split_files_size_bytes < total_shard_files_size_bytes:
        raise NotEnoughSpaceException(f"Not enough space. Free space under {checkpoint_path}:"  \
                                      f" {free / 1024 / 1024 / 1024:.02f}GB. Model total size: {total_shard_files_size_bytes / 1024 / 1024 / 1024:.02f}GB. " \
                                      f"existing space under {checkpoint_path} assuming can reuse: {total_saved_split_files_size_bytes/ 1024 / 1024 / 1024:.02f}GB. "
                                      )

def compress_layer_state_dict(layer_state_dict, compression=None):
    compressed_layer_state_dict = None
    if compression == '4bit':
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_nf4(v.cuda(), blocksize=64)
            compressed_layer_state_dict[k] = v_quant
            for quant_state_k, quant_state_v in save_quant_state_to_dict(quant_state).items():
                compressed_layer_state_dict[k + ".4bit." + quant_state_k] = quant_state_v
    elif compression == '8bit':
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_blockwise(v.cuda(), blocksize=2048)
            absmax = quant_state.absmax.clone().contiguous()
            code = quant_state.code.clone().contiguous()
            compressed_layer_state_dict[k] = v_quant
            compressed_layer_state_dict[k + ".8bit.absmax"] = absmax
            compressed_layer_state_dict[k + ".8bit.code"] = code

    return compressed_layer_state_dict if compressed_layer_state_dict is not None else layer_state_dict

def remove_real_and_linked_file(to_delete):
    if (os.path.realpath(to_delete) != to_delete):
        targetpath = os.path.realpath(to_delete)

    os.remove(to_delete)
    if (targetpath):
         os.remove(targetpath)



def split_and_save_layers(checkpoint_path, layer_shards_saving_path=None, splitted_model_dir_name='splitted_model',
                          compression=None, layer_names=None, delete_original=False, repo_id=None, hf_token=None):
    """
    Save the all layers of a model sharded checkpoint using safetensors.
    """

    if compression is not None:
        assert bitsandbytes_installed, f"when using compression bitsandbytes has to be installed."
        splitted_model_dir_name = splitted_model_dir_name + "." + compression

    checkpoint_path = Path(checkpoint_path)


    saving_path = checkpoint_path / splitted_model_dir_name

    if layer_shards_saving_path is not None:
        saving_path = Path(layer_shards_saving_path) / splitted_model_dir_name


    safetensors_format = False
    single_shard_file = None  # set when model is a single safetensors file with no index
    if os.path.exists(checkpoint_path / 'pytorch_model.bin.index.json'):
        with open(checkpoint_path / 'pytorch_model.bin.index.json', 'rb') as f:
            index = json.load(f)['weight_map']
    elif os.path.exists(checkpoint_path / 'model.safetensors.index.json'):
        safetensors_format = True
        with open(checkpoint_path / 'model.safetensors.index.json', 'rb') as f:
            index = json.load(f)['weight_map']
    elif os.path.exists(checkpoint_path / 'model.safetensors'):
        # Single-shard safetensors (e.g. GPTQ models) — build index from file metadata
        safetensors_format = True
        single_shard_file = 'model.safetensors'
        from safetensors import safe_open
        with safe_open(str(checkpoint_path / 'model.safetensors'), framework='pt', device='cpu') as f:
            index = {k: single_shard_file for k in f.keys()}
        print(f"Single-shard safetensors detected: {len(index)} tensors in {single_shard_file}")
    else:
        raise FileNotFoundError(
            f"No model weight files found in {checkpoint_path}. "
            "Expected model.safetensors.index.json, model.safetensors, or pytorch_model.bin.index.json."
        )

    if layer_names is None:
        n_layers = len(set([int(k.split('.')[2]) for k in index.keys() if 'model.layers' in k]))
    else:
        prefix_dot = layer_names['layer_prefix'] + '.'
        n_layers = len(set([int(k[len(prefix_dot):].split('.')[0]) for k in index.keys() if k.startswith(prefix_dot)]))

    if layer_names is None:
        layers = ['model.embed_tokens.'] + [f'model.layers.{i}.' for i in range(n_layers)] + ['model.norm.', 'lm_head.']
    else:
        layers = [layer_names['embed']] + [f'{layer_names["layer_prefix"]}.{i}' for i in range(n_layers)] + [layer_names['norm'], layer_names['lm_head']]

        if 'rotary_pos_emb' in layer_names:
            layers = [layer_names['rotary_pos_emb']] + layers
        layers = [l + "." for l in layers]


    # check if splitting exists and all files are there
    found_layers = None
    #print(f"checking exists: {saving_path}")
    if os.path.exists(saving_path):
        # dir already exists, check if all layer files are there

        found_layers = {}
        for layer in layers:
            found_layers[layer] = ModelPersister.get_model_persister().model_persist_exist(layer, saving_path)

        print(f"found_layers:{found_layers}")
        if all(found_layers.values()):
            # already downloaded, return saving path...
            print(f"saved layers already found in {saving_path}")
            return str(saving_path)
        else:
            print(f"some layer splits found, some are not, re-save all layers in case there's some corruptions.")

    if not delete_original:
        check_space(checkpoint_path, layer_shards_saving_path, compression, splitted_model_dir_name=splitted_model_dir_name)


    all_shard_files = sorted(set(index.values()))
    n_shards = len(all_shard_files)

    if not os.path.exists(saving_path):
        saving_path.mkdir(parents=True, exist_ok=True)

    single_modelfile = None

    # Build a reverse map: shard_file -> list of keys it contains
    shard_to_keys = {}
    for k, v in index.items():
        shard_to_keys.setdefault(v, []).append(k)

    # For single-shard safetensors, keep the file open across all layers to avoid
    # re-reading the full file (e.g. 16 GB) once per layer.
    _single_shard_handle = None
    if single_shard_file is not None and safetensors_format:
        from safetensors import safe_open
        _single_shard_handle = safe_open(
            str(checkpoint_path / single_shard_file), framework='pt', device='cpu'
        )

    try:
        for layer_idx, layer in enumerate(tqdm(layers)):

            # Check if already done
            if ModelPersister.get_model_persister().model_persist_exist(layer, saving_path):
                continue

            # Get ALL shard files that contain keys for this layer
            layer_keys_needed = set(k for k in index if k.startswith(layer))
            if not layer_keys_needed:
                # No keys for this layer in the index — skip (no stub written)
                continue

            shard_files_for_layer = set(index[k] for k in layer_keys_needed)

            # Streaming: load one shard at a time, extract only this layer's keys, free immediately
            layer_state_dict = {}

            if _single_shard_handle is not None:
                # Single-shard: use the already-open handle, read only this layer's tensors
                for k in layer_keys_needed:
                    layer_state_dict[k] = _single_shard_handle.get_tensor(k)
            elif n_shards > 1:
                for shard_file in sorted(shard_files_for_layer):
                    to_load = checkpoint_path / shard_file
                    if not os.path.exists(to_load):
                        assert repo_id is not None, f"Shard file {to_load} not found locally and no repo_id provided for download."
                        huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(to_load),
                                                        token=hf_token)
                    print(f'Layer {layer_idx}: streaming shard {shard_file}')
                    if not safetensors_format:
                        shard_data = torch.load(to_load, map_location='cpu')
                    else:
                        shard_data = load_file(to_load, device='cpu')
                    # Extract only the keys belonging to this layer
                    for k in shard_to_keys.get(shard_file, []):
                        if k.startswith(layer):
                            layer_state_dict[k] = shard_data[k]
                    # Free the full shard immediately
                    del shard_data
                    clean_memory()
            else:
                shard_files = list(shard_files_for_layer)
                single_modelfile = shard_files[0]
                to_load = checkpoint_path / single_modelfile
                if not os.path.exists(to_load):
                    assert repo_id is not None, f"Shard file {to_load} not found locally and no repo_id provided for download."
                    huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(to_load),
                                                    token=hf_token)
                if not safetensors_format:
                    shard_data = torch.load(to_load, map_location='cpu')
                else:
                    shard_data = load_file(to_load, device='cpu')
                for k in layer_keys_needed:
                    if k in shard_data:
                        layer_state_dict[k] = shard_data[k]
                del shard_data
                clean_memory()

            layer_state_dict = compress_layer_state_dict(layer_state_dict, compression)

            ModelPersister.get_model_persister().persist_model(layer_state_dict, layer, saving_path)

            del layer_state_dict
            clean_memory()

    finally:
        if _single_shard_handle is not None:
            del _single_shard_handle
            clean_memory()

    # deleting single modelfile if only a single modelfile was existing in hf repo 
    # and deletion of single modelfile should happen in the end if delete_original=True
    if delete_original and single_modelfile != None:
        to_delete = checkpoint_path / single_modelfile
        print(f"deleting original file: {to_delete}")
        remove_real_and_linked_file(to_delete)

    return str(saving_path)

def find_or_create_local_splitted_path(model_local_path_or_repo_id, layer_shards_saving_path=None, compression=None,
                                       layer_names=None, hf_token=None, delete_original=False):
    """
    find the model's local cache path, download the cache if not exists, then split and save the model.

    Parameters
    ----------
    model_local_path_or_repo_id : str
        model local path or hf repo id
    layer_shards_saving_path : str, optional
        optional path to save the splitted model, by default directly under the model local path

    Returns
    -------
    model_local_path : str
        local model path
    saved_layer_shards_path : str
        the path saved layer shards
    compression: str, optinal
        setting to '4bit' or '8bit' to enable compression from 16 bits to 4 bits/8 bits which speeed up 4x or 2x inference time with a tiny accuracy loss.
    hf_token: str, optional
        huggingface api token could be provided, by default None
    """

    # try local model path, if the model exist split and save there
    if os.path.exists(model_local_path_or_repo_id):
        p = Path(model_local_path_or_repo_id)
        has_model = (
            os.path.exists(p / 'pytorch_model.bin.index.json') or
            os.path.exists(p / 'model.safetensors.index.json') or
            os.path.exists(p / 'model.safetensors') or          # single-shard (e.g. GPTQ)
            os.path.exists(p / 'pytorch_model.bin')             # single-shard legacy
        )
        if has_model:
            print(f"found local model files in {model_local_path_or_repo_id}...")
            return Path(model_local_path_or_repo_id), split_and_save_layers(model_local_path_or_repo_id, layer_shards_saving_path,
                                                                            compression=compression, layer_names=layer_names, delete_original=delete_original)
        else:
            print(
                f"Found local directory in {model_local_path_or_repo_id}, but didn't find downloaded model. Try using {model_local_path_or_repo_id} as a HF repo...")

    # it should be a repo id at this point...
    hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token,
        #allow_patterns= ["model.safetensors.index.json", 'pytorch_model.bin.index.json'],
        ignore_patterns=['*.safetensors', '*.bin'])


    # check if there's safetensors saved, if so, exclude torch saves
    # delay download now...
    '''
    hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token, allow_patterns="model.safetensors.index.json")
    if len(glob(str(Path(hf_cache_path) / "model.safetensors.index.json"))) > 0:
        # there's safe tensor version, exclude torch version
        hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token,
                                                          ignore_patterns=['pytorch_model.bin.index.json', '*.bin'])

    else:
        hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id,
                                                          token=hf_token)
    '''

    #assert os.path.exists(Path(hf_cache_path) / 'pytorch_model.bin.index.json') or \
    #       os.path.exists(Path(hf_cache_path) / 'model.safetensors.index.json'), \
    #       f"{hf_cache_path}/pytorch_model.bin.index.json or {hf_cache_path}/model.safetensors.index.json should exists."

    # if splitted_model subdir exists under cache use it, otherwise split and save
    return Path(hf_cache_path), split_and_save_layers(hf_cache_path, layer_shards_saving_path,
                                                      compression=compression, layer_names=layer_names,
                                                      delete_original=delete_original, repo_id=model_local_path_or_repo_id, hf_token=hf_token)
