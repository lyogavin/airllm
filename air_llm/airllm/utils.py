import gc
import importlib
import importlib.util
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


bnb = None


def require_bitsandbytes():
    """Return a usable bitsandbytes module or fail before model download starts."""
    global bnb
    if importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError("Compression requires bitsandbytes: `pip install bitsandbytes`")
    if bnb is None:
        bnb = importlib.import_module("bitsandbytes")
    native_library = getattr(getattr(bnb, "cextension", None), "lib", None)
    if native_library is None or type(native_library).__name__ == "ErrorHandlerMockBNBNativeLibrary":
        raise ImportError(
            "bitsandbytes is installed, but its native CUDA/ROCm library is unavailable for this platform. "
            "Use a compatible wheel/build or a checkpoint that is already quantized."
        )
    return bnb


def bitsandbytes_available():
    try:
        require_bitsandbytes()
        return True
    except (ImportError, RuntimeError):
        return False


import huggingface_hub


COMPRESSION_SIZE_RATIOS = {
    None: 1.0,
    "8bit": 0.5,
    # NF4 packs two values per byte and stores one fp32 absmax per 64 values.
    "4bit": 0.2813,
}


def estimated_split_size_bytes(model_size_bytes, compression=None):
    """Estimate the final layer-shard footprint for an AirLLM compression mode."""
    if compression not in COMPRESSION_SIZE_RATIOS:
        raise ValueError(f"Unsupported compression mode: {compression!r}")
    return int(model_size_bytes * COMPRESSION_SIZE_RATIOS[compression])


def checkpoint_total_size_bytes(checkpoint_path):
    """Return checkpoint weight size, preferring index metadata when available.

    Hugging Face downloads the index before the large weight files. Reading
    ``metadata.total_size`` lets us reject an undersized offload volume before
    hundreds of gigabytes have been downloaded.
    """
    checkpoint_path = Path(checkpoint_path)
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint_path / index_name
        if index_path.exists():
            with open(index_path, "rb") as f:
                metadata = json.load(f).get("metadata", {})
            total_size = metadata.get("total_size")
            if total_size is not None:
                return int(total_size)

    total = 0
    for pattern in ("*.safetensors", "*.bin"):
        for model_file in checkpoint_path.glob(pattern):
            total += model_file.stat().st_size
    return total


def checkpoint_quantization_method(checkpoint_path):
    config_path = Path(checkpoint_path) / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, "rb") as f:
        config = json.load(f)
    text_config = config.get("text_config") or {}
    quantization_config = config.get("quantization_config") or text_config.get("quantization_config")
    if isinstance(quantization_config, dict):
        return quantization_config.get("quant_method") or quantization_config.get("format") or "pre-quantized"
    return None


def configured_decoder_layer_count(checkpoint_path):
    """Read the inference decoder count, excluding auxiliary MTP layers."""
    config_path = Path(checkpoint_path) / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, "rb") as f:
        config = json.load(f)
    text_config = config.get("text_config") or {}
    value = text_config.get("num_hidden_layers", config.get("num_hidden_layers"))
    return int(value) if value is not None else None


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
    bnb_module = require_bitsandbytes()
    qs_packed_dict["quant_state." + "bitsandbytes__" + self.quant_type] = bnb_module.utils.pack_dict_to_tensor(non_tensor_dict)
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


def uncompress_layer_state_dict(layer_state_dict):
    uncompressed_layer_state_dict = None
    if any(['4bit' in k for k in layer_state_dict.keys()]):
        bnb_module = require_bitsandbytes()
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if '4bit' not in k:
                quant_state_dict = {kk[len(k):]: kv for kk, kv in layer_state_dict.items() if kk.startswith(k) and k != kk}
                quant_state = bnb_module.functional.QuantState.from_dict(qs_dict=quant_state_dict, device="cuda")

                dqv = bnb_module.functional.dequantize_nf4(v.cuda(), quant_state)
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict
    elif any(['8bit' in k for k in layer_state_dict.keys()]):
        bnb_module = require_bitsandbytes()
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if '8bit' not in k:

                absmax = layer_state_dict[k + ".8bit.absmax"]
                code = layer_state_dict[k + ".8bit.code"]

                dqv = bnb_module.functional.dequantize_blockwise(v.cuda(),
                                                          bnb_module.functional.QuantState(absmax=absmax.cuda(),
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
    checkpoint_path = Path(checkpoint_path)
    total_shard_files_size_bytes = checkpoint_total_size_bytes(checkpoint_path)
    required_split_size_bytes = estimated_split_size_bytes(total_shard_files_size_bytes, compression)

    total_saved_split_files_size_bytes = 0
    if layer_shards_saving_path is not None:
        for saved_split_file in glob(str(Path(layer_shards_saving_path) / splitted_model_dir_name / '*')):
            total_saved_split_files_size_bytes += os.path.getsize(saved_split_file)

    capacity_path = checkpoint_path if layer_shards_saving_path is None else Path(layer_shards_saving_path)
    while not capacity_path.exists() and capacity_path != capacity_path.parent:
        capacity_path = capacity_path.parent
    if not capacity_path.exists():
        raise FileNotFoundError(f"Offload volume does not exist: {layer_shards_saving_path}")
    total, used, free = shutil.disk_usage(capacity_path)

    if free + total_saved_split_files_size_bytes < required_split_size_bytes:
        raise NotEnoughSpaceException(f"Not enough space. Free space under {checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path}:"  \
                                      f" {free / 1024 / 1024 / 1024:.02f}GB. Estimated split size: {required_split_size_bytes / 1024 / 1024 / 1024:.02f}GB. " \
                                      f"existing space under {checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path} assuming can reuse: {total_saved_split_files_size_bytes/ 1024 / 1024 / 1024:.02f}GB. "
                                      )

def compress_layer_state_dict(layer_state_dict, compression=None):
    compressed_layer_state_dict = None
    if compression == '4bit':
        bnb_module = require_bitsandbytes()
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb_module.functional.quantize_nf4(v.cuda(), blocksize=64)
            compressed_layer_state_dict[k] = v_quant
            for quant_state_k, quant_state_v in save_quant_state_to_dict(quant_state).items():
                compressed_layer_state_dict[k + ".4bit." + quant_state_k] = quant_state_v
    elif compression == '8bit':
        bnb_module = require_bitsandbytes()
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb_module.functional.quantize_blockwise(v.cuda(), blocksize=2048)
            absmax = quant_state.absmax.clone().contiguous()
            code = quant_state.code.clone().contiguous()
            compressed_layer_state_dict[k] = v_quant
            compressed_layer_state_dict[k + ".8bit.absmax"] = absmax
            compressed_layer_state_dict[k + ".8bit.code"] = code

    return compressed_layer_state_dict if compressed_layer_state_dict is not None else layer_state_dict

def remove_real_and_linked_file(to_delete):
    targetpath = None
    if (os.path.realpath(to_delete) != to_delete):
        targetpath = os.path.realpath(to_delete)

    os.remove(to_delete)
    if targetpath and os.path.exists(targetpath):
         os.remove(targetpath)



def split_and_save_layers(checkpoint_path, layer_shards_saving_path=None, splitted_model_dir_name='splitted_model',
                          compression=None, layer_names=None, delete_original=False, repo_id=None, hf_token=None,
                          cache_dir=None):
    """
    Save the all layers of a model sharded checkpoint using safetensors.
    """

    if compression is not None:
        require_bitsandbytes()
        quantization_method = checkpoint_quantization_method(checkpoint_path)
        if quantization_method is not None:
            raise ValueError(
                f"AirLLM compression cannot be stacked on a pre-quantized checkpoint ({quantization_method})."
            )
        splitted_model_dir_name = splitted_model_dir_name + "." + compression

    checkpoint_path = Path(checkpoint_path)


    saving_path = checkpoint_path / splitted_model_dir_name

    if layer_shards_saving_path is not None:
        saving_path = Path(layer_shards_saving_path) / splitted_model_dir_name


    # Build a weight_map (param name -> file that stores it). Multi-shard checkpoints ship an
    # index.json; small/modern models often ship a single file with no index, so synthesize one.
    safetensors_format = False
    if os.path.exists(checkpoint_path / 'pytorch_model.bin.index.json'):
        with open(checkpoint_path / 'pytorch_model.bin.index.json', 'rb') as f:
            index = json.load(f)['weight_map']
    elif os.path.exists(checkpoint_path / 'model.safetensors.index.json'):
        safetensors_format = True
        with open(checkpoint_path / 'model.safetensors.index.json', 'rb') as f:
            index = json.load(f)['weight_map']
    elif os.path.exists(checkpoint_path / 'model.safetensors'):
        # single-file safetensors checkpoint: map every tensor to that one file
        safetensors_format = True
        from safetensors import safe_open
        with safe_open(str(checkpoint_path / 'model.safetensors'), framework='pt') as f:
            index = {k: 'model.safetensors' for k in f.keys()}
    elif os.path.exists(checkpoint_path / 'pytorch_model.bin'):
        # single-file torch checkpoint: map every tensor to that one file
        safetensors_format = False
        single_sd = torch.load(checkpoint_path / 'pytorch_model.bin', map_location='cpu')
        index = {k: 'pytorch_model.bin' for k in single_sd.keys()}
        del single_sd
    else:
        raise FileNotFoundError(
            f"No model weights found under {checkpoint_path}. Expected one of: "
            f"model.safetensors(.index.json) or pytorch_model.bin(.index.json).")

    configured_layers = configured_decoder_layer_count(checkpoint_path)
    if configured_layers is not None:
        n_layers = configured_layers
    elif layer_names is None:
        n_layers = len(set([int(k.split('.')[2]) for k in index.keys() if 'model.layers' in k]))
    else:
        n_layers = len(set([int(k[len(layer_names['layer_prefix']):].split('.')[1]) for k in index.keys() if layer_names['layer_prefix'] in k]))

    if layer_names is None:
        layers = ['model.embed_tokens.'] + [f'model.layers.{i}.' for i in range(n_layers)] + ['model.norm.', 'lm_head.']
    else:
        layers = [layer_names['embed']] + [f'{layer_names["layer_prefix"]}.{i}' for i in range(n_layers)] + [layer_names['norm'], layer_names['lm_head']]

        if 'rotary_pos_emb' in layer_names:
            layers = [layer_names['rotary_pos_emb']] + layers
        layers = [l + "." for l in layers]

    # Drop layers that have no weights in the checkpoint. This happens for tied embeddings,
    # where lm_head shares storage with embed_tokens and has no entry of its own. Without this we
    # would try to save an empty shard (which fails) and never detect the split as complete.
    layers = [l for l in layers if any(k.startswith(l) for k in index.keys())]


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

    saving_path.mkdir(parents=True, exist_ok=True)

    # The split itself must fit even when original checkpoint shards are deleted as we go.
    check_space(checkpoint_path, layer_shards_saving_path, compression,
                splitted_model_dir_name=splitted_model_dir_name)


    shard = 0
    n_shards = len(set(index.values()))
    state_dict = {}

    # Map shard ordinal -> actual checkpoint filename, taken straight from the index. We must NOT
    # reconstruct names like f"model-000{n:02d}-of-000{n_shards:02d}.safetensors": repos differ in
    # zero-padding width (e.g. DeepSeek uses model-00001-of-000004.safetensors) and in extension.
    shard_num_to_file = {}
    for v in set(index.values()):
        parts = v.split('-')
        if len(parts) > 1:
            try:
                shard_num_to_file[int(parts[1])] = v
            except ValueError:
                pass

    single_modelfile = None

    for layer in tqdm(layers):

        # Optionnally load next shard
        # checking whether after spliting from '-', if second element exists. otherwise it throws errors for single 'model.safetensor' files
        shards = [int(v.split('-')[1]) for k, v in index.items() if k.startswith(layer) and '-' in v and len(v.split('-')) > 1]
        if len(shards) > 0:
            # A layer can span several shards (especially fp8 checkpoints, where each weight has a
            # companion weight_scale_inv tensor). Load *every* shard up to the highest one this layer
            # references, not just the next one -- otherwise the layer is saved missing some tensors
            # (e.g. the block scales), which silently corrupts fp8 weights.
            while max(shards) > shard:
                # optionally delete the original file we're done with (its tensors are already in RAM)
                if delete_original and shard != 0:
                    to_delete = checkpoint_path / shard_num_to_file[shard]

                    print(f"deleting original file: {to_delete}")
                    remove_real_and_linked_file(to_delete)
                shard += 1
                print(f'Loading shard {shard}/{n_shards}')

                to_load = checkpoint_path / shard_num_to_file[shard]

                # check if to_load exist, if not downloaad it...
                if not os.path.exists(to_load):
                    assert repo_id is not None
                    huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(to_load),
                                                    token=hf_token, cache_dir=cache_dir)

                if not safetensors_format:
                    state_dict.update(torch.load(to_load, map_location='cpu'))
                else:
                    state_dict.update(load_file(to_load, device='cpu'))

        else:
            shards = [v for k, v in index.items() if k.startswith(layer)]
            single_modelfile = shards[0]
            to_load = checkpoint_path / single_modelfile
            # check if to_load exist, if not downloaad it...
            if not os.path.exists(to_load):
                assert repo_id is not None
                huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(to_load),
                                                token=hf_token, cache_dir=cache_dir)
            if not safetensors_format:
                state_dict.update(torch.load(to_load, map_location='cpu'))
            else:
                state_dict.update(load_file(to_load, device='cpu'))

        # Get layer state dict
        layer_state_dict = dict([(k, v) for k, v in state_dict.items() if k.startswith(layer)])

        layer_state_dict = compress_layer_state_dict(layer_state_dict, compression)

        # Save layer state dict as using safetensors

        marker_exists = ModelPersister.get_model_persister().model_persist_exist(layer, saving_path)
        if not marker_exists:
            ModelPersister.get_model_persister().persist_model(layer_state_dict, layer, saving_path)

        # Free memory
        for k in layer_state_dict.keys():
            if k in state_dict:
                del state_dict[k]
        del layer_state_dict
        clean_memory()

    # deleting single modelfile if only a single modelfile was existing in hf repo 
    # and deletion of single modelfile should happen in the end if delete_original=True
    if delete_original and single_modelfile != None:
        to_delete = checkpoint_path / single_modelfile
        print(f"deleting original file: {to_delete}")
        remove_real_and_linked_file(to_delete)
    elif delete_original:
        # The rolling deletion above removes a shard only when the next shard is
        # opened. Remove the final shard (and any non-sequential leftovers) after
        # every requested inference layer has been persisted successfully.
        for checkpoint_file in set(index.values()):
            to_delete = checkpoint_path / checkpoint_file
            if to_delete.exists():
                print(f"deleting original file: {to_delete}")
                remove_real_and_linked_file(to_delete)

    return str(saving_path)

def find_or_create_local_splitted_path(model_local_path_or_repo_id, layer_shards_saving_path=None, compression=None,
                                       layer_names=None, hf_token=None, delete_original=False, cache_dir=None):
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
    cache_dir: str, optional
        Hugging Face download cache. Set this together with
        ``layer_shards_saving_path`` when the system drive is too small.
    """

    # try local model path, if the model exist split and save there
    if os.path.exists(model_local_path_or_repo_id):
        local_model_path = Path(model_local_path_or_repo_id)
        local_weight_files = (
            'pytorch_model.bin.index.json',
            'model.safetensors.index.json',
            'pytorch_model.bin',
            'model.safetensors',
        )
        if any((local_model_path / filename).exists() for filename in local_weight_files):
            print(f"found local model weights...")
            return Path(model_local_path_or_repo_id), split_and_save_layers(model_local_path_or_repo_id, layer_shards_saving_path,
                                                                            compression=compression, layer_names=layer_names, delete_original=delete_original)
        else:
            print(
                f"Found local directory in {model_local_path_or_repo_id}, but didn't find downloaded model. Try using {model_local_path_or_repo_id} as a HF repo...")

    # it should be a repo id at this point...
    # First grab everything except the (potentially huge) weight files. For multi-shard models the
    # index.json tells us the structure and we stream each shard on demand during splitting.
    hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token,
        cache_dir=cache_dir,
        #allow_patterns= ["model.safetensors.index.json", 'pytorch_model.bin.index.json'],
        ignore_patterns=['*.safetensors', '*.bin'])

    # Single-file checkpoints have no index.json, so there's nothing to stream on demand and we
    # can't infer the structure without the file itself. Download the single weight file now.
    has_index = os.path.exists(Path(hf_cache_path) / 'model.safetensors.index.json') or \
                os.path.exists(Path(hf_cache_path) / 'pytorch_model.bin.index.json')
    if not has_index:
        hf_cache_path = huggingface_hub.snapshot_download(
            model_local_path_or_repo_id, token=hf_token,
            cache_dir=cache_dir,
            allow_patterns=['model.safetensors', 'pytorch_model.bin'])


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
                                                      delete_original=delete_original, repo_id=model_local_path_or_repo_id,
                                                      hf_token=hf_token, cache_dir=cache_dir)
