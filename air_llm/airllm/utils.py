import gc
import json
import os
import re
import ctypes
import shutil
from tqdm import tqdm
from pathlib import Path
from glob import glob
import time

from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True


import torch
import torch.nn as nn
from safetensors import safe_open
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


# Qwen4-Exp / Flash-Next stores the ~51B PLE n-gram table as ``*.ngram_embedding.shard_N.weight``.
# Transformers concatenates those shards into a single ``nn.Embedding.weight`` at load time.
# On machines that cannot hold ~102GB of host RAM we instead stream the shards into a file-backed
# mmap and gather rows from disk (the modeling code already looks up on ``weight.device``).
_NGRAM_SHARD_RE = re.compile(r'^(?P<prefix>.*\.ngram_embedding)\.shard_(?P<idx>\d+)\.weight$')
_DTYPE_TAGS = {
    torch.bfloat16: 'bf16',
    torch.float16: 'fp16',
    torch.float32: 'fp32',
}
_TAG_DTYPES = {tag: dtype for dtype, tag in _DTYPE_TAGS.items()}


def merge_ngram_embedding_shards(state_dict):
    """Concatenate ``ngram_embedding.shard_N.weight`` tensors into ``ngram_embedding.weight``.

    Keys that are not n-gram shards pass through unchanged. Shard indices must be a contiguous
    0..N-1 range; a gap would silently drop rows from the hash table.
    """
    groups = {}
    out = {}
    for key, value in state_dict.items():
        match = _NGRAM_SHARD_RE.match(key)
        if match:
            groups.setdefault(match.group('prefix'), []).append((int(match.group('idx')), value))
        else:
            out[key] = value
    for prefix, shards in groups.items():
        shards.sort(key=lambda item: item[0])
        got = [idx for idx, _ in shards]
        expected = list(range(len(shards)))
        if got != expected:
            raise ValueError(
                f"ngram embedding shards for {prefix} are not contiguous 0..{len(shards) - 1}: {got}"
            )
        out[f'{prefix}.weight'] = torch.cat([tensor for _, tensor in shards], dim=0)
    return out


def load_merged_ngram_embedding(local_path, layer_name):
    """Load a split n-gram file and concatenate shards without holding two full copies.

    ``load_file`` plus ``torch.cat`` would peak at ~2x the table (~200GB for Flash-Next). This
    reads one shard at a time into a preallocated host tensor.
    """
    filepath = Path(local_path) / (layer_name + ".safetensors")
    with safe_open(str(filepath), framework="pt") as handle:
        groups = {}
        rest = {}
        for key in handle.keys():
            match = _NGRAM_SHARD_RE.match(key)
            if match:
                groups.setdefault(match.group('prefix'), []).append((int(match.group('idx')), key))
            else:
                rest[key] = handle.get_tensor(key)
        for prefix, shards in groups.items():
            shards.sort(key=lambda item: item[0])
            got = [idx for idx, _ in shards]
            expected = list(range(len(shards)))
            if got != expected:
                raise ValueError(
                    f"ngram embedding shards for {prefix} are not contiguous "
                    f"0..{len(shards) - 1}: {got}"
                )
            shapes = [tuple(handle.get_slice(key).get_shape()) for _, key in shards]
            first = handle.get_tensor(shards[0][1])
            dest = first.new_empty((sum(shape[0] for shape in shapes),) + shapes[0][1:])
            dest[0:shapes[0][0]].copy_(first)
            del first
            offset = shapes[0][0]
            for (_, key), shape in zip(shards[1:], shapes[1:]):
                piece = handle.get_tensor(key)
                dest[offset:offset + shape[0]].copy_(piece)
                offset += shape[0]
                del piece
            rest[f'{prefix}.weight'] = dest
    return rest


def ngram_mmap_exists(saving_path, layer_name):
    return (os.path.exists(str(saving_path / (layer_name + 'mmap')))
            and os.path.exists(str(saving_path / (layer_name + 'mmap.json')))
            and os.path.exists(str(saving_path / (layer_name + 'mmap.done'))))


def _ngram_shard_items(layer_prefix, index):
    items = []
    for key, filename in index.items():
        if not key.startswith(layer_prefix):
            continue
        match = _NGRAM_SHARD_RE.match(key)
        if not match:
            continue
        items.append((int(match.group('idx')), key, filename))
    items.sort(key=lambda item: item[0])
    got = [idx for idx, _, _ in items]
    if got != list(range(len(items))):
        raise ValueError(
            f"ngram embedding shards for {layer_prefix} are not contiguous "
            f"0..{len(items) - 1}: {got}"
        )
    return items


def persist_ngram_mmap(layer_prefix, index, checkpoint_path, saving_path,
                       repo_id=None, hf_token=None):
    """Stream n-gram shards into a file-backed mmap without holding the full table in RAM."""
    if ngram_mmap_exists(saving_path, layer_prefix):
        return
    items = _ngram_shard_items(layer_prefix, index)
    if not items:
        return

    first_file = Path(checkpoint_path) / items[0][2]
    if not first_file.exists():
        assert repo_id is not None
        huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(first_file),
                                          token=hf_token)

    open_handles = {}
    try:
        def _handle(filename):
            path = str(Path(checkpoint_path) / filename)
            if path not in open_handles:
                src = Path(checkpoint_path) / filename
                if not src.exists():
                    assert repo_id is not None
                    huggingface_hub.snapshot_download(
                        repo_id, allow_patterns=os.path.basename(src), token=hf_token)
                cm = safe_open(path, framework='pt')
                open_handles[path] = (cm, cm.__enter__())
            return open_handles[path][1]

        def _tensor(filename, key):
            return _handle(filename).get_tensor(key)

        first = _tensor(items[0][2], items[0][1])
        cols = first.shape[1]
        dtype = first.dtype
        if dtype not in _DTYPE_TAGS:
            raise ValueError(f"unsupported n-gram mmap dtype {dtype}")
        rows = first.shape[0]
        shapes_rest = []
        for _, key, filename in items[1:]:
            shapes_rest.append(tuple(_handle(filename).get_slice(key).get_shape()))
            rows += shapes_rest[-1][0]

        raw_path = Path(saving_path) / (layer_prefix + 'mmap')
        tmp_path = Path(saving_path) / (layer_prefix + 'mmap.tmp')
        nbytes = rows * cols * first.element_size()
        print(f"writing n-gram mmap ({rows} x {cols} {_DTYPE_TAGS[dtype]}, "
              f"{nbytes / (1024 ** 3):.1f}GB) to {raw_path}")
        with open(tmp_path, 'wb') as out:
            out.write(first.contiguous().view(torch.uint8).numpy().tobytes())
            del first
            for (_, key, filename), shape in zip(items[1:], shapes_rest):
                piece = _tensor(filename, key)
                if tuple(piece.shape) != shape:
                    raise ValueError(f"{key} shape {tuple(piece.shape)} != {shape}")
                out.write(piece.contiguous().view(torch.uint8).numpy().tobytes())
                del piece
        tmp_path.replace(raw_path)
        meta = {
            'rows': rows,
            'cols': cols,
            'dtype': _DTYPE_TAGS[dtype],
            'nbytes': nbytes,
        }
        with open(Path(saving_path) / (layer_prefix + 'mmap.json'), 'w') as f:
            json.dump(meta, f)
        (Path(saving_path) / (layer_prefix + 'mmap.done')).touch()
        print(f"saved n-gram mmap as: {raw_path}")
    finally:
        for cm, _handle_obj in open_handles.values():
            cm.__exit__(None, None, None)


def open_ngram_mmap_table(local_path, layer_name):
    """Map a persisted n-gram table without copying it into anonymous RAM."""
    meta_path = Path(local_path) / (layer_name + '.mmap.json')
    raw_path = Path(local_path) / (layer_name + '.mmap')
    meta = json.loads(meta_path.read_text())
    dtype = _TAG_DTYPES[meta['dtype']]
    rows, cols, nbytes = meta['rows'], meta['cols'], meta['nbytes']
    storage = torch.UntypedStorage.from_file(str(raw_path), shared=True, nbytes=nbytes)
    table = torch.tensor([], dtype=dtype)
    table.set_(storage, 0, (rows, cols))
    return table


class MmapEmbedding(nn.Module):
    """``nn.Embedding`` stand-in over a file-backed CPU table.

    The table is a plain attribute, not a parameter or buffer, so a parent decoder layer's
    ``module.to('meta')`` cannot evict it. A zero-size ``weight`` lives on CPU so modeling code
    that gathers on ``self.ngram_embedding.weight.device`` still runs the lookup on the host.
    """

    def __init__(self, table):
        super().__init__()
        if table.dim() != 2:
            raise ValueError(f"mmap embedding table must be 2D, got {tuple(table.shape)}")
        self.num_embeddings, self.embedding_dim = table.shape
        self._table = table
        self.weight = nn.Parameter(
            torch.empty(0, table.shape[1], dtype=table.dtype), requires_grad=False)

    def forward(self, input):
        return nn.functional.embedding(input, self._table)


@contextmanager
def _force_meta_embeddings():
    """Construct ``nn.Embedding`` on meta so huge tables never materialize on CPU.

    ``accelerate.init_empty_weights`` allocates each Parameter on CPU and then ``.to('meta')``.
    That is fine for ordinary weights, but Flash-Next's PLE table is
    ``nn.Embedding(~320M, 160)`` -- ~191GB of empty fp32 -- which OOMs a 64GB host before the
    move. Creating the module with ``device=meta`` keeps construction memory-free; AirLLM later
    replaces this module with a file-backed ``MmapEmbedding``.
    """
    orig = torch.nn.Embedding.__init__

    def _init(self, num_embeddings, embedding_dim, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault('device', torch.device('meta'))
        orig(self, num_embeddings, embedding_dim, *args, **kwargs)

    torch.nn.Embedding.__init__ = _init
    try:
        yield
    finally:
        torch.nn.Embedding.__init__ = orig


def _wrap_forward_int64_scatter(orig_forward):
    """Make ``tensor.scatter(..., index)`` accept int32 indices for one forward.

    Transformers' Qwen4-Exp sparse-attention indexer fills ``selected_token_indices`` as
    ``int32`` and scatters them into a bool mask. ``torch.Tensor.scatter`` requires ``int64``,
    so generate crashes on the first sparse-attention layer. The wrap is scoped to that
    forward so the rest of PyTorch is unchanged.
    """
    def forward(self, *args, **kwargs):
        orig_scatter = torch.Tensor.scatter

        def scatter(tensor, dim, index, *a, **kw):
            if torch.is_tensor(index) and index.dtype != torch.int64:
                index = index.to(torch.int64)
            return orig_scatter(tensor, dim, index, *a, **kw)

        torch.Tensor.scatter = scatter
        try:
            return orig_forward(self, *args, **kwargs)
        finally:
            torch.Tensor.scatter = orig_scatter

    return forward


def cpu_resident_module_names(layer_names, index_keys):
    """Module prefixes whose weights should stay on CPU after the split.

    ``cpu_resident`` is an explicit list. ``cpu_resident_marker`` (e.g. the Flash-Next PLE table)
    is scanned out of the checkpoint so we do not have to hard-code which decoder layer owns it.
    """
    names = list(layer_names.get('cpu_resident', []))
    marker = layer_names.get('cpu_resident_marker')
    if marker:
        found = []
        for key in index_keys:
            pos = key.find(marker)
            if pos == -1:
                continue
            prefix = key[:pos + len(marker)]
            if prefix not in names and prefix not in found:
                found.append(prefix)
        names.extend(found)
    return names


def layer_owner(key, layer_prefixes):
    """Longest matching prefix wins, so a nested cpu-resident module is not swallowed by its parent."""
    owner = None
    for prefix in layer_prefixes:
        if key.startswith(prefix) and (owner is None or len(prefix) > len(owner)):
            owner = prefix
    return owner


def layer_tensor_names(local_path, layer_name):
    """List the tensors in a layer shard without reading any tensor data."""
    with safe_open(str(Path(local_path) / (layer_name + ".safetensors")), framework="pt") as f:
        return list(f.keys())


def load_layer_subset(local_path, layer_name, keys):
    """Read only `keys` from a layer shard.

    safetensors can seek to individual tensors, so a single MoE expert costs its own few MB rather
    than the whole ~16GB layer file. That is what makes per-expert streaming worthwhile.
    """
    out = {}
    with safe_open(str(Path(local_path) / (layer_name + ".safetensors")), framework="pt") as f:
        for k in keys:
            out[k] = f.get_tensor(k)
    return out


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
    total_shard_files_size_bytes = 0
    for model_shard_file in glob(str(checkpoint_path / '*')):
        total_shard_files_size_bytes += os.path.getsize(model_shard_file)

    total_saved_split_files_size_bytes = 0
    if layer_shards_saving_path is not None:
        for saved_split_file in glob(str(Path(layer_shards_saving_path) / splitted_model_dir_name / '*')):
            total_saved_split_files_size_bytes += os.path.getsize(saved_split_file)

    if compression == '4bit':
        total_shard_files_size_bytes = int(total_shard_files_size_bytes / 0.2813)
    elif compression == '8bit':
        total_shard_files_size_bytes = total_shard_files_size_bytes // 2

    total, used, free = shutil.disk_usage(checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path)

    if free + total_saved_split_files_size_bytes < total_shard_files_size_bytes:
        raise NotEnoughSpaceException(f"Not enough space. Free space under {checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path}:"  \
                                      f" {free / 1024 / 1024 / 1024:.02f}GB. Model total size: {total_shard_files_size_bytes / 1024 / 1024 / 1024:.02f}GB. " \
                                      f"existing space under {checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path} assuming can reuse: {total_saved_split_files_size_bytes/ 1024 / 1024 / 1024:.02f}GB. "
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



def link_or_copy_file(src, dst):
    """Point dst at src's data without duplicating it, falling back to a real copy.

    A hard link is preferred over a symlink because it keeps the data alive even if the original
    checkpoint file is later deleted (``delete_original``), and because it costs no extra disk.
    Hard links need both paths on one filesystem, so we degrade to a symlink and finally to a copy.
    Hugging Face caches store files as symlinks into a blob dir, so we always link the real file.
    """
    src = Path(os.path.realpath(str(src)))
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    try:
        os.link(src, dst)
        return 'hardlink'
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return 'symlink'
    except OSError:
        pass
    shutil.copyfile(src, dst)
    return 'copy'


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

    if layer_names is None:
        n_layers = len(set([int(k.split('.')[2]) for k in index.keys() if 'model.layers' in k]))
    else:
        n_layers = len(set([int(k[len(layer_names['layer_prefix']):].split('.')[1]) for k in index.keys() if layer_names['layer_prefix'] in k]))

    if layer_names is None:
        layers = ['model.embed_tokens.'] + [f'model.layers.{i}.' for i in range(n_layers)] + ['model.norm.', 'lm_head.']
    else:
        layers = [layer_names['embed']] + [f'{layer_names["layer_prefix"]}.{i}' for i in range(n_layers)] + [layer_names['norm'], layer_names['lm_head']]

        if 'rotary_pos_emb' in layer_names:
            layers = [layer_names['rotary_pos_emb']] + layers
        # Modules that are not part of the streamed sequence but still need their weights on disk,
        # e.g. a multimodal model's vision tower / projector, or extra top-level norms. They get
        # their own shard and are loaded once and kept resident.
        layers = layers + list(layer_names.get('resident', []))
        layers = layers + cpu_resident_module_names(layer_names, index.keys())
        layers = [l + "." for l in layers]

    # Drop layers that have no weights in the checkpoint. This happens for tied embeddings,
    # where lm_head shares storage with embed_tokens and has no entry of its own. Without this we
    # would try to save an empty shard (which fails) and never detect the split as complete.
    layers = [l for l in layers if any(k.startswith(l) for k in index.keys())]
    owned = {layer_owner(k, layers) for k in index}
    layers = [l for l in layers if l in owned]

    # Split in ascending shard order. The loop below only ever walks the shard counter forward, so
    # a module whose weights sit in an earlier shard than its predecessor's would silently be saved
    # incomplete. That ordering isn't guaranteed once non-sequential modules (a vision tower, extra
    # norms) are in the list, so sort by the last shard each module touches. This is a stable sort,
    # so plain embed -> layers -> norm -> lm_head checkpoints keep their existing order.
    # When two modules share a last shard, the longer prefix goes first so a nested cpu-resident
    # table is extracted before its parent decoder layer can swallow it.
    def _last_shard_of(layer):
        nums = [int(v.split('-')[1]) for k, v in index.items()
                if k.startswith(layer) and '-' in v and len(v.split('-')) > 1]
        return max(nums) if nums else -1

    layers.sort(key=lambda layer: (_last_shard_of(layer), -len(layer)))


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

    # Some checkpoints are already sharded exactly one module per file (Kimi K3, for instance, ships
    # one ~17GB shard per decoder layer). Re-writing those into per-layer files would duplicate the
    # entire checkpoint on disk -- 1.5TB+ for a 2.8T-parameter model -- and take hours, to produce
    # byte-identical content. When a shard holds nothing but one module's tensors we link to it
    # instead of copying.
    passthrough = {}
    # Linking only produces a file the loader can read when shards are stored in the same format
    # the persister writes; the MLX persister, for instance, writes .mlx.npz.
    persister_is_safetensors = type(ModelPersister.get_model_persister()).__name__ == 'SafetensorModelPersister'
    if compression is None and safetensors_format and persister_is_safetensors:
        shard_contents = defaultdict(list)
        for k, v in index.items():
            shard_contents[v].append(k)
        for layer in layers:
            files = {v for k, v in index.items() if k.startswith(layer)}
            if len(files) != 1:
                continue
            only_file = next(iter(files))
            if all(k.startswith(layer) for k in shard_contents[only_file]):
                passthrough[layer] = only_file

    if passthrough:
        print(f"{len(passthrough)}/{len(layers)} modules are already one-per-shard; "
              f"linking to the original files instead of copying them.")

    # Must exist before check_space, which stats the filesystem it lives on.
    saving_path.mkdir(parents=True, exist_ok=True)

    # A copy is only made for the layers we cannot link, so only those need free space.
    if not delete_original and len(passthrough) < len(layers):
        check_space(checkpoint_path, layer_shards_saving_path, compression, splitted_model_dir_name=splitted_model_dir_name)


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

        marker = (layer_names or {}).get('cpu_resident_marker')
        if marker and marker in layer:
            persist_ngram_mmap(layer, index, checkpoint_path, saving_path,
                               repo_id=repo_id, hf_token=hf_token)
            continue

        if layer in passthrough:
            src = checkpoint_path / passthrough[layer]
            if not os.path.exists(src):
                assert repo_id is not None
                huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(src),
                                                  token=hf_token)
            if not ModelPersister.get_model_persister().model_persist_exist(layer, saving_path):
                link_or_copy_file(src, saving_path / (layer + 'safetensors'))
                (saving_path / (layer + 'safetensors.done')).touch()
            # Keep the shard cursor in step with what we skipped, so a later layer that does need
            # loading doesn't walk back through (and read) every shard we just linked past.
            src_parts = passthrough[layer].split('-')
            if len(src_parts) > 1:
                try:
                    shard = max(shard, int(src_parts[1]))
                except ValueError:
                    pass
            continue

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
                                                    token=hf_token)

                if not safetensors_format:
                    loaded = torch.load(to_load, map_location='cpu')
                else:
                    loaded = load_file(to_load, device='cpu')
                marker = (layer_names or {}).get('cpu_resident_marker')
                if marker:
                    loaded = {k: v for k, v in loaded.items() if marker not in k}
                state_dict.update(loaded)

        else:
            shards = [v for k, v in index.items() if k.startswith(layer)]
            single_modelfile = shards[0]
            to_load = checkpoint_path / single_modelfile
            # check if to_load exist, if not downloaad it...
            if not os.path.exists(to_load):
                assert repo_id is not None
                huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(to_load),
                                                token=hf_token)
            if not safetensors_format:
                loaded = torch.load(to_load, map_location='cpu')
            else:
                loaded = load_file(to_load, device='cpu')
            marker = (layer_names or {}).get('cpu_resident_marker')
            if marker:
                loaded = {k: v for k, v in loaded.items() if marker not in k}
            state_dict.update(loaded)

        # Get layer state dict. Longest prefix wins, so e.g. a PLE n-gram table nested under a
        # decoder layer is not written into that layer's shard (and later streamed onto the GPU).
        layer_state_dict = dict([(k, v) for k, v in state_dict.items()
                                 if layer_owner(k, layers) == layer])

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
        # Accept single-file checkpoints too, not just sharded ones with an index: the splitter
        # handles both, so requiring an index needlessly sent local single-file models down the
        # "treat it as a repo id" path, where they fail as an invalid repo name.
        local_weight_files = ('pytorch_model.bin.index.json', 'model.safetensors.index.json',
                              'model.safetensors', 'pytorch_model.bin')
        if any(os.path.exists(Path(model_local_path_or_repo_id) / f) for f in local_weight_files):
            print(f"found local checkpoint...")
            return Path(model_local_path_or_repo_id), split_and_save_layers(model_local_path_or_repo_id, layer_shards_saving_path,
                                                                            compression=compression, layer_names=layer_names, delete_original=delete_original)
        else:
            print(
                f"Found local directory in {model_local_path_or_repo_id}, but didn't find downloaded model. Try using {model_local_path_or_repo_id} as a HF repo...")

    # it should be a repo id at this point...
    # First grab everything except the (potentially huge) weight files. For multi-shard models the
    # index.json tells us the structure and we stream each shard on demand during splitting.
    hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token,
        #allow_patterns= ["model.safetensors.index.json", 'pytorch_model.bin.index.json'],
        ignore_patterns=['*.safetensors', '*.bin'])

    # Single-file checkpoints have no index.json, so there's nothing to stream on demand and we
    # can't infer the structure without the file itself. Download the single weight file now.
    has_index = os.path.exists(Path(hf_cache_path) / 'model.safetensors.index.json') or \
                os.path.exists(Path(hf_cache_path) / 'pytorch_model.bin.index.json')
    if not has_index:
        hf_cache_path = huggingface_hub.snapshot_download(
            model_local_path_or_repo_id, token=hf_token,
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
                                                      delete_original=delete_original, repo_id=model_local_path_or_repo_id, hf_token=hf_token)
