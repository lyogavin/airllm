"""Disk and peak-VRAM estimates that do not download model weight shards."""

import importlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import huggingface_hub

from .utils import cpu_resident_module_names, layer_owner

_WEIGHT_EXTS = ('.safetensors', '.bin')
_INDEX_NAMES = ('model.safetensors.index.json', 'pytorch_model.bin.index.json')
_SINGLE_WEIGHTS = ('model.safetensors', 'pytorch_model.bin')
_VRAM_OVERHEAD_BYTES = 256 * 1024 * 1024


def _fmt_bytes(n):
    n = float(n or 0)
    for unit, scale in (('GB', 1024 ** 3), ('MB', 1024 ** 2), ('KB', 1024)):
        if abs(n) >= scale or unit == 'KB':
            return f"{n / scale:.2f} {unit}"
    return f"{int(n)} B"


def _dtype_nbytes(dtype):
    if dtype is None:
        return 2
    text = str(dtype).replace('torch.', '').lower()
    if text in ('float32', 'fp32'):
        return 4
    if text in ('float16', 'fp16', 'bfloat16', 'bf16'):
        return 2
    if 'int8' in text or 'fp8' in text or 'float8' in text:
        return 1
    if 'int4' in text or 'nf4' in text or 'fp4' in text:
        return 1
    return 2


def _split_factor(compression):
    if compression == '4bit':
        return 0.28
    if compression == '8bit':
        return 0.5
    return 1.0


def _dir_size(path):
    total = 0
    files = 0
    path = Path(path)
    if not path.exists():
        return 0, 0
    for p in path.rglob('*'):
        if p.is_file():
            total += p.stat().st_size
            files += 1
    return total, files


def _read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _local_weight_files(checkpoint_path):
    files = {}
    for name in os.listdir(checkpoint_path):
        lower = name.lower()
        if lower.endswith(_WEIGHT_EXTS) and 'index' not in lower:
            files[name] = os.path.getsize(Path(checkpoint_path) / name)
    return files


def _load_weight_map(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    for index_name in _INDEX_NAMES:
        index_path = checkpoint_path / index_name
        if index_path.exists():
            payload = _read_json(index_path)
            return payload.get('weight_map') or {}, payload.get('metadata') or {}
    for single in _SINGLE_WEIGHTS:
        single_path = checkpoint_path / single
        if single_path.exists():
            try:
                from safetensors import safe_open
                with safe_open(str(single_path), framework='pt') as handle:
                    return {k: single for k in handle.keys()}, {}
            except Exception:
                return {single: single}, {}
    return {}, {}


def _layer_prefixes(weight_map, layer_names=None):
    keys = list(weight_map.keys())
    if not keys:
        return []
    if layer_names is None:
        layer_ids = set()
        for k in keys:
            if 'model.layers' in k:
                try:
                    layer_ids.add(int(k.split('.')[2]))
                except (IndexError, ValueError):
                    pass
        n_layers = len(layer_ids)
        layers = (['model.embed_tokens.']
                  + [f'model.layers.{i}.' for i in range(n_layers)]
                  + ['model.norm.', 'lm_head.'])
    else:
        prefix = layer_names['layer_prefix']
        layer_ids = set()
        for k in keys:
            if prefix in k:
                try:
                    layer_ids.add(int(k[len(prefix):].split('.')[1]))
                except (IndexError, ValueError):
                    pass
        n_layers = len(layer_ids)
        layers = ([layer_names['embed']]
                  + [f'{prefix}.{i}' for i in range(n_layers)]
                  + [layer_names['norm'], layer_names['lm_head']])
        if 'rotary_pos_emb' in layer_names:
            layers = [layer_names['rotary_pos_emb']] + layers
        layers = layers + list(layer_names.get('resident', []))
        layers = layers + cpu_resident_module_names(layer_names, keys)
        layers = [name + '.' for name in layers]
    layers = [name for name in layers if any(k.startswith(name) for k in keys)]
    owned = {layer_owner(k, layers) for k in weight_map}
    return [name for name in layers if name in owned]


def _layer_bytes_from_files(weight_map, file_sizes, layers):
    file_owners = defaultdict(set)
    owner_file_tensors = defaultdict(lambda: defaultdict(int))
    for key, fname in weight_map.items():
        owner = layer_owner(key, layers) if layers else None
        if owner is None:
            continue
        file_owners[fname].add(owner)
        owner_file_tensors[owner][fname] += 1

    layer_bytes = defaultdict(int)
    for fname, size in file_sizes.items():
        owners = file_owners.get(fname)
        if not owners:
            continue
        if len(owners) == 1:
            layer_bytes[next(iter(owners))] += size
            continue
        total_tensors = sum(owner_file_tensors[owner][fname] for owner in owners)
        if total_tensors <= 0:
            continue
        for owner in owners:
            n = owner_file_tensors[owner][fname]
            layer_bytes[owner] += int(size * n / total_tensors)
    return dict(layer_bytes)


def _read_config(checkpoint_path=None, repo_id=None, hf_token=None):
    if checkpoint_path is not None:
        cfg_path = Path(checkpoint_path) / 'config.json'
        if cfg_path.exists():
            return _read_json(cfg_path)
    if repo_id:
        try:
            path = huggingface_hub.hf_hub_download(repo_id, 'config.json', token=hf_token)
            return _read_json(path)
        except Exception:
            return {}
    return {}


def _kv_cache_bytes(config, max_seq_len, batch=1):
    hidden = config.get('hidden_size') or config.get('n_embd') or config.get('d_model')
    n_layers = (config.get('num_hidden_layers') or config.get('n_layer')
                or config.get('num_layers') or config.get('n_layers'))
    n_heads = config.get('num_attention_heads') or config.get('n_head')
    n_kv = config.get('num_key_value_heads') or n_heads
    if not all((hidden, n_layers, n_heads, n_kv)):
        return 0
    head_dim = config.get('head_dim') or (int(hidden) // int(n_heads))
    nbytes = _dtype_nbytes(config.get('torch_dtype'))
    return int(n_layers) * 2 * int(batch) * int(max_seq_len) * int(n_kv) * int(head_dim) * nbytes


def _hub_file_sizes(repo_id, hf_token=None):
    info = huggingface_hub.HfApi().model_info(repo_id, files_metadata=True, token=hf_token)
    files = {}
    for sibling in info.siblings or []:
        name = getattr(sibling, 'rfilename', None)
        size = getattr(sibling, 'size', None)
        if not name or size is None:
            continue
        lower = name.lower()
        if lower.endswith(_WEIGHT_EXTS) and 'index' not in lower:
            files[name] = int(size)
    return files


def _hub_weight_map(repo_id, hf_token=None):
    for index_name in _INDEX_NAMES:
        try:
            path = huggingface_hub.hf_hub_download(repo_id, index_name, token=hf_token)
        except Exception:
            continue
        payload = _read_json(path)
        return payload.get('weight_map') or {}, payload.get('metadata') or {}, Path(path).parent
    return {}, {}, None


def _prefer_safetensors(file_sizes):
    safetensors = {k: v for k, v in file_sizes.items() if k.endswith('.safetensors')}
    if safetensors:
        return safetensors
    return file_sizes


def _split_dir_name(compression):
    if compression:
        return f'splitted_model.{compression}'
    return 'splitted_model'


def _find_split_cache(checkpoint_path, layer_shards_saving_path, compression):
    name = _split_dir_name(compression)
    candidates = []
    if layer_shards_saving_path:
        candidates.append(Path(layer_shards_saving_path) / name)
    if checkpoint_path:
        candidates.append(Path(checkpoint_path) / name)
    for path in candidates:
        if path.exists():
            size, n_files = _dir_size(path)
            return str(path), size, n_files
    return None, 0, 0


def _arch_layer_names(arch):
    """Layer-name dict from the streaming class registered in ``ARCH_OVERRIDES``."""
    from .auto_model import ARCH_OVERRIDES
    cls_name = ARCH_OVERRIDES.get(arch or "")
    stem = {
        "AirLLMChatGLM": "chatglm", "AirLLMQWen": "qwen", "AirLLMBaichuan": "baichuan",
        "AirLLMInternLM": "internlm", "AirLLMKimiK3": "kimi_k3",
        "AirLLMQwen3_5": "qwen3_5", "AirLLMQwen4Exp": "qwen4_exp",
    }.get(cls_name or "")
    if not stem:
        return None
    cls = getattr(importlib.import_module(f".airllm_{stem}", __package__), cls_name)
    obj = cls.__new__(cls)
    obj.set_layer_names_dict()
    return obj.layer_names_dict


def estimate_resources(model_local_path_or_repo_id, layer_shards_saving_path=None,
                       compression=None, hf_token=None, layer_names=None,
                       max_seq_len=512, prefetching=True):
    """Estimate checkpoint disk, split-copy disk, layer-cache size, and peak VRAM.

    Weight shards are not downloaded. For a Hub repo this uses file metadata plus
    the tiny ``config.json`` / index files. For a local checkpoint it stats files
    already on disk.

    Returns a dict. Pretty-print with :func:`format_resource_estimate`.
    """
    notes = []
    checkpoint_path = None
    repo_id = None
    source = 'hub'
    file_sizes = {}
    weight_map = {}
    metadata = {}

    if os.path.exists(model_local_path_or_repo_id):
        local = Path(model_local_path_or_repo_id)
        has_weights = any((local / name).exists() for name in _INDEX_NAMES + _SINGLE_WEIGHTS)
        if has_weights:
            source = 'local'
            checkpoint_path = local
            file_sizes = _prefer_safetensors(_local_weight_files(local))
            weight_map, metadata = _load_weight_map(local)
        else:
            repo_id = model_local_path_or_repo_id
            notes.append(f"Found {local} but no checkpoint files; treating it as a Hub repo id.")
    else:
        repo_id = model_local_path_or_repo_id

    if source == 'hub':
        file_sizes = _prefer_safetensors(_hub_file_sizes(repo_id, hf_token=hf_token))
        weight_map, metadata, snap = _hub_weight_map(repo_id, hf_token=hf_token)
        checkpoint_path = snap

    if not file_sizes and metadata.get('total_size'):
        file_sizes = {'_index_total': int(metadata['total_size'])}

    config = _read_config(checkpoint_path=checkpoint_path, repo_id=repo_id, hf_token=hf_token)
    # Nested VL checkpoints keep the decoder shape under text_config / llm_config.
    config = {**config, **(config.get('text_config') or config.get('llm_config') or {})}
    if layer_names is None:
        layer_names = _arch_layer_names((config.get('architectures') or [None])[0])

    checkpoint_bytes = int(sum(file_sizes.values()))
    n_shards = len(file_sizes)
    layers = _layer_prefixes(weight_map, layer_names=layer_names) if weight_map else []
    layer_bytes = _layer_bytes_from_files(weight_map, file_sizes, layers) if layers else {}
    n_modules = len(layers)

    n_layers = (config.get('num_hidden_layers') or config.get('n_layer')
                or config.get('num_layers') or 0)
    if not n_layers and layers:
        decoder = [name for name in layers if name.strip('.').rsplit('.', 1)[-1].isdigit()]
        n_layers = len(decoder) or n_modules

    cpu, resident = set(), set()
    if layer_names:
        cpu = {n.rstrip('.') for n in cpu_resident_module_names(layer_names, weight_map)}
        resident = {n.rstrip('.') for n in layer_names.get('resident', [])}
    streamed, resident_bytes, cpu_bytes = [], 0, 0
    for name, nbytes in layer_bytes.items():
        bare = name.rstrip('.')
        if bare in cpu:
            cpu_bytes += nbytes
            continue
        if bare in resident:
            resident_bytes += nbytes
        else:
            streamed.append((name, nbytes))
    streamed.sort(key=lambda kv: kv[1], reverse=True)
    largest_layer_name, largest_layer_bytes = streamed[0] if streamed else (None, 0)
    second_layer_bytes = streamed[1][1] if prefetching and len(streamed) > 1 else 0
    if not streamed and n_layers:
        largest_layer_bytes = checkpoint_bytes // int(n_layers)
        second_layer_bytes = largest_layer_bytes if prefetching else 0
        notes.append("Per-layer sizes were not in the index; peak VRAM uses checkpoint / n_layers.")

    avg_layer_bytes = (checkpoint_bytes // int(n_layers)) if n_layers else 0
    kv_bytes = _kv_cache_bytes(config, max_seq_len=max_seq_len)
    # Resident modules stay on GPU; prefetch holds another streamed layer. CPU-mmap tables do not.
    peak_vram = (largest_layer_bytes + second_layer_bytes + resident_bytes
                 + kv_bytes + _VRAM_OVERHEAD_BYTES)

    split_bytes = int(checkpoint_bytes * _split_factor(compression))
    disk_keep_both = checkpoint_bytes + split_bytes
    disk_delete_original = split_bytes

    disk_root = layer_shards_saving_path or checkpoint_path or os.getcwd()
    try:
        _, _, free_bytes = shutil.disk_usage(disk_root)
    except OSError:
        free_bytes = 0
        notes.append(f"Could not stat free disk under {disk_root}.")

    cache_path, cache_bytes, cache_files = _find_split_cache(
        checkpoint_path, layer_shards_saving_path, compression)

    if cache_path:
        notes.append("Layer cache already present; from_pretrained will skip splitting.")
    else:
        notes.append("Keeping the original plus the split copy needs ~2x disk unless delete_original=True.")

    if compression:
        notes.append(f"compression={compression} estimated split size is {int(_split_factor(compression)*100)}% of the checkpoint.")

    n_experts = config.get('num_experts') or config.get('num_local_experts') or config.get('n_routed_experts')
    if n_experts and (layer_names or {}).get('expert_prefix'):
        notes.append(f"MoE with {n_experts} experts; per-expert streaming may use less VRAM than a full layer.")
    elif n_experts:
        notes.append(f"MoE with {n_experts} packed experts; they stream with the layer and count toward peak VRAM.")
    if layer_names and (resident or cpu):
        extra = " and a prefetched second layer" if prefetching else ""
        notes.append(f"Peak VRAM includes resident modules{extra}; CPU-mmap tables are excluded.")

    dtype = config.get('torch_dtype') or 'unknown'
    arch = (config.get('architectures') or ['unknown'])[0]

    return {
        'model': str(model_local_path_or_repo_id),
        'architecture': arch,
        'source': source,
        'dtype': dtype,
        'checkpoint_bytes': checkpoint_bytes,
        'n_shards': n_shards,
        'n_layers': int(n_layers or 0),
        'n_modules': int(n_modules),
        'split_bytes': split_bytes,
        'disk_keep_both_bytes': disk_keep_both,
        'disk_delete_original_bytes': disk_delete_original,
        'free_bytes': int(free_bytes),
        'disk_ok_keep_both': free_bytes >= disk_keep_both if checkpoint_bytes else None,
        'disk_ok_delete_original': free_bytes >= disk_delete_original if checkpoint_bytes else None,
        'largest_layer': largest_layer_name,
        'largest_layer_bytes': int(largest_layer_bytes),
        'avg_layer_bytes': int(avg_layer_bytes),
        'second_layer_bytes': int(second_layer_bytes),
        'resident_bytes': int(resident_bytes),
        'cpu_resident_bytes': int(cpu_bytes),
        'kv_cache_bytes': int(kv_bytes),
        'overhead_bytes': _VRAM_OVERHEAD_BYTES,
        'max_seq_len': int(max_seq_len),
        'estimated_peak_vram_bytes': int(peak_vram),
        'compression': compression,
        'already_split': bool(cache_path),
        'split_cache_path': cache_path,
        'split_cache_bytes': int(cache_bytes),
        'split_cache_files': int(cache_files),
        'notes': notes,
    }


def format_resource_estimate(info):
    """Human-readable report for :func:`estimate_resources`."""
    keep = 'OK' if info.get('disk_ok_keep_both') else 'NOT ENOUGH'
    delete = 'OK' if info.get('disk_ok_delete_original') else 'NOT ENOUGH'
    cache_line = 'not split yet'
    if info.get('already_split'):
        cache_line = (f"{_fmt_bytes(info['split_cache_bytes'])}  "
                      f"({info['split_cache_files']} files at {info['split_cache_path']})")
    largest = _fmt_bytes(info.get('largest_layer_bytes'))
    if info.get('largest_layer'):
        largest = f"{largest}  ({info['largest_layer'].rstrip('.')})"
    lines = [
        'AirLLM resource estimate',
        f"  model:                  {info.get('model')}",
        f"  architecture:           {info.get('architecture')}  dtype={info.get('dtype')}",
        f"  checkpoint:             {_fmt_bytes(info.get('checkpoint_bytes'))}  ({info.get('n_shards')} files)",
        f"  split copy:             {_fmt_bytes(info.get('split_bytes'))}",
        f"  disk if keeping both:    {_fmt_bytes(info.get('disk_keep_both_bytes'))}  [{keep}]",
        f"  disk if delete_original: {_fmt_bytes(info.get('disk_delete_original_bytes'))}  [{delete}]",
        f"  free disk:              {_fmt_bytes(info.get('free_bytes'))}",
        f"  layers:                 {info.get('n_layers')} decoder"
        + (f", {info.get('n_modules')} streamed modules" if info.get('n_modules') else ""),
        f"  largest layer:          {largest}",
        f"  KV cache @ seq {info.get('max_seq_len')}:     {_fmt_bytes(info.get('kv_cache_bytes'))}",
        f"  estimated peak VRAM:    {_fmt_bytes(info.get('estimated_peak_vram_bytes'))}",
        f"  layer cache:            {cache_line}",
    ]
    for note in info.get('notes') or []:
        lines.append(f"  note: {note}")
    return '\n'.join(lines)
