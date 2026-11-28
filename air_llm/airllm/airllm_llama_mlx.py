
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from sentencepiece import SentencePieceProcessor
from .persist import ModelPersister
import psutil
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationMixin, GenerationConfig
from .utils import clean_memory, load_layer, \
    find_or_create_local_splitted_path



@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    rope_traditional: bool = True

def sanitize_config(config, weights=None):
    config.pop("model_type", None)
    n_heads = config["n_heads"] if 'n_heads' in config else config['num_attention_heads']
    if "n_kv_heads" not in config:
        config["n_kv_heads"] = n_heads
    if "head_dim" not in config:
        config["head_dim"] = config["dim"] // n_heads
    #if "hidden_dim" not in config:
    #    config["hidden_dim"] = weights["layers.0.feed_forward.w1.weight"].shape[0]
    #if config.get("vocab_size", -1) < 0:
    #    config["vocab_size"] = weights["output.weight"].shape[-1]
    if "rope_theta" not in config:
        config["rope_theta"] = 10000
    unused = ["multiple_of", "ffn_dim_multiplier"]
    for k in unused:
        config.pop(k, None)
    return config

def get_model_args_from_config(config):
    params = {}
    params["dim"] = config.hidden_size
    params["hidden_dim"] = config.intermediate_size
    params["n_heads"] = config.num_attention_heads
    if hasattr(config, "num_key_value_heads"):
        params["n_kv_heads"] = config.num_key_value_heads
    params["n_layers"] = config.num_hidden_layers
    params["vocab_size"] = config.vocab_size
    params["norm_eps"] = config.rms_norm_eps
    params["rope_traditional"] = False
    # FIX #368: respect rope_theta from the HF config instead of always
    # falling back to 10000 in sanitize_config(). Llama 3.x ships
    # rope_theta=500000; ignoring it corrupts all outputs.
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        params["rope_theta"] = rope_theta

    sconfig = sanitize_config(params)

    # quantization = config.pop("quantization", None)
    model_args = ModelArgs(**sconfig)
    return model_args

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        # Fused kernel; accumulates in float32 internally like the old hand-written version.
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = nn.RoPE(
            args.head_dim, traditional=args.rope_traditional, base=args.rope_theta
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Keys/values stay at n_kv_heads: the fused SDPA kernel handles grouped-query attention
        # natively, so the KV cache is n_heads/n_kv_heads times smaller than pre-tiling it.
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache

def sample(logits, temperature=0):
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temperature))


def _coerce_to_mlx_array(x):
    """Convert common tokenizer outputs to an MLX array for embedding lookup."""
    if isinstance(x, mx.array):
        return x

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    return mx.array(x)


def _readahead(path, chunk=16 << 20):
    """Read a file into the OS page cache so the next mx.load() of it is served from RAM."""
    try:
        with open(path, 'rb', buffering=0) as f:
            while f.read(chunk):
                pass
    except OSError:
        pass


def _tree_nbytes(tree):
    # Imported here so the module still imports under the mocked mlx used by the CPU-only tests.
    from mlx.utils import tree_flatten
    return sum(v.nbytes for _, v in tree_flatten(tree) if isinstance(v, mx.array))


class AirLLMLlamaMlx:

    # Fraction of the RAM available at start-up that resident layers may use when
    # layer_cache_bytes is left at "auto".
    AUTO_CACHE_FRACTION = 0.5

    # customize layer names here
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                       'layer_prefix': 'model.layers',
                       'norm': 'model.norm',
                       'lm_head': 'lm_head',}


    def record_memory(self, msg=None):
        if not self.show_memory_util:
            return

        available = psutil.virtual_memory().available / 1024 / 1024
        if self.least_available is None:
            self.least_available = available
        else:
            self.least_available = min(available, self.least_available)

        consumed = self.initial_available - available
        max_consumed = self.initial_available - self.least_available

        print(f"[{msg}] - available mem: {available:.02f}mb, consumed: {consumed:.02f}mb, least available:{self.least_available:.02f}mb, max consumed: {max_consumed:.02f}mb")

    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=None, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, test_nonlayered=False, show_memory_util=False,
                 delete_original=False, layer_cache_bytes=0):
        """
        layer_cache_bytes: how much memory loaded layers may keep occupying between tokens.
            0 (default) streams every layer from disk on every token, the same memory profile as
            before. "auto" uses AUTO_CACHE_FRACTION of the RAM available at start-up, so a model
            that fits is read from disk only once and anything that does not fit is still
            streamed; an int is a byte budget; None means unlimited.
        prefetching: read the next streamed layer's shard into the OS page cache on a worker
            thread while the current layer computes.
        """

        self.hf_token = hf_token
        self.set_layer_names_dict()
        self.test_nonlayered = test_nonlayered
        self.show_memory_util = show_memory_util
        self.least_available = None
        self.initial_available = psutil.virtual_memory().available / 1024 / 1024

        if test_nonlayered:
            layer_cache_bytes = None
        elif layer_cache_bytes == "auto":
            layer_cache_bytes = int(psutil.virtual_memory().available * self.AUTO_CACHE_FRACTION)
        self.layer_cache_bytes = layer_cache_bytes
        self._resident = {}          # layer name -> built module kept in memory
        self._resident_bytes = 0
        self.prefetching = prefetching
        self._executor = ThreadPoolExecutor(max_workers=1) if prefetching else None

        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(model_local_path_or_repo_id,
                                                                                         layer_shards_saving_path,
                                                                                         compression=compression,
                                                                                         layer_names=self.layer_names_dict,
                                                                                         hf_token=hf_token,
                                                                                         delete_original=delete_original)
        if hf_token is not None:
            self.config = AutoConfig.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            self.config = AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=True)


        self.model_args = get_model_args_from_config(self.config)

        self.layer_names = [self.layer_names_dict['embed']] + \
                           [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in range(self.model_args.n_layers)] + \
                           [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']]

        lm_head_shard = Path(self.checkpoint_path) / (self.layer_names_dict['lm_head'] + ".mlx.npz")
        self._tied_lm_head = not lm_head_shard.exists()
        if self._tied_lm_head and not getattr(self.config, "tie_word_embeddings", False):
            raise FileNotFoundError(f"lm_head shard not found: {lm_head_shard}")

        self.tokenizer = self.get_tokenizer(hf_token=hf_token)


    def load_lm_head_weights(self):
        """Return the weights for the output projection.

        Models with tied input/output embeddings (config.tie_word_embeddings) have no lm_head
        entry in the checkpoint, so the splitter writes no lm_head shard for them. In that case
        the output projection shares the token embedding matrix, so load it from the embed shard.
        """
        persister = ModelPersister.get_model_persister()
        if not self._tied_lm_head:
            return persister.load_model(self.layer_names_dict['lm_head'], self.checkpoint_path)['output']
        return persister.load_model(self.layer_names_dict['embed'], self.checkpoint_path)['tok_embeddings']

    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    # ---- module loading ---------------------------------------------------------------------

    def _shard_path(self, layer_name):
        return str(Path(self.checkpoint_path) / (layer_name + ".mlx.npz"))

    def _build(self, layer_name):
        """Construct the module for layer_name and load its weights (lazily, via mx.load)."""
        persister = ModelPersister.get_model_persister()
        args = self.model_args
        if layer_name == self.layer_names_dict['embed']:
            module = nn.Embedding(args.vocab_size, args.dim)
            module.update(persister.load_model(layer_name, self.checkpoint_path)['tok_embeddings'])
        elif layer_name == self.layer_names_dict['norm']:
            module = RMSNorm(args.dim, eps=args.norm_eps)
            module.update(persister.load_model(layer_name, self.checkpoint_path)['norm'])
        elif layer_name == self.layer_names_dict['lm_head']:
            module = nn.Linear(args.dim, args.vocab_size, bias=False)
            module.update(self.load_lm_head_weights())
        else:
            il = int(layer_name.rsplit('.', 1)[1])
            module = TransformerBlock(args=args)
            module.update(persister.load_model(layer_name, self.checkpoint_path)['layers'][il])
        return module

    def _get(self, layer_name):
        """Return (module, resident). Resident modules stay in memory across tokens."""
        module = self._resident.get(layer_name)
        if module is not None:
            return module, True
        module = self._build(layer_name)
        nbytes = _tree_nbytes(module.parameters())
        budget = self.layer_cache_bytes
        if budget is None or self._resident_bytes + nbytes <= budget:
            self._resident[layer_name] = module
            self._resident_bytes += nbytes
            return module, True
        return module, False

    def _prefetch(self, layer_name):
        """Warm the page cache for a layer that will have to be read from disk."""
        if self._executor is None or layer_name in self._resident:
            return
        if layer_name == self.layer_names_dict['lm_head'] and self._tied_lm_head:
            layer_name = self.layer_names_dict['embed']
            if layer_name in self._resident:
                return
        self._executor.submit(_readahead, self._shard_path(layer_name))

    def _project_out(self, x, embed_module):
        """Apply the output projection to the last position's hidden state."""
        x = x[:, -1]
        if self._tied_lm_head:
            # Tied weights: reuse the embedding matrix instead of loading it a second time.
            if embed_module is None:
                embed_module, _ = self._get(self.layer_names_dict['embed'])
            return embed_module.as_linear(x)
        lm_head, _ = self._get(self.layer_names_dict['lm_head'])
        return lm_head(x)

    def _step(self, x, mask, cache):
        """Run one forward pass through every layer. Returns logits for the last position."""
        names = self.layer_names
        n_layers = self.model_args.n_layers

        self.record_memory('before_tok_embeddings')
        self._prefetch(names[1])
        embed, embed_resident = self._get(names[0])
        x = embed(x)
        if not embed_resident:
            mx.eval(x)          # bound memory: let the streamed embedding table go now
            embed = None
        self.record_memory('after_tok_embeddings')

        first_pass = len(cache) == 0
        for il in range(n_layers):
            self._prefetch(names[il + 2])
            layer, resident = self._get(names[il + 1])
            x, c = layer(x, mask=mask, cache=None if first_pass else cache[il])
            if first_pass:
                cache.append(c)
            else:
                cache[il] = c
            if not resident:
                # Streamed layer: evaluate now so its weights can be released before the next
                # one is loaded. Resident layers are left lazy and evaluated as one graph.
                mx.eval(x, cache[il])
            if self.show_memory_util:
                self.record_memory(f'after layer {il}')

        norm, _ = self._get(self.layer_names_dict['norm'])
        x = norm(x)
        if not self._tied_lm_head or embed is None:
            self._prefetch(self.layer_names_dict['lm_head'])
        y = self._project_out(x, embed)
        mx.eval(y, cache)
        self.record_memory('after_lmhead')
        return y

    # ---- generation -------------------------------------------------------------------------

    def generate(self, x, temperature=0, max_new_tokens=20, **kwargs):
        x = _coerce_to_mlx_array(x)
        tokens = []
        for token in self.model_generate(x, temperature=temperature):
            tokens.append(token.item())
            if len(tokens) >= max_new_tokens:
                break
        return self.tokenizer.decode(tokens)

    def model_generate(self, x, temperature=0, max_new_tokens=None):
        x = _coerce_to_mlx_array(x)
        cache = []

        # Additive causal mask for the prompt; single-token decode steps need none.
        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1]).astype(mx.float16)

        y = sample(self._step(x, mask, cache), temperature)
        yield y

        n = 1
        while max_new_tokens is None or n < max_new_tokens:
            y = sample(self._step(y[:, None], None, cache), temperature)
            n += 1
            yield y

    def clear_layer_cache(self):
        """Drop resident layers (e.g. to free memory between prompts)."""
        self._resident.clear()
        self._resident_bytes = 0
