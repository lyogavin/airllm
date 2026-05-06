
import argparse
import json
import time
import gc
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from sentencepiece import SentencePieceProcessor
from .persist import ModelPersister
import psutil
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationMixin, LlamaForCausalLM, GenerationConfig
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

    sconfig = sanitize_config(params)

    # quantization = config.pop("quantization", None)
    model_args = ModelArgs(**sconfig)
    return model_args

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


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

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
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

class AirLLMLlamaMlx:

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

        print(f"[{msg}] - available mem: {available:.02f}mb, consumed: {consumed:.02f}mb, least available:{available:.02f}mb, max consumed: {max_consumed:.02f}mb")

    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=None, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, test_nonlayered=False, show_memory_util=False,
                 delete_original=False, reuse_modules=True):

        self.hf_token = hf_token
        self.set_layer_names_dict()
        self.test_nonlayered = test_nonlayered
        self.show_memory_util = show_memory_util
        self.least_available = None
        self.initial_available = psutil.virtual_memory().available / 1024 / 1024
        # When True, allocate one TransformerBlock/Embedding/RMSNorm/Linear instance up-front
        # and just .update() weights into it per layer. Skips ~66 Python instantiations per
        # 3-token gen on a 22-layer model. See projects/kir-airllm/03-real-numbers.
        self.reuse_modules = reuse_modules and not test_nonlayered



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

        self.tokenizer = self.get_tokenizer(hf_token=hf_token)

        if self.reuse_modules:
            self._block = TransformerBlock(args=self.model_args)
            self._embed = nn.Embedding(self.model_args.vocab_size, self.model_args.dim)
            self._norm_mod = RMSNorm(self.model_args.dim, eps=self.model_args.norm_eps)
            self._output_mod = nn.Linear(self.model_args.dim, self.model_args.vocab_size, bias=False)


    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)


    def generate(self, x, temperature=0, max_new_tokens=None, **kwargs):
        tokens = []
        for token in self.model_generate(x, temperature=temperature):
            tokens.append(token)


            if len(tokens) >= max_new_tokens:
                break


        s = self.tokenizer.decode([t.item() for t in tokens])
        return s

    def model_generate(self, x, temperature=0, max_new_tokens=None):
        cache = []
        TEST_NO_LAYERED = True

        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])

        # First we process the prompt x the same was as in __call__ but
        # save the caches in cache

        self.record_memory('before_tok_embeddings')
        if self.reuse_modules:
            embed_mod = self._embed
        else:
            embed_mod = nn.Embedding(self.model_args.vocab_size, self.model_args.dim)
        mask = mask.astype(embed_mod.weight.dtype)

        self.record_memory('before_loading_tok')
        update_weights = ModelPersister.get_model_persister().load_model(self.layer_names_dict['embed'], self.checkpoint_path)

        self.record_memory('after_loading_tok')
        embed_mod.update(update_weights['tok_embeddings'])

        x = embed_mod(x)
        # force execution
        mx.eval(x)

        if self.test_nonlayered:
            print(f"self.test_nonlayered:{self.test_nonlayered}, save layers")
            self.layers = []
            self.tok_embeddings = embed_mod
        elif not self.reuse_modules:
            del embed_mod
            gc.collect()

        self.record_memory('after_tok_embeddings')

        for il in tqdm(range(self.model_args.n_layers), desc='running layers'):
            self.record_memory(f'before layer {il}')
            if self.reuse_modules:
                l = self._block
            else:
                l = TransformerBlock(args=self.model_args)
            l.update(
                ModelPersister.get_model_persister().load_model(f'{self.layer_names_dict["layer_prefix"]}.{il}',
                                                                     self.checkpoint_path)['layers'][il]
            )

            x, c = l(x, mask=mask)
            # force execution
            mx.eval(x)
            # We store the per layer cache in a simple python list
            cache.append(c)

            if self.test_nonlayered:
                self.layers.append(l)
            elif not self.reuse_modules:
                del l
                gc.collect()
            self.record_memory(f'after layer {il}')

        self.record_memory('before_norm')
        if self.reuse_modules:
            norm_mod = self._norm_mod
        else:
            norm_mod = RMSNorm(self.model_args.dim, eps=self.model_args.norm_eps)
        norm_mod.update(
            ModelPersister.get_model_persister().load_model(self.layer_names_dict['norm'], self.checkpoint_path)['norm']
        )
        x = norm_mod(x)
        # force execution
        mx.eval(x)
        if self.test_nonlayered:
            self.norm = norm_mod
        elif not self.reuse_modules:
            del norm_mod
            gc.collect()
        self.record_memory('after_norm')

        # We only care about the last logits that generate the next token
        self.record_memory('before_lmhead')
        if self.reuse_modules:
            output_mod = self._output_mod
        else:
            output_mod = nn.Linear(self.model_args.dim, self.model_args.vocab_size, bias=False)
        output_mod.update(
            ModelPersister.get_model_persister().load_model(self.layer_names_dict['lm_head'], self.checkpoint_path)['output']
        )
        y = output_mod(x[:, -1])
        # force execution
        mx.eval(y)

        if self.test_nonlayered:
            self.output = output_mod
        elif not self.reuse_modules:
            del output_mod
            gc.collect()
        self.record_memory('after_lmhead')
        y = sample(y)


        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y



        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None]

            self.record_memory('before_tok_embeddings')
            if self.reuse_modules:
                embed_mod = self._embed
                embed_mod.update(
                    ModelPersister.get_model_persister().load_model(self.layer_names_dict['embed'], self.checkpoint_path)['tok_embeddings'])
            elif self.test_nonlayered:
                embed_mod = self.tok_embeddings
            else:
                embed_mod = nn.Embedding(self.model_args.vocab_size, self.model_args.dim)
                embed_mod.update(
                    ModelPersister.get_model_persister().load_model(self.layer_names_dict['embed'], self.checkpoint_path)['tok_embeddings'])
            x = embed_mod(x)

            # force execution
            mx.eval(x)
            if not self.test_nonlayered and not self.reuse_modules:
                del embed_mod
                gc.collect()
            self.record_memory('after_tok_embeddings')

            for i in tqdm(range(len(cache)), desc='running layers'):
                self.record_memory(f'before layer {i}')

                if self.reuse_modules:
                    l = self._block
                    l.update(ModelPersister.get_model_persister().load_model(f'{self.layer_names_dict["layer_prefix"]}.{i}',
                                                                             self.checkpoint_path)['layers'][i])
                elif self.test_nonlayered:
                    l = self.layers[i]
                else:
                    l = TransformerBlock(args=self.model_args)
                    l.update(ModelPersister.get_model_persister().load_model(f'{self.layer_names_dict["layer_prefix"]}.{i}',
                                                                             self.checkpoint_path)['layers'][i])

                x, cache[i] = l(x, mask=None, cache=cache[i])
                # force execution
                mx.eval(x)
                if not self.test_nonlayered and not self.reuse_modules:
                    del l
                    gc.collect()
                self.record_memory(f'after layer {i}')

            self.record_memory('before_norm')
            if self.reuse_modules:
                norm_mod = self._norm_mod
                norm_mod.update(ModelPersister.get_model_persister().load_model(self.layer_names_dict['norm'], self.checkpoint_path)['norm'])
            elif self.test_nonlayered:
                norm_mod = self.norm
            else:
                norm_mod = RMSNorm(self.model_args.dim, eps=self.model_args.norm_eps)
                norm_mod.update(ModelPersister.get_model_persister().load_model(self.layer_names_dict['norm'], self.checkpoint_path)['norm'])
            x = norm_mod(x)
            # force execution
            mx.eval(x)

            if not self.test_nonlayered and not self.reuse_modules:
                del norm_mod
                gc.collect()

            self.record_memory('after_norm')

            if self.reuse_modules:
                output_mod = self._output_mod
                output_mod.update(ModelPersister.get_model_persister().load_model(self.layer_names_dict['lm_head'], self.checkpoint_path)['output'])
            elif self.test_nonlayered:
                output_mod = self.output
            else:
                output_mod = nn.Linear(self.model_args.dim, self.model_args.vocab_size, bias=False)
                output_mod.update(ModelPersister.get_model_persister().load_model(self.layer_names_dict['lm_head'], self.checkpoint_path)['output'])
            y = sample(output_mod(x[:, -1]))

            # force execution
            mx.eval(y)
            if not self.test_nonlayered and not self.reuse_modules:
                del output_mod
                gc.collect()

            self.record_memory('after_lmhead')
            yield y