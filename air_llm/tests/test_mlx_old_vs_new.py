"""Check the faster MLX path against the implementation it replaced.

The "old" classes/functions below are verbatim copies of what airllm_llama_mlx.py and
persist/mlx_model_persister.py used before the speed-up, so each test shows exactly what
changed and that the result did not.

Needs real MLX (Apple silicon). Run from the repo root:
    PYTHONPATH=air_llm python -m unittest air_llm.tests.test_mlx_old_vs_new -v
"""
import sys
import unittest

try:  # imported at collection time so the mocked-mlx tests don't shadow the real package
    import mlx.core as mx
    import mlx.nn as nn
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False

needs_mlx = unittest.skipUnless(sys.platform == "darwin" and HAVE_MLX, "needs MLX on Apple silicon")


# ---- old implementations (copied from before the change) ------------------------------------

def old_map_torch_to_mlx(model):
    model = {k.replace("model.", ""): v for k, v in model.items()}
    model = {k.replace("mlp", "feed_forward"): v for k, v in model.items()}
    model = {k.replace("down_proj", "w2"): v for k, v in model.items()}
    model = {k.replace("up_proj", "w3"): v for k, v in model.items()}
    model = {k.replace("gate_proj", "w1"): v for k, v in model.items()}
    model = {k.replace("input_layernorm", "attention_norm"): v for k, v in model.items()}
    model = {k.replace("post_attention_layernorm", "ffn_norm"): v for k, v in model.items()}
    model = {k.replace("lm_head", "output"): v for k, v in model.items()}
    model = {k.replace("embed_tokens", "tok_embeddings"): v for k, v in model.items()}
    model = {k.replace("self_attn", "attention"): v for k, v in model.items()}
    model = {k.replace("q_proj", "wq"): v for k, v in model.items()}
    model = {k.replace("k_proj", "wk"): v for k, v in model.items()}
    model = {k.replace("v_proj", "wv"): v for k, v in model.items()}
    model = {k.replace("o_proj", "wo"): v for k, v in model.items()}
    return model


def old_rms_norm(weight, x, eps):
    def _norm(x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + eps)
    output = _norm(x.astype(mx.float32)).astype(x.dtype)
    return weight * output


def old_attention(attn, x, mask=None, cache=None):
    """Old Attention.__call__: K/V tiled to n_heads, hand-written softmax(QK^T)V."""
    B, L, D = x.shape
    queries, keys, values = attn.wq(x), attn.wk(x), attn.wv(x)
    queries = queries.reshape(B, L, attn.n_heads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

    def repeat(a):
        a = mx.concatenate([mx.expand_dims(a, 2)] * attn.repeats, axis=2)
        return a.reshape([B, attn.n_heads, L, -1])

    keys, values = map(repeat, (keys, values))
    if cache is not None:
        key_cache, value_cache = cache
        queries = attn.rope(queries, offset=key_cache.shape[2])
        keys = attn.rope(keys, offset=key_cache.shape[2])
        keys = mx.concatenate([key_cache, keys], axis=2)
        values = mx.concatenate([value_cache, values], axis=2)
    else:
        queries = attn.rope(queries)
        keys = attn.rope(keys)
    scores = (queries * attn.scale) @ keys.transpose(0, 1, 3, 2)
    if mask is not None:
        scores += mask
    scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
    output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.wo(output), (keys, values)


# ---- tests ----------------------------------------------------------------------------------

def _args():
    from ..airllm.airllm_llama_mlx import ModelArgs
    # 6 query heads sharing 2 KV heads (grouped-query attention, like Llama 3 / SmolLM2).
    return ModelArgs(dim=96, n_layers=2, head_dim=16, hidden_dim=128, n_heads=6, n_kv_heads=2,
                     norm_eps=1e-5, vocab_size=50, rope_theta=10000.0, rope_traditional=False)


@needs_mlx
class TestKeyRename(unittest.TestCase):
    """Old: 14 full dict rebuilds per shard load. New: one pass, same result."""

    def test_same_keys_as_old_mapping(self):
        from ..airllm.persist.mlx_model_persister import map_torch_to_mlx
        keys = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
        for i in (0, 11):
            p = f"model.layers.{i}."
            keys += [p + n for n in (
                "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight")]
        state = {k: object() for k in keys}
        self.assertEqual(map_torch_to_mlx(state), old_map_torch_to_mlx(state))


@needs_mlx
class TestFusedKernels(unittest.TestCase):

    def test_rms_norm_matches_old(self):
        """Old: four separate ops in Python. New: mx.fast.rms_norm."""
        from ..airllm.airllm_llama_mlx import RMSNorm
        mx.random.seed(0)
        norm = RMSNorm(96, eps=1e-5)
        norm.weight = mx.random.normal((96,)).astype(mx.float16)
        x = mx.random.normal((1, 7, 96)).astype(mx.float16)
        new, old = norm(x), old_rms_norm(norm.weight, x, 1e-5)
        self.assertEqual(new.dtype, old.dtype)
        self.assertTrue(mx.allclose(new, old, atol=2e-2, rtol=1e-2).item())

    def test_attention_matches_old_on_prefill_and_decode(self):
        """Old: K/V tiled to n_heads + manual softmax. New: fused SDPA on untiled K/V."""
        from ..airllm.airllm_llama_mlx import Attention
        mx.random.seed(1)
        args = _args()
        attn = Attention(args)
        L = 7
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        x = mx.random.normal((1, L, args.dim))

        new_out, new_cache = attn(x, mask=mask)
        old_out, old_cache = old_attention(attn, x, mask=mask)
        self.assertTrue(mx.allclose(new_out, old_out, atol=1e-4).item())

        for _ in range(3):  # a few decode steps reusing each version's own cache
            t = mx.random.normal((1, 1, args.dim))
            new_out, new_cache = attn(t, cache=new_cache)
            old_out, old_cache = old_attention(attn, t, cache=old_cache)
            self.assertTrue(mx.allclose(new_out, old_out, atol=1e-4).item())

    def test_kv_cache_is_not_tiled(self):
        """The KV cache is n_heads / n_kv_heads times smaller than before (3x here)."""
        from ..airllm.airllm_llama_mlx import Attention
        args = _args()
        attn = Attention(args)
        x = mx.random.normal((1, 5, args.dim))
        _, (k_new, _) = attn(x)
        _, (k_old, _) = old_attention(attn, x)
        self.assertEqual(k_new.shape, (1, args.n_kv_heads, 5, args.head_dim))
        self.assertEqual(k_old.shape, (1, args.n_heads, 5, args.head_dim))


def _stub_model(budget, tied=False):
    """An AirLLMLlamaMlx with no checkpoint: _build() returns small fresh modules and counts loads."""
    from ..airllm.airllm_llama_mlx import AirLLMLlamaMlx
    m = AirLLMLlamaMlx.__new__(AirLLMLlamaMlx)
    m.set_layer_names_dict()
    m.layer_cache_bytes = budget
    m._resident, m._resident_bytes = {}, 0
    m._executor = None
    m._tied_lm_head = tied
    m.loads = []

    def build(name):
        m.loads.append(name)
        if name == m.layer_names_dict['embed']:
            return nn.Embedding(50, 8)
        return nn.Linear(8, 8)          # 64 + 8 float32 values = 288 bytes
    m._build = build
    return m


@needs_mlx
class TestResidentLayers(unittest.TestCase):
    """Old: every layer was re-read from disk on every token. New: kept while within budget."""

    def test_unlimited_budget_loads_each_layer_once(self):
        m = _stub_model(budget=None)
        for _ in range(3):              # three "tokens"
            for name in ("a", "b", "c"):
                _, resident = m._get(name)
                self.assertTrue(resident)
        self.assertEqual(m.loads, ["a", "b", "c"])

    def test_zero_budget_streams_like_before(self):
        m = _stub_model(budget=0)
        for _ in range(2):
            for name in ("a", "b"):
                _, resident = m._get(name)
                self.assertFalse(resident)
        self.assertEqual(m.loads, ["a", "b", "a", "b"])

    def test_partial_budget_keeps_what_fits_and_streams_the_rest(self):
        m = _stub_model(budget=288 * 2)  # room for exactly two layers
        for _ in range(2):
            for name in ("a", "b", "c"):
                m._get(name)
        self.assertEqual(m.loads, ["a", "b", "c", "c"])
        self.assertEqual(m._resident_bytes, 288 * 2)

    def test_tied_lm_head_reuses_embedding_instead_of_loading_it_again(self):
        """Old: tied models loaded the embedding table twice per token (input + output)."""
        m = _stub_model(budget=0, tied=True)
        embed, _ = m._get(m.layer_names_dict['embed'])
        x = mx.random.normal((1, 3, 8))
        logits = m._project_out(x, embed)
        expected = x[:, -1] @ embed.weight.T
        self.assertTrue(mx.allclose(logits, expected, atol=1e-5).item())
        self.assertEqual(m.loads, [m.layer_names_dict['embed']])   # no second load


if __name__ == "__main__":
    unittest.main()
