"""Unit tests for the layer-count parser used when splitting checkpoints.

The splitter (utils._count_layers) must count the decoder layers of a checkpoint
regardless of how deeply the LM is nested under wrapper modules. Different model
families use different key prefixes, and VLM / multimodal checkpoints nest the LM
one or more levels deeper (e.g. ``language_model.model.layers.0...``), which used
to crash the parser with:

    ValueError: invalid literal for int() with base 10: 'layers'

These tests exercise a matrix of the prefixes seen across the architectures AirLLM
supports plus the nested shapes that triggered the bug (see #335 / #336).
"""

import sys
import types
import unittest
from pathlib import Path

_AIRLLM_DIR = Path(__file__).resolve().parents[1] / 'airllm'

# Bind the package without running airllm/__init__.py: that module pulls in the MLX
# backend on macOS and the full transformers stack elsewhere, neither of which these
# pure-parser tests need.
if 'airllm' not in sys.modules:
    _pkg = types.ModuleType('airllm')
    _pkg.__path__ = [str(_AIRLLM_DIR)]
    sys.modules['airllm'] = _pkg

from airllm.utils import _count_layers


def layer_keys(prefix, n, sub='self_attn.q_proj.weight'):
    """Build a checkpoint-style key list for a single prefix: prefix.N.<sub>."""
    return [f'{prefix}.{i}.{sub}' for i in range(n)]


class TestLayerCountParsing(unittest.TestCase):
    """Direct tests of _count_layers over real-world key prefixes and nesting shapes."""

    def test_flat_default_llama_layout(self):
        # Generic / Llama / Baichuan / InternLM / Mistral / Mixtral / Qwen2 (model.layers)
        keys = layer_keys('model.layers', 5) + ['model.embed_tokens.weight', 'lm_head.weight']
        self.assertEqual(_count_layers(keys, 'model.layers.'), 5)

    def test_qwen1_transformer_h_layout(self):
        # Qwen1 uses transformer.h.N... (airllm_qwen.py)
        keys = layer_keys('transformer.h', 4, 'self_attention.c_attn.weight') + ['transformer.wte.weight']
        self.assertEqual(_count_layers(keys, 'transformer.h.'), 4)

    def test_chatglm_nested_layer_prefix(self):
        # ChatGLM uses transformer.encoder.layers.N... (airllm_chatglm.py)
        prefix = 'transformer.encoder.layers'
        keys = layer_keys(prefix, 6, 'self_attention.dense.weight') + ['transformer.word_embeddings.weight']
        self.assertEqual(_count_layers(keys, prefix + '.'), 6)

    def test_kimi_k3_language_model_prefix(self):
        # Kimi K3 nests the decoder under language_model (airllm_kimi_k3.py)
        prefix = 'language_model.model.layers'
        keys = [f'{prefix}.{i}.block_sparse_moe.experts.0.w1.weight' for i in range(3)]
        self.assertEqual(_count_layers(keys, prefix + '.'), 3)

    def test_nested_vlm_default_layout(self):
        # The exact bug: 'model.layers' appears mid-key, not at position [0,1,2].
        # Old code did int(k.split('.')[2]) -> 'layers' -> ValueError.
        keys = layer_keys('language_model.model.layers', 7) + ['language_model.model.embed_tokens.weight']
        self.assertEqual(_count_layers(keys, 'model.layers.'), 7)

    def test_deep_multi_level_nesting(self):
        # LM wrapped more than once: marker is found deep inside the key.
        keys = layer_keys('model.language_model.model.layers', 4)
        self.assertEqual(_count_layers(keys, 'model.layers.'), 4)

    def test_qwen3_style_ge_model_prefix(self):
        # Shape reported for Qwen3.6 in #335: ge_model.layers.N...
        keys = layer_keys('ge_model.layers', 28) + ['ge_model.embed_tokens.weight']
        self.assertEqual(_count_layers(keys, 'model.layers.'), 28)

    def test_non_layer_keys_are_ignored(self):
        keys = ['model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight',
                'vision_tower.encoder.blocks.0.mlp.fc0.weight', 'mm_projector.proj.0.weight']
        self.assertEqual(_count_layers(keys, 'model.layers.'), 0)

    def test_empty_index_is_not_a_crash(self):
        self.assertEqual(_count_layers([], 'model.layers.'), 0)

    def test_non_numeric_segment_is_skipped(self):
        # 'model.layers.' is present but the next segment is not a digit; must not crash.
        keys = ['model.layers.attention.weight', 'model.layers.gate.weight']
        self.assertEqual(_count_layers(keys, 'model.layers.'), 0)

    def test_duplicate_layer_indices_counted_once(self):
        keys = (layer_keys('model.layers', 3) + layer_keys('model.layers', 3)
                + layer_keys('model.layers', 3))
        self.assertEqual(_count_layers(keys, 'model.layers.'), 3)

    def test_issue_335_reproduction_does_not_raise(self):
        # Original crash, verbatim from the issue.
        k = 'language_model.model.layers.0.self_attn.q_proj.weight'
        self.assertEqual(_count_layers([k], 'model.layers.'), 1)
        with self.assertRaises(ValueError):
            int(k.split('.')[2])  # the old code path

    def test_marker_not_at_key_start(self):
        # Marker immediately after a wrapper module, the reported failing shape.
        keys = ['language_model.model.layers.0.self_attn.q_proj.weight',
                'language_model.model.layers.1.self_attn.q_proj.weight']
        self.assertEqual(_count_layers(keys, 'model.layers.'), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
