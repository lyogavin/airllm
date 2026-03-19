"""
Tests for single-file model support in split_and_save_layers.

Covers the case where a model ships as a single model.safetensors or
pytorch_model.bin file (no shard index), which is common for models <= ~7B.
"""
import json
import os
import tempfile
import shutil
import unittest

import torch
from safetensors.torch import save_file


class TestSingleFileModelSplit(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="airllm_single_file_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_fake_model_state(self):
        """Minimal Llama-style state dict with 1 decoder layer."""
        hidden = 64
        inter = 128
        vocab = 100
        heads = 4
        return {
            "model.embed_tokens.weight":              torch.randn(vocab, hidden),
            "model.layers.0.input_layernorm.weight":  torch.randn(hidden),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(hidden // heads, hidden),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(hidden // heads, hidden),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden, hidden),
            "model.layers.0.mlp.gate_proj.weight":    torch.randn(inter, hidden),
            "model.layers.0.mlp.up_proj.weight":      torch.randn(inter, hidden),
            "model.layers.0.mlp.down_proj.weight":    torch.randn(hidden, inter),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(hidden),
            "model.norm.weight":                      torch.randn(hidden),
            "lm_head.weight":                         torch.randn(vocab, hidden),
        }

    # ------------------------------------------------------------------
    # single model.safetensors (no index)
    # ------------------------------------------------------------------
    def test_split_single_safetensors_file(self):
        state = self._make_fake_model_state()
        save_file(state, os.path.join(self.tmpdir, "model.safetensors"))

        from airllm.utils import split_and_save_layers
        split_path = split_and_save_layers(self.tmpdir)

        self.assertTrue(os.path.isdir(split_path))
        expected_files = [
            "model.embed_tokens.safetensors",
            "model.layers.0.safetensors",
            "model.norm.safetensors",
            "lm_head.safetensors",
        ]
        for fname in expected_files:
            self.assertTrue(
                os.path.exists(os.path.join(split_path, fname)),
                f"Expected shard file missing: {fname}",
            )

    # ------------------------------------------------------------------
    # single pytorch_model.bin (no index)
    # ------------------------------------------------------------------
    def test_split_single_pytorch_bin_file(self):
        state = self._make_fake_model_state()
        torch.save(state, os.path.join(self.tmpdir, "pytorch_model.bin"))

        from airllm.utils import split_and_save_layers
        split_path = split_and_save_layers(self.tmpdir)

        self.assertTrue(os.path.isdir(split_path))
        self.assertTrue(
            os.path.exists(os.path.join(split_path, "model.embed_tokens.safetensors"))
        )

    # ------------------------------------------------------------------
    # sharded model.safetensors.index.json still works (regression)
    # ------------------------------------------------------------------
    def test_split_sharded_safetensors_still_works(self):
        state = self._make_fake_model_state()
        shard_file = "model-00001-of-00001.safetensors"
        save_file(state, os.path.join(self.tmpdir, shard_file))
        index = {"metadata": {}, "weight_map": {k: shard_file for k in state}}
        with open(os.path.join(self.tmpdir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)

        from airllm.utils import split_and_save_layers
        split_path = split_and_save_layers(self.tmpdir)

        self.assertTrue(
            os.path.exists(os.path.join(split_path, "model.embed_tokens.safetensors"))
        )

    # ------------------------------------------------------------------
    # no weights at all → FileNotFoundError
    # ------------------------------------------------------------------
    def test_raises_when_no_weights(self):
        from airllm.utils import split_and_save_layers
        with self.assertRaises(FileNotFoundError):
            split_and_save_layers(self.tmpdir)


if __name__ == "__main__":
    unittest.main()
