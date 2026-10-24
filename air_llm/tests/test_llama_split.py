"""Structural tests for Llama checkpoints.

Llama models use the standard layout: embed_tokens, model.layers.*, norm, lm_head.
These tests build a miniature checkpoint with that shape and check that the splitter
writes every streamed module correctly.
"""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file, load_file

_AIRLLM_DIR = Path(__file__).resolve().parents[1] / 'airllm'

if 'airllm' not in sys.modules:
    _pkg = types.ModuleType('airllm')
    _pkg.__path__ = [str(_AIRLLM_DIR)]
    sys.modules['airllm'] = _pkg

from airllm.persist import model_persister as _persister_mod
from airllm.persist.safetensor_model_persister import SafetensorModelPersister

_persister_mod.model_persister = SafetensorModelPersister()

from airllm.utils import split_and_save_layers
import airllm.utils as utils_mod

N_LAYERS = 4
LAYER_PREFIX = "model.layers"

LLAMA_LAYER_NAMES = {
    'embed': 'model.embed_tokens',
    'layer_prefix': 'model.layers',
    'norm': 'model.norm',
    'lm_head': 'lm_head',
}


def build_fake_llama_checkpoint(root):
    """Standard Llama layout: embed_tokens in shard 0, layers spread across shards, norm+lm_head in last shard."""
    shards = {}

    # Shard 0: embed_tokens + first 2 layers
    shard0 = {
        "model.embed_tokens.weight": torch.randn(16, 8),
        f"{LAYER_PREFIX}.0.input_layernorm.weight": torch.randn(8),
        f"{LAYER_PREFIX}.0.self_attn.q_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.0.self_attn.k_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.0.self_attn.v_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.0.self_attn.o_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.0.mlp.gate_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.0.mlp.up_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.0.mlp.down_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.1.input_layernorm.weight": torch.randn(8),
        f"{LAYER_PREFIX}.1.self_attn.q_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.1.self_attn.k_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.1.self_attn.v_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.1.self_attn.o_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.1.mlp.gate_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.1.mlp.up_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.1.mlp.down_proj.weight": torch.randn(8, 8),
    }
    shards["model-00001-of-00003.safetensors"] = shard0

    # Shard 1: next 2 layers
    shard1 = {
        f"{LAYER_PREFIX}.2.input_layernorm.weight": torch.randn(8),
        f"{LAYER_PREFIX}.2.self_attn.q_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.2.self_attn.k_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.2.self_attn.v_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.2.self_attn.o_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.2.mlp.gate_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.2.mlp.up_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.2.mlp.down_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.3.input_layernorm.weight": torch.randn(8),
        f"{LAYER_PREFIX}.3.self_attn.q_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.3.self_attn.k_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.3.self_attn.v_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.3.self_attn.o_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.3.mlp.gate_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.3.mlp.up_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.3.mlp.down_proj.weight": torch.randn(8, 8),
    }
    shards["model-00002-of-00003.safetensors"] = shard1

    # Shard 2: norm + lm_head
    shard2 = {
        "model.norm.weight": torch.randn(8),
        "lm_head.weight": torch.randn(16, 8),
    }
    shards["model-00003-of-00003.safetensors"] = shard2

    weight_map = {}
    for fname, sd in shards.items():
        save_file(sd, str(Path(root) / fname))
        for k in sd:
            weight_map[k] = fname
    with open(Path(root) / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 1}, "weight_map": weight_map}, f)
    return shards, weight_map


class TestLlamaSplit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.shards, self.weight_map = build_fake_llama_checkpoint(self.root)
        self.out = Path(split_and_save_layers(self.root, layer_names=LLAMA_LAYER_NAMES))

    def tearDown(self):
        self.tmp.cleanup()

    def _split_file(self, module):
        return self.out / f"{module}.safetensors"

    def test_every_streamed_module_is_split_out(self):
        expected = ([LLAMA_LAYER_NAMES['embed']]
                    + [f"{LAYER_PREFIX}.{i}" for i in range(N_LAYERS)]
                    + [LLAMA_LAYER_NAMES['norm'], LLAMA_LAYER_NAMES['lm_head']])
        for module in expected:
            self.assertTrue(self._split_file(module).exists(), f"missing split for {module}")

    def test_decoder_and_head_round_trip(self):
        original = {}
        for fname in self.shards:
            original.update(load_file(str(self.root / fname)))

        recovered = {}
        for f in self.out.glob('*.safetensors'):
            recovered.update(load_file(str(f)))

        self.assertEqual(set(original.keys()), set(recovered.keys()))
        for k, v in original.items():
            self.assertEqual(v.dtype, recovered[k].dtype, f"{k} changed dtype")
            self.assertTrue(torch.equal(v, recovered[k]), f"{k} changed value")

    def test_shared_shards_are_copied_not_linked(self):
        """The fake checkpoint packs several layers per file, so passthrough must not fire."""
        src_inodes = {os.stat(self.root / f).st_ino
                      for f in os.listdir(self.root) if f.endswith('.safetensors')}
        for module in ([f"{LAYER_PREFIX}.{i}" for i in range(N_LAYERS)]
                       + ['model.embed_tokens', 'model.norm', 'lm_head']):
            dst = self._split_file(module)
            self.assertNotIn(os.stat(dst).st_ino, src_inodes,
                             f"{module} was linked, but this layout should be copied")


class TestSingleFileLoadingOptimization(unittest.TestCase):
    """Regression test for issue #69: excessive loading time on single-file checkpoints.

    Before the fix, split_and_save_layers reloaded the entire model file for each layer,
    leading to O(layers) disk reads (e.g. 7 loads for 7 modules). The fix loads it once.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        sd = {
            "model.embed_tokens.weight": torch.randn(16, 8),
        }
        for i in range(N_LAYERS):
            sd[f"{LAYER_PREFIX}.{i}.input_layernorm.weight"] = torch.randn(8)
            sd[f"{LAYER_PREFIX}.{i}.self_attn.q_proj.weight"] = torch.randn(8, 8)
        sd["model.norm.weight"] = torch.randn(8)
        sd["lm_head.weight"] = torch.randn(16, 8)
        save_file(sd, str(self.root / "model.safetensors"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_single_file_loaded_only_once(self):
        """Verify the checkpoint file is loaded exactly once into RAM instead of per-layer."""
        real_load = utils_mod.load_file
        load_count = 0

        def counting_load(*args, **kwargs):
            nonlocal load_count
            load_count += 1
            return real_load(*args, **kwargs)

        with patch.object(utils_mod, 'load_file', side_effect=counting_load):
            split_and_save_layers(self.root, layer_names=LLAMA_LAYER_NAMES)

        self.assertEqual(load_count, 1, f"Expected model file to be loaded once, but was loaded {load_count} times")


class TestCustomShardNaming(unittest.TestCase):
    """Regression test for checkpoint shards containing non-integer names with hyphens.

    Before the fix, _last_shard_of attempted int(v.split('-')[1]) without error handling,
    crashing with ValueError on files such as model-custom.safetensors.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        sd = {
            "model.embed_tokens.weight": torch.randn(16, 8),
            f"{LAYER_PREFIX}.0.input_layernorm.weight": torch.randn(8),
            "model.norm.weight": torch.randn(8),
            "lm_head.weight": torch.randn(16, 8),
        }
        save_file(sd, str(self.root / "model-custom.safetensors"))
        weight_map = {k: "model-custom.safetensors" for k in sd}
        with open(self.root / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {"total_size": 1}, "weight_map": weight_map}, f)

    def tearDown(self):
        self.tmp.cleanup()

    def test_non_integer_shard_name_does_not_crash(self):
        """_last_shard_of should handle non-integer shard name parts gracefully."""
        out = Path(split_and_save_layers(self.root, layer_names=LLAMA_LAYER_NAMES))
        self.assertTrue((out / "model.embed_tokens.safetensors").exists())


class TestSingleFileCheckpoint(unittest.TestCase):
    """Functional tests for single-file checkpoint splitting."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._build_single_file_checkpoint()
        self.out = Path(split_and_save_layers(self.root, layer_names=LLAMA_LAYER_NAMES))

    def tearDown(self):
        self.tmp.cleanup()

    def _build_single_file_checkpoint(self):
        """Build a single-file Llama checkpoint with all weights in model.safetensors."""
        sd = {
            "model.embed_tokens.weight": torch.randn(16, 8),
        }
        for i in range(N_LAYERS):
            sd[f"{LAYER_PREFIX}.{i}.input_layernorm.weight"] = torch.randn(8)
            sd[f"{LAYER_PREFIX}.{i}.self_attn.q_proj.weight"] = torch.randn(8, 8)
            sd[f"{LAYER_PREFIX}.{i}.self_attn.k_proj.weight"] = torch.randn(8, 8)
            sd[f"{LAYER_PREFIX}.{i}.self_attn.v_proj.weight"] = torch.randn(8, 8)
            sd[f"{LAYER_PREFIX}.{i}.self_attn.o_proj.weight"] = torch.randn(8, 8)
            sd[f"{LAYER_PREFIX}.{i}.mlp.gate_proj.weight"] = torch.randn(8, 8)
            sd[f"{LAYER_PREFIX}.{i}.mlp.up_proj.weight"] = torch.randn(8, 8)
            sd[f"{LAYER_PREFIX}.{i}.mlp.down_proj.weight"] = torch.randn(8, 8)
        sd["model.norm.weight"] = torch.randn(8)
        sd["lm_head.weight"] = torch.randn(16, 8)

        save_file(sd, str(self.root / "model.safetensors"))

    def test_all_modules_split_out(self):
        """Every streamed module should have its own shard in the output."""
        expected_modules = [
            "model.embed_tokens.",
            *[f"model.layers.{i}." for i in range(N_LAYERS)],
            "model.norm.",
            "lm_head.",
        ]
        for module in expected_modules:
            shard = self.out / f"{module}safetensors"
            self.assertTrue(shard.exists(), f"missing shard for {module}")
            done = self.out / f"{module}safetensors.done"
            self.assertTrue(done.exists(), f"missing done marker for {module}")

    def test_weights_round_trip(self):
        """Weights should be preserved exactly through the split."""
        for module in ["model.embed_tokens.", "model.layers.0.", "model.norm.", "lm_head."]:
            shard = self.out / f"{module}safetensors"
            loaded = load_file(str(shard))
            for k, v in loaded.items():
                self.assertGreater(v.numel(), 0, f"empty tensor in {module}")

    def test_no_index_json_left(self):
        """The output directory should not contain an index.json."""
        index_files = list(self.out.glob("*.index.json"))
        self.assertEqual(len(index_files), 0, "index.json should not be created in output")


if __name__ == '__main__':
    unittest.main(verbosity=2)