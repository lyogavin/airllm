"""Structural tests for Qwen3.8-Flash-Next / Qwen4-Exp checkpoints.

Flash-Next is ``Qwen4ExpForConditionalGeneration``: decoder nested under ``model.language_model``,
vision at ``model.visual``, no final RMSNorm (output mix is ``hyper_connection_mixer``), packed MoE
expert tensors, a sharded n-gram embedding under one decoder layer, and an ``mtp.*`` head that
transformers ignores. These tests check that the splitter peels the n-gram table out of its parent
layer, keeps the vision tower, drops MTP, and concatenates n-gram shards into a single weight.
"""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

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

from airllm.utils import (
    split_and_save_layers,
    merge_ngram_embedding_shards,
    load_merged_ngram_embedding,
    open_ngram_mmap_table,
    cpu_resident_module_names,
    layer_owner,
    _force_meta_embeddings,
    _wrap_forward_int64_scatter,
)

N_LAYERS = 4
LAYER_PREFIX = "model.language_model.layers"
NGRAM_MODULE = f"{LAYER_PREFIX}.1.ple.ple_embedding.ngram_embedding"

FLASH_LAYER_NAMES = {
    'embed': 'model.language_model.embed_tokens',
    'layer_prefix': LAYER_PREFIX,
    'norm': 'model.language_model.hyper_connection_mixer',
    'lm_head': 'lm_head',
    'resident': [
        'model.visual',
    ],
    'cpu_resident_marker': 'ple.ple_embedding.ngram_embedding',
}


def build_fake_checkpoint(root):
    """Layer 1 packs MoE experts plus n-gram shards in the same file as the rest of that layer."""
    shards = {}

    shard0 = {
        "model.visual.patch_embed.proj.weight": torch.randn(8, 3, 2, 2),
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(8, 8),
        "model.visual.merger.linear_fc1.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.0.linear_attn.in_proj_a.weight": torch.randn(4, 8),
        f"{LAYER_PREFIX}.0.mlp.experts.gate_up_proj": torch.randn(4, 8, 8),
        f"{LAYER_PREFIX}.0.mlp.experts.down_proj": torch.randn(4, 8, 4),
        f"{LAYER_PREFIX}.0.mlp.shared_expert.gate_proj.weight": torch.randn(8, 8),
    }
    shards["model-00001-of-00003.safetensors"] = shard0

    shard1 = {
        "model.language_model.embed_tokens.weight": torch.randn(16, 8),
        f"{LAYER_PREFIX}.1.linear_attn.out_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.1.mlp.experts.gate_up_proj": torch.randn(4, 8, 8),
        f"{LAYER_PREFIX}.1.mlp.experts.down_proj": torch.randn(4, 8, 4),
        f"{LAYER_PREFIX}.1.ple.key_proj.weight": torch.randn(8, 8),
        f"{NGRAM_MODULE}.shard_0.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        f"{NGRAM_MODULE}.shard_1.weight": torch.arange(12, 24, dtype=torch.float32).reshape(3, 4),
        f"{LAYER_PREFIX}.2.self_attn.q_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.3.mlp.experts.gate_up_proj": torch.randn(4, 8, 8),
        "model.language_model.hyper_connection_mixer.hc_norm.weight": torch.randn(8),
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": torch.randn(4, 8),
    }
    shards["model-00002-of-00003.safetensors"] = shard1

    shard2 = {
        "lm_head.weight": torch.randn(16, 8),
        "mtp.fc.weight": torch.randn(8, 8),
        "mtp.layers.0.mlp.gate_proj.weight": torch.randn(8, 8),
        "mtp.hyper_connection_mixer.hc_norm.weight": torch.randn(8),
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


class TestNgramShardMerge(unittest.TestCase):
    def test_concatenates_contiguous_shards_along_dim0(self):
        prefix = NGRAM_MODULE
        shard0 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        shard1 = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
        merged = merge_ngram_embedding_shards({
            f"{prefix}.shard_1.weight": shard1,
            f"{prefix}.shard_0.weight": shard0,
            f"{LAYER_PREFIX}.1.ple.key_proj.weight": torch.ones(3),
        })
        self.assertIn(f"{prefix}.weight", merged)
        self.assertNotIn(f"{prefix}.shard_0.weight", merged)
        self.assertTrue(torch.equal(merged[f"{prefix}.weight"], torch.cat([shard0, shard1], dim=0)))
        self.assertTrue(torch.equal(merged[f"{LAYER_PREFIX}.1.ple.key_proj.weight"], torch.ones(3)))

    def test_file_loader_concatenates_without_keeping_shard_keys(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            prefix = NGRAM_MODULE
            shard0 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
            shard1 = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
            path = Path(tmp.name) / f"{prefix}.safetensors"
            save_file({
                f"{prefix}.shard_0.weight": shard0,
                f"{prefix}.shard_1.weight": shard1,
                f"{LAYER_PREFIX}.1.ple.key_proj.weight": torch.ones(3),
            }, str(path))
            loaded = load_merged_ngram_embedding(tmp.name, prefix)
            self.assertIn(f"{prefix}.weight", loaded)
            self.assertNotIn(f"{prefix}.shard_0.weight", loaded)
            self.assertTrue(torch.equal(loaded[f"{prefix}.weight"], torch.cat([shard0, shard1], dim=0)))
            self.assertTrue(torch.equal(loaded[f"{LAYER_PREFIX}.1.ple.key_proj.weight"], torch.ones(3)))
        finally:
            tmp.cleanup()

    def test_rejects_gapped_shard_indices(self):
        with self.assertRaises(ValueError):
            merge_ngram_embedding_shards({
                f"{NGRAM_MODULE}.shard_0.weight": torch.zeros(2, 2),
                f"{NGRAM_MODULE}.shard_2.weight": torch.ones(2, 2),
            })

    def test_marker_discovers_the_parent_layer(self):
        keys = [
            f"{NGRAM_MODULE}.shard_0.weight",
            f"{LAYER_PREFIX}.1.mlp.experts.down_proj",
        ]
        names = cpu_resident_module_names(FLASH_LAYER_NAMES, keys)
        self.assertEqual(names, [NGRAM_MODULE])

    def test_longest_prefix_wins(self):
        prefixes = [f"{LAYER_PREFIX}.1.", f"{NGRAM_MODULE}."]
        self.assertEqual(layer_owner(f"{NGRAM_MODULE}.shard_0.weight", prefixes), f"{NGRAM_MODULE}.")
        self.assertEqual(
            layer_owner(f"{LAYER_PREFIX}.1.mlp.experts.down_proj", prefixes),
            f"{LAYER_PREFIX}.1.",
        )


class TestQwen38FlashNextSplit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.shards, self.weight_map = build_fake_checkpoint(self.root)
        self.out = Path(split_and_save_layers(self.root, layer_names=FLASH_LAYER_NAMES))

    def tearDown(self):
        self.tmp.cleanup()

    def _split_file(self, module):
        return self.out / f"{module}.safetensors"

    def test_architecture_maps_to_the_qwen4_exp_subclass(self):
        src = (_AIRLLM_DIR / 'auto_model.py').read_text()
        self.assertIn('"Qwen4ExpForConditionalGeneration": "AirLLMQwen4Exp"', src)
        init_src = (_AIRLLM_DIR / '__init__.py').read_text()
        self.assertIn('AirLLMQwen4Exp', init_src)
        self.assertTrue((_AIRLLM_DIR / 'airllm_qwen4_exp.py').exists())

    def test_every_streamed_and_resident_module_is_split_out(self):
        expected = ([FLASH_LAYER_NAMES['embed']]
                    + [f"{LAYER_PREFIX}.{i}" for i in range(N_LAYERS)]
                    + [FLASH_LAYER_NAMES['norm'], FLASH_LAYER_NAMES['lm_head']]
                    + FLASH_LAYER_NAMES['resident'])
        for module in expected:
            self.assertTrue(self._split_file(module).exists(), f"missing split for {module}")
        self.assertTrue((self.out / f"{NGRAM_MODULE}.mmap").exists(), "missing n-gram mmap")
        self.assertTrue((self.out / f"{NGRAM_MODULE}.mmap.json").exists())

    def test_ngram_table_is_peeled_out_of_its_decoder_layer(self):
        layer1 = load_file(str(self._split_file(f"{LAYER_PREFIX}.1")))
        table = open_ngram_mmap_table(self.out, NGRAM_MODULE)
        shard0 = load_file(str(self.root / "model-00002-of-00003.safetensors"))[
            f"{NGRAM_MODULE}.shard_0.weight"]
        shard1 = load_file(str(self.root / "model-00002-of-00003.safetensors"))[
            f"{NGRAM_MODULE}.shard_1.weight"]
        self.assertTrue(torch.equal(table, torch.cat([shard0, shard1], dim=0)))
        self.assertFalse(any('ngram_embedding' in k for k in layer1),
                         f"n-gram table leaked into decoder layer 1: {set(layer1)}")
        self.assertFalse(self._split_file(NGRAM_MODULE).exists(),
                         "n-gram table should be an mmap, not a safetensors shard")
        self.assertIn(f"{LAYER_PREFIX}.1.mlp.experts.gate_up_proj", layer1)
        self.assertIn(f"{LAYER_PREFIX}.1.ple.key_proj.weight", layer1)

    def test_packed_experts_stay_on_the_decoder_layer(self):
        layer0 = load_file(str(self._split_file(f"{LAYER_PREFIX}.0")))
        self.assertIn(f"{LAYER_PREFIX}.0.mlp.experts.gate_up_proj", layer0)
        self.assertIn(f"{LAYER_PREFIX}.0.mlp.experts.down_proj", layer0)
        self.assertFalse(any(f"{LAYER_PREFIX}.0.mlp.experts.0." in k for k in layer0))

    def test_output_mix_is_the_norm_slot(self):
        sd = load_file(str(self._split_file("model.language_model.hyper_connection_mixer")))
        self.assertIn("model.language_model.hyper_connection_mixer.hc_norm.weight", sd)
        self.assertFalse(self._split_file("model.language_model.norm").exists())

    def test_vision_tower_is_one_shard_and_excludes_decoder_weights(self):
        sd = load_file(str(self._split_file("model.visual")))
        self.assertTrue(sd, "vision split is empty")
        self.assertTrue(all(k.startswith("model.visual.") for k in sd),
                        f"vision split leaked decoder tensors: {set(sd)}")

    def test_mtp_is_dropped(self):
        recovered = {}
        for f in self.out.glob('*.safetensors'):
            recovered.update(load_file(str(f)))
        self.assertFalse(any(k.startswith("mtp.") for k in recovered),
                         f"MTP leaked into the split: {[k for k in recovered if k.startswith('mtp.')]}")
        self.assertFalse(any(p.name.startswith("mtp") for p in self.out.glob('*.safetensors')))

    def test_non_mtp_round_trip(self):
        original = {}
        for fname in self.shards:
            original.update(load_file(str(self.root / fname)))
        ngram_keys = [k for k in original if 'ngram_embedding.shard_' in k]
        ngram_cat = torch.cat(
            [original[k] for k in sorted(ngram_keys, key=lambda k: int(k.rsplit('_', 1)[-1].split('.')[0]))],
            dim=0)
        keep = {k: v for k, v in original.items()
                if not k.startswith("mtp.") and k not in ngram_keys}

        recovered = {}
        for f in self.out.glob('*.safetensors'):
            recovered.update(load_file(str(f)))

        self.assertEqual(set(keep.keys()), set(recovered.keys()))
        for k, v in keep.items():
            self.assertEqual(v.dtype, recovered[k].dtype, f"{k} changed dtype")
            self.assertTrue(torch.equal(v, recovered[k]), f"{k} changed value")
        self.assertTrue(torch.equal(open_ngram_mmap_table(self.out, NGRAM_MODULE), ngram_cat))

    def test_shared_shards_are_copied_not_linked(self):
        src_inodes = {os.stat(self.root / f).st_ino
                      for f in os.listdir(self.root) if f.endswith('.safetensors')}
        for module in ([f"{LAYER_PREFIX}.{i}" for i in range(N_LAYERS)]
                       + ['model.language_model.embed_tokens', 'model.visual', 'lm_head']):
            dst = self._split_file(module)
            self.assertNotIn(os.stat(dst).st_ino, src_inodes,
                             f"{module} was linked, but this layout should be copied")


class TestForceMetaEmbeddings(unittest.TestCase):
    """Flash-Next's PLE table is ``nn.Embedding(~320M, 160)``. Born on CPU that is ~191GB fp32."""

    def test_huge_embedding_is_born_on_meta(self):
        with _force_meta_embeddings():
            emb = torch.nn.Embedding(320_001_536, 160)
        self.assertEqual(emb.weight.device.type, 'meta')
        self.assertEqual(tuple(emb.weight.shape), (320_001_536, 160))
        # Patch must not leak: a later Embedding should use the default (CPU) device.
        small = torch.nn.Embedding(4, 8)
        self.assertEqual(small.weight.device.type, 'cpu')
        self.assertEqual(tuple(small.weight.shape), (4, 8))


class TestIndexerScatterInt64(unittest.TestCase):
    def test_int32_scatter_indices_are_promoted(self):
        class Dummy(torch.nn.Module):
            def forward(self, _):
                mask = torch.zeros(1, 5, dtype=torch.bool)
                index = torch.tensor([[1, 2]], dtype=torch.int32)
                return mask.scatter(-1, index, True)

        Dummy.forward = _wrap_forward_int64_scatter(Dummy.forward)
        out = Dummy()(None)
        self.assertEqual(out.dtype, torch.bool)
        self.assertEqual(out.tolist(), [[False, True, True, False, False]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
