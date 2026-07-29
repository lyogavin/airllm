"""Structural tests for Kimi-K3-style checkpoints.

K3 is multimodal (decoder nested under ``language_model``), ships one shard per decoder layer, and
carries modules outside the streamed sequence (vision tower, projector, extra residual norms).
These tests build a miniature checkpoint with the same shape and check that the splitter links
rather than copies the one-module-per-shard files, still materialises the shared shard correctly,
and never loses a module because of shard ordering.
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

# Bind the package without running airllm/__init__.py: that module pulls in the MLX backend on
# macOS and the full transformers stack elsewhere, none of which these splitter tests need.
if 'airllm' not in sys.modules:
    _pkg = types.ModuleType('airllm')
    _pkg.__path__ = [str(_AIRLLM_DIR)]
    sys.modules['airllm'] = _pkg

from airllm.persist import model_persister as _persister_mod
from airllm.persist.safetensor_model_persister import SafetensorModelPersister

# The persister is otherwise chosen by platform, so pin the safetensors one to exercise the
# Linux/CUDA path regardless of where the test runs.
_persister_mod.model_persister = SafetensorModelPersister()

from airllm.utils import split_and_save_layers

N_LAYERS = 3
LAYER_PREFIX = "language_model.model.layers"

KIMI_LAYER_NAMES = {
    'embed': 'language_model.model.embed_tokens',
    'layer_prefix': LAYER_PREFIX,
    'norm': 'language_model.model.norm',
    'lm_head': 'language_model.lm_head',
    'resident': [
        'language_model.model.output_attn_res_norm',
        'language_model.model.output_attn_res_proj',
        'mm_projector',
        'vision_tower',
    ],
}


def build_fake_checkpoint(root):
    """One shard per layer, then a shared shard, then projector and vision tower."""
    n_shards = N_LAYERS + 3
    shards = {}

    for i in range(N_LAYERS):
        name = f"model-{i + 1:05d}-of-{n_shards:06d}.safetensors"
        shards[name] = {
            # A packed 4-bit payload plus its scale, like MXFP4 stores.
            f"{LAYER_PREFIX}.{i}.block_sparse_moe.experts.0.w1.weight_packed":
                torch.randint(0, 255, (8, 4), dtype=torch.uint8),
            f"{LAYER_PREFIX}.{i}.block_sparse_moe.experts.0.w1.weight_scale":
                torch.randn(8, 1, dtype=torch.float32),
            f"{LAYER_PREFIX}.{i}.input_layernorm.weight": torch.randn(8),
        }

    shared = f"model-{N_LAYERS + 1:05d}-of-{n_shards:06d}.safetensors"
    shards[shared] = {
        "language_model.model.embed_tokens.weight": torch.randn(16, 8),
        "language_model.model.norm.weight": torch.randn(8),
        "language_model.lm_head.weight": torch.randn(16, 8),
        "language_model.model.output_attn_res_norm.weight": torch.randn(8),
        "language_model.model.output_attn_res_proj.weight": torch.randn(8, 8),
    }
    # Deliberately out of order relative to the resident list: the projector lives in an *earlier*
    # shard than the vision tower, which the splitter must handle without losing tensors.
    proj = f"model-{N_LAYERS + 2:05d}-of-{n_shards:06d}.safetensors"
    shards[proj] = {"mm_projector.proj.0.weight": torch.randn(8, 8)}
    vis = f"model-{N_LAYERS + 3:05d}-of-{n_shards:06d}.safetensors"
    shards[vis] = {"vision_tower.encoder.blocks.0.mlp.fc0.weight": torch.randn(8, 8)}

    weight_map = {}
    for fname, sd in shards.items():
        save_file(sd, str(Path(root) / fname))
        for k in sd:
            weight_map[k] = fname
    with open(Path(root) / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 1}, "weight_map": weight_map}, f)
    return shards, weight_map


class TestKimiK3Split(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.shards, self.weight_map = build_fake_checkpoint(self.root)
        self.out = Path(split_and_save_layers(self.root, layer_names=KIMI_LAYER_NAMES))

    def tearDown(self):
        self.tmp.cleanup()

    def _split_file(self, module):
        return self.out / f"{module}.safetensors"

    def test_every_module_is_split_out(self):
        expected = ([KIMI_LAYER_NAMES['embed']]
                    + [f"{LAYER_PREFIX}.{i}" for i in range(N_LAYERS)]
                    + [KIMI_LAYER_NAMES['norm'], KIMI_LAYER_NAMES['lm_head']]
                    + KIMI_LAYER_NAMES['resident'])
        for module in expected:
            self.assertTrue(self._split_file(module).exists(), f"missing split for {module}")

    def test_one_shard_per_layer_is_linked_not_copied(self):
        """The whole point: a 1.5TB checkpoint must not be duplicated."""
        for i in range(N_LAYERS):
            src = self.root / f"model-{i + 1:05d}-of-{N_LAYERS + 3:06d}.safetensors"
            dst = self._split_file(f"{LAYER_PREFIX}.{i}")
            self.assertEqual(os.stat(src).st_ino, os.stat(dst).st_ino,
                             f"layer {i} was copied instead of linked")

    def test_projector_and_vision_tower_are_linked(self):
        for module, shard_no in (('mm_projector', N_LAYERS + 2), ('vision_tower', N_LAYERS + 3)):
            src = self.root / f"model-{shard_no:05d}-of-{N_LAYERS + 3:06d}.safetensors"
            self.assertEqual(os.stat(src).st_ino, os.stat(self._split_file(module)).st_ino,
                             f"{module} was copied instead of linked")

    def test_shared_shard_modules_are_materialised_separately(self):
        """embed / norm / lm_head / residual modules share one shard, so they must be real files."""
        shared_src = self.root / f"model-{N_LAYERS + 1:05d}-of-{N_LAYERS + 3:06d}.safetensors"
        for module in ('language_model.model.embed_tokens', 'language_model.model.norm',
                       'language_model.lm_head', 'language_model.model.output_attn_res_norm',
                       'language_model.model.output_attn_res_proj'):
            dst = self._split_file(module)
            self.assertNotEqual(os.stat(shared_src).st_ino, os.stat(dst).st_ino)
            keys = set(load_file(str(dst)).keys())
            self.assertTrue(keys, f"{module} split is empty")
            self.assertTrue(all(k.startswith(module + '.') for k in keys),
                            f"{module} split leaked other tensors: {keys}")

    def test_contents_match_the_original_checkpoint(self):
        """Linked or copied, every tensor must survive the split bit-for-bit."""
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

    def test_packed_4bit_payload_keeps_its_integer_dtype(self):
        sd = load_file(str(self._split_file(f"{LAYER_PREFIX}.0")))
        packed = [v for k, v in sd.items() if k.endswith('weight_packed')][0]
        self.assertEqual(packed.dtype, torch.uint8)


class TestStandardCheckpointStillSplits(unittest.TestCase):
    """Regression guard: the ordinary layout (several layers per shard) must be unaffected.

    Passthrough must not kick in here, because no shard holds exactly one module.
    """

    N = 4

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        shard_a = {"model.embed_tokens.weight": torch.randn(16, 8)}
        for i in range(2):
            shard_a[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(8, 8)
        shard_b = {}
        for i in range(2, self.N):
            shard_b[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(8, 8)
        shard_b["model.norm.weight"] = torch.randn(8)
        shard_b["lm_head.weight"] = torch.randn(16, 8)

        weight_map = {}
        for fname, sd in (("model-00001-of-00002.safetensors", shard_a),
                          ("model-00002-of-00002.safetensors", shard_b)):
            save_file(sd, str(self.root / fname))
            for k in sd:
                weight_map[k] = fname
        with open(self.root / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {"total_size": 1}, "weight_map": weight_map}, f)

        self.original = dict(shard_a)
        self.original.update(shard_b)
        self.out = Path(split_and_save_layers(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_modules_written_as_real_files(self):
        expected = (['model.embed_tokens'] + [f'model.layers.{i}' for i in range(self.N)]
                    + ['model.norm', 'lm_head'])
        src_inodes = {os.stat(self.root / f).st_ino
                      for f in os.listdir(self.root) if f.endswith('.safetensors')}
        for module in expected:
            dst = self.out / f"{module}.safetensors"
            self.assertTrue(dst.exists(), f"missing split for {module}")
            self.assertNotIn(os.stat(dst).st_ino, src_inodes,
                             f"{module} was linked, but this layout should be copied")

    def test_contents_round_trip(self):
        recovered = {}
        for f in self.out.glob('*.safetensors'):
            recovered.update(load_file(str(f)))
        self.assertEqual(set(self.original.keys()), set(recovered.keys()))
        for k, v in self.original.items():
            self.assertTrue(torch.equal(v, recovered[k]), f"{k} changed value")


if __name__ == '__main__':
    unittest.main(verbosity=2)
