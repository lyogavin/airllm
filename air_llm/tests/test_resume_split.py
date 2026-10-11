"""Resume an interrupted layer split without rewriting already-saved shards.

Splitting a 70B+ checkpoint can take hours. A crash used to re-save every layer; now complete
layers (payload + ``.done`` marker) are kept and only the rest are written.
"""

import json
import os
import sys
import tempfile
import time
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

from airllm.utils import split_and_save_layers, remove_incomplete_layer_artifacts

N_LAYERS = 3
LAYER_PREFIX = "model.layers"

# Trailing-dot names match what the splitter writes (``model.layers.0.safetensors``).
EXPECTED_MODULES = (
    ['model.embed_tokens']
    + [f"{LAYER_PREFIX}.{i}" for i in range(N_LAYERS)]
    + ['model.norm', 'lm_head']
)


def build_fake_checkpoint(root):
    """Two shards that share modules, so resume must re-read a shard for a missing layer."""
    shards = {}
    shard0 = {
        "model.embed_tokens.weight": torch.randn(16, 8),
        f"{LAYER_PREFIX}.0.input_layernorm.weight": torch.randn(8),
        f"{LAYER_PREFIX}.0.self_attn.q_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.1.input_layernorm.weight": torch.randn(8),
        f"{LAYER_PREFIX}.1.self_attn.q_proj.weight": torch.randn(8, 8),
    }
    shards["model-00001-of-00002.safetensors"] = shard0

    shard1 = {
        f"{LAYER_PREFIX}.2.input_layernorm.weight": torch.randn(8),
        f"{LAYER_PREFIX}.2.self_attn.q_proj.weight": torch.randn(8, 8),
        "model.norm.weight": torch.randn(8),
        "lm_head.weight": torch.randn(16, 8),
    }
    shards["model-00002-of-00002.safetensors"] = shard1

    weight_map = {}
    for fname, sd in shards.items():
        save_file(sd, str(Path(root) / fname))
        for k in sd:
            weight_map[k] = fname
    with open(Path(root) / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 1}, "weight_map": weight_map}, f)
    return shards


def _split_path(out, module):
    return Path(out) / f"{module}.safetensors"


def _stamp(path, when):
    os.utime(path, (when, when))
    done = Path(str(path) + ".done")
    if done.exists():
        os.utime(done, (when, when))


class TestResumeSplit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.shards = build_fake_checkpoint(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _split(self):
        return Path(split_and_save_layers(self.root))

    def test_fresh_split_writes_every_module(self):
        out = self._split()
        for module in EXPECTED_MODULES:
            self.assertTrue(_split_path(out, module).exists(), f"missing {module}")
            self.assertTrue((_split_path(out, module).parent / f"{module}.safetensors.done").exists())

    def test_resume_restores_deleted_layers_and_keeps_the_rest(self):
        out = self._split()
        kept = "model.layers.0"
        missing = "model.layers.1"
        old = time.time() - 120
        kept_file = _split_path(out, kept)
        _stamp(kept_file, old)
        kept_mtime = kept_file.stat().st_mtime
        kept_ino = os.stat(kept_file).st_ino

        _split_path(out, missing).unlink()
        (out / f"{missing}.safetensors.done").unlink()

        out2 = self._split()
        self.assertEqual(out, out2)
        self.assertTrue(_split_path(out, missing).exists())
        self.assertTrue((out / f"{missing}.safetensors.done").exists())
        self.assertEqual(os.stat(kept_file).st_ino, kept_ino)
        self.assertEqual(kept_file.stat().st_mtime, kept_mtime)

        original = {}
        for fname in self.shards:
            original.update(load_file(str(self.root / fname)))
        recovered = load_file(str(_split_path(out, missing)))
        for k, v in recovered.items():
            self.assertTrue(torch.equal(v, original[k]), f"{k} changed on resume")

    def test_incomplete_payload_without_done_marker_is_rewritten(self):
        out = self._split()
        target = "lm_head"
        payload = _split_path(out, target)
        payload.write_bytes(b"truncated")
        (out / f"{target}.safetensors.done").unlink()

        self._split()
        sd = load_file(str(payload))
        self.assertIn("lm_head.weight", sd)
        self.assertEqual(tuple(sd["lm_head.weight"].shape), (16, 8))

    def test_remove_incomplete_skips_complete_layers(self):
        out = self._split()
        layer = "model.layers.2."
        self.assertFalse(remove_incomplete_layer_artifacts(layer, out))
        self.assertTrue(_split_path(out, "model.layers.2").exists())

    def test_second_call_is_a_no_op_when_everything_is_saved(self):
        out = self._split()
        old = time.time() - 120
        files = list(out.glob("*.safetensors"))
        for f in files:
            _stamp(f, old)
        mtimes = {f.name: f.stat().st_mtime for f in files}

        self._split()
        for f in files:
            self.assertEqual(f.stat().st_mtime, mtimes[f.name], f"{f.name} was rewritten")


if __name__ == '__main__':
    unittest.main(verbosity=2)
