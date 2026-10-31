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

from airllm.utils import (
    split_and_save_layers,
    remove_incomplete_layer_artifacts,
    open_ngram_mmap_table,
)

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



def _save_checkpoint(root, shards):
    weight_map = {}
    for fname, sd in shards.items():
        save_file(sd, str(Path(root) / fname))
        for k in sd:
            weight_map[k] = fname
    with open(Path(root) / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 1}, "weight_map": weight_map}, f)


def _snapshot(path):
    st = os.stat(path)
    return st.st_ino, st.st_mtime


FLASH_PREFIX = "model.language_model.layers"
FLASH_NGRAM = f"{FLASH_PREFIX}.1.ple.ple_embedding.ngram_embedding"
FLASH_LAYER_NAMES = {
    'embed': 'model.language_model.embed_tokens',
    'layer_prefix': FLASH_PREFIX,
    'norm': 'model.language_model.hyper_connection_mixer',
    'lm_head': 'lm_head',
    'cpu_resident_marker': 'ple.ple_embedding.ngram_embedding',
}


class TestResumeFlashNextNgramSibling(unittest.TestCase):
    """Layer 1 is peeled into ``...ngram_embedding.mmap`` plus ``layers.1.safetensors``.

    Losing one of the two siblings must not rewrite the other, which still has its done marker.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.shards = {
            "model-00001-of-00002.safetensors": {
                "model.language_model.embed_tokens.weight": torch.randn(16, 8),
                f"{FLASH_PREFIX}.0.self_attn.q_proj.weight": torch.randn(8, 8),
            },
            "model-00002-of-00002.safetensors": {
                f"{FLASH_PREFIX}.1.mlp.experts.gate_up_proj": torch.randn(4, 8, 8),
                f"{FLASH_PREFIX}.1.ple.key_proj.weight": torch.randn(8, 8),
                f"{FLASH_NGRAM}.shard_0.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                f"{FLASH_NGRAM}.shard_1.weight": torch.arange(12, 24, dtype=torch.float32).reshape(3, 4),
                "model.language_model.hyper_connection_mixer.hc_norm.weight": torch.randn(8),
                "lm_head.weight": torch.randn(16, 8),
            },
        }
        _save_checkpoint(self.root, self.shards)
        self.out = self._split()
        old = time.time() - 120
        for f in self.out.iterdir():
            os.utime(f, (old, old))
        self.layer1 = self.out / f"{FLASH_PREFIX}.1.safetensors"
        self.mmap = self.out / f"{FLASH_NGRAM}.mmap"
        self.table = torch.cat([self.shards["model-00002-of-00002.safetensors"][f"{FLASH_NGRAM}.shard_{i}.weight"]
                                for i in range(2)], dim=0)

    def tearDown(self):
        self.tmp.cleanup()

    def _split(self):
        return Path(split_and_save_layers(self.root, layer_names=FLASH_LAYER_NAMES))

    def test_deleting_the_parent_layer_keeps_the_ngram_mmap(self):
        mmap_files = [self.out / f"{FLASH_NGRAM}.{s}" for s in ("mmap", "mmap.json", "mmap.done")]
        before = {f.name: _snapshot(f) for f in mmap_files}
        self.layer1.unlink()
        (self.out / f"{FLASH_PREFIX}.1.safetensors.done").unlink()

        self._split()

        self.assertEqual({f.name: _snapshot(f) for f in mmap_files}, before, "n-gram mmap was rewritten")
        self.assertTrue((self.out / f"{FLASH_PREFIX}.1.safetensors.done").exists())
        layer1 = load_file(str(self.layer1))
        src = self.shards["model-00002-of-00002.safetensors"]
        self.assertEqual(set(layer1), {f"{FLASH_PREFIX}.1.mlp.experts.gate_up_proj",
                                       f"{FLASH_PREFIX}.1.ple.key_proj.weight"})
        for k, v in layer1.items():
            self.assertTrue(torch.equal(v, src[k]), f"{k} changed on resume")
        self.assertTrue(torch.equal(open_ngram_mmap_table(self.out, FLASH_NGRAM), self.table))

    def test_deleting_only_the_mmap_keeps_the_parent_layer(self):
        before = _snapshot(self.layer1)
        self.mmap.unlink()

        self._split()

        self.assertEqual(_snapshot(self.layer1), before, "parent layer 1 was rewritten")
        self.assertTrue((self.out / f"{FLASH_PREFIX}.1.safetensors.done").exists())
        self.assertTrue((self.out / f"{FLASH_NGRAM}.mmap.done").exists())
        self.assertTrue(torch.equal(open_ngram_mmap_table(self.out, FLASH_NGRAM), self.table))


KIMI_PREFIX = "language_model.model.layers"
KIMI_LAYER_NAMES = {
    'embed': 'language_model.model.embed_tokens',
    'layer_prefix': KIMI_PREFIX,
    'norm': 'language_model.model.norm',
    'lm_head': 'language_model.lm_head',
}


class TestResumePassthroughLinks(unittest.TestCase):
    """Kimi-style one-file-per-layer shards are hard-linked into the split dir.

    Resume must never unlink or rewrite the original checkpoint file behind such a link.
    """

    N = 3

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        n_shards = self.N + 1
        self.shards = {}
        for i in range(self.N):
            self.shards[f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"] = {
                f"{KIMI_PREFIX}.{i}.mlp.w1.weight": torch.randn(8, 8),
                f"{KIMI_PREFIX}.{i}.input_layernorm.weight": torch.randn(8),
            }
        self.shards[f"model-{n_shards:05d}-of-{n_shards:05d}.safetensors"] = {
            "language_model.model.embed_tokens.weight": torch.randn(16, 8),
            "language_model.model.norm.weight": torch.randn(8),
            "language_model.lm_head.weight": torch.randn(16, 8),
        }
        _save_checkpoint(self.root, self.shards)
        self.out = self._split()

    def tearDown(self):
        self.tmp.cleanup()

    def _split(self):
        return Path(split_and_save_layers(self.root, layer_names=KIMI_LAYER_NAMES))

    def _src(self, i):
        return self.root / f"model-{i + 1:05d}-of-{self.N + 1:05d}.safetensors"

    def _dst(self, i):
        return self.out / f"{KIMI_PREFIX}.{i}.safetensors"

    def _assert_originals_intact(self):
        for fname, sd in self.shards.items():
            path = self.root / fname
            self.assertTrue(path.exists(), f"original {fname} was removed")
            loaded = load_file(str(path))
            self.assertEqual(set(loaded), set(sd))
            for k, v in sd.items():
                self.assertTrue(torch.equal(loaded[k], v), f"{k} changed in original {fname}")

    def test_linked_layer_missing_its_marker_is_not_unlinked(self):
        """A crash between link and marker leaves a complete link with no ``.done``."""
        self.assertEqual(os.stat(self._src(1)).st_ino, os.stat(self._dst(1)).st_ino)
        # Any unlink or link of the shared inode bumps its ctime, so an unchanged ctime proves resume
        # left the link (and the original checkpoint file behind it) alone.
        before = (os.stat(self._dst(1)).st_ino, os.stat(self._src(1)).st_ctime_ns)
        (self.out / f"{KIMI_PREFIX}.1.safetensors.done").unlink()
        time.sleep(0.05)

        self._split()

        self.assertEqual((os.stat(self._dst(1)).st_ino, os.stat(self._src(1)).st_ctime_ns), before,
                         "linked layer was unlinked and relinked")
        self.assertTrue((self.out / f"{KIMI_PREFIX}.1.safetensors.done").exists())
        self.assertEqual(os.stat(self._src(1)).st_nlink, 2)
        self._assert_originals_intact()

    def test_resume_relinks_a_deleted_layer_and_keeps_originals(self):
        self._dst(1).unlink()
        (self.out / f"{KIMI_PREFIX}.1.safetensors.done").unlink()
        embed = self.out / "language_model.model.embed_tokens.safetensors"
        embed.unlink()
        (self.out / "language_model.model.embed_tokens.safetensors.done").unlink()

        self._split()

        for i in range(self.N):
            self.assertEqual(os.stat(self._src(i)).st_ino, os.stat(self._dst(i)).st_ino,
                             f"layer {i} is no longer linked to its original")
        src = self.shards[f"model-{self.N + 1:05d}-of-{self.N + 1:05d}.safetensors"]
        self.assertTrue(torch.equal(load_file(str(embed))["language_model.model.embed_tokens.weight"],
                                    src["language_model.model.embed_tokens.weight"]))
        self._assert_originals_intact()


if __name__ == '__main__':
    unittest.main(verbosity=2)
