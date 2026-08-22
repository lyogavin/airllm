"""Structural tests for Qwen3.5 / Qwen3.8 VL checkpoints.

Qwen3.8-27B is ``Qwen3_5ForConditionalGeneration``: the decoder is nested under
``model.language_model``, the vision tower is ``model.visual``, and the checkpoint also carries an
``mtp.*`` Multi-Token Prediction head that transformers ignores. These tests build a miniature
checkpoint with that shape (several layers per shard, like the real 18-file dump) and check that
the splitter writes every streamed module, keeps the vision tower, and drops MTP.
"""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

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
from airllm.airllm_base import AirLLMBaseModel

N_LAYERS = 4
LAYER_PREFIX = "model.language_model.layers"

QWEN38_LAYER_NAMES = {
    'embed': 'model.language_model.embed_tokens',
    'layer_prefix': LAYER_PREFIX,
    'norm': 'model.language_model.norm',
    'lm_head': 'lm_head',
    'resident': [
        'model.visual',
    ],
}


def build_fake_checkpoint(root):
    """Several decoder layers per shard, plus a vision tower mixed into shard 0 and MTP in the last."""
    shards = {}

    shard0 = {
        "model.visual.patch_embed.proj.weight": torch.randn(8, 3, 2, 2),
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(8, 8),
        "model.visual.merger.linear_fc1.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.0.input_layernorm.weight": torch.randn(8),
        f"{LAYER_PREFIX}.0.linear_attn.A_log": torch.randn(4),
        f"{LAYER_PREFIX}.0.linear_attn.in_proj_a.weight": torch.randn(4, 8),
        f"{LAYER_PREFIX}.0.mlp.gate_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.1.input_layernorm.weight": torch.randn(8),
        f"{LAYER_PREFIX}.1.self_attn.q_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.1.mlp.down_proj.weight": torch.randn(8, 8),
    }
    shards["model-00001-of-00003.safetensors"] = shard0

    shard1 = {
        "model.language_model.embed_tokens.weight": torch.randn(16, 8),
        f"{LAYER_PREFIX}.2.input_layernorm.weight": torch.randn(8),
        f"{LAYER_PREFIX}.2.linear_attn.out_proj.weight": torch.randn(8, 8),
        f"{LAYER_PREFIX}.3.input_layernorm.weight": torch.randn(8),
        f"{LAYER_PREFIX}.3.self_attn.o_proj.weight": torch.randn(8, 8),
        "model.language_model.norm.weight": torch.randn(8),
    }
    shards["model-00002-of-00003.safetensors"] = shard1

    shard2 = {
        "lm_head.weight": torch.randn(16, 8),
        "mtp.fc.weight": torch.randn(8, 8),
        "mtp.layers.0.mlp.gate_proj.weight": torch.randn(8, 8),
        "mtp.norm.weight": torch.randn(8),
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


class TestQwen38Split(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.shards, self.weight_map = build_fake_checkpoint(self.root)
        self.out = Path(split_and_save_layers(self.root, layer_names=QWEN38_LAYER_NAMES))

    def tearDown(self):
        self.tmp.cleanup()

    def _split_file(self, module):
        return self.out / f"{module}.safetensors"

    def test_architecture_maps_to_the_qwen3_5_subclass(self):
        src = (_AIRLLM_DIR / 'auto_model.py').read_text()
        self.assertIn('"Qwen3_5ForConditionalGeneration": "AirLLMQwen3_5"', src)
        init_src = (_AIRLLM_DIR / '__init__.py').read_text()
        self.assertIn('AirLLMQwen3_5', init_src)

    def test_every_streamed_and_resident_module_is_split_out(self):
        expected = ([QWEN38_LAYER_NAMES['embed']]
                    + [f"{LAYER_PREFIX}.{i}" for i in range(N_LAYERS)]
                    + [QWEN38_LAYER_NAMES['norm'], QWEN38_LAYER_NAMES['lm_head']]
                    + QWEN38_LAYER_NAMES['resident'])
        for module in expected:
            self.assertTrue(self._split_file(module).exists(), f"missing split for {module}")

    def test_vision_tower_is_one_shard_and_excludes_decoder_weights(self):
        sd = load_file(str(self._split_file("model.visual")))
        self.assertTrue(sd, "vision split is empty")
        self.assertTrue(all(k.startswith("model.visual.") for k in sd),
                        f"vision split leaked decoder tensors: {set(sd)}")
        self.assertIn("model.visual.patch_embed.proj.weight", sd)
        self.assertIn("model.visual.blocks.0.attn.qkv.weight", sd)

    def test_mtp_is_dropped(self):
        """transformers ignores mtp.*; keeping it would waste a resident slot for a dead module."""
        recovered = {}
        for f in self.out.glob('*.safetensors'):
            recovered.update(load_file(str(f)))
        self.assertFalse(any(k.startswith("mtp.") for k in recovered),
                         f"MTP leaked into the split: {[k for k in recovered if k.startswith('mtp.')]}")
        self.assertFalse(any(p.name.startswith("mtp") for p in self.out.glob('*.safetensors')))

    def test_decoder_and_head_round_trip(self):
        original = {}
        for fname in self.shards:
            original.update(load_file(str(self.root / fname)))
        keep = {k: v for k, v in original.items() if not k.startswith("mtp.")}

        recovered = {}
        for f in self.out.glob('*.safetensors'):
            recovered.update(load_file(str(f)))

        self.assertEqual(set(keep.keys()), set(recovered.keys()))
        for k, v in keep.items():
            self.assertEqual(v.dtype, recovered[k].dtype, f"{k} changed dtype")
            self.assertTrue(torch.equal(v, recovered[k]), f"{k} changed value")

    def test_shared_shards_are_copied_not_linked(self):
        """The real 27B dump packs several layers per file, so passthrough must not fire."""
        src_inodes = {os.stat(self.root / f).st_ino
                      for f in os.listdir(self.root) if f.endswith('.safetensors')}
        for module in ([f"{LAYER_PREFIX}.{i}" for i in range(N_LAYERS)]
                       + ['model.language_model.embed_tokens', 'model.visual', 'lm_head']):
            dst = self._split_file(module)
            self.assertNotIn(os.stat(dst).st_ino, src_inodes,
                             f"{module} was linked, but this layout should be copied")

    def test_text_only_mode_does_not_materialize_resident_vision_tower(self):
        model = AirLLMBaseModel.__new__(AirLLMBaseModel)
        model.load_resident_modules = False
        model.layer_names_dict = {'resident': ['model.visual']}
        model.load_layer_to_cpu = Mock()

        model._load_resident_modules()

        model.load_layer_to_cpu.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
