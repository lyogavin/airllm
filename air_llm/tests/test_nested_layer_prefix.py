"""Nested decoder layouts must not be parsed as if `model.layers` were a substring.

Gemma 4 stores the decoder at ``model.language_model.layers.<int>`` and a vision
tower at ``model.vision_tower.encoder.layers.<int>``. The old counter did
``'model.layers' in key`` and then ``int('layers')``.
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_AIRLLM_DIR = Path(__file__).resolve().parents[1] / 'airllm'

if 'airllm' not in sys.modules:
    _pkg = types.ModuleType('airllm')
    _pkg.__path__ = [str(_AIRLLM_DIR)]
    sys.modules['airllm'] = _pkg

from airllm.persist import model_persister as _persister_mod
from airllm.persist.safetensor_model_persister import SafetensorModelPersister

_persister_mod.model_persister = SafetensorModelPersister()

from airllm.utils import align_layer_names, layer_ids_for_prefix, split_and_save_layers

GENERIC = {
    'embed': 'model.embed_tokens',
    'layer_prefix': 'model.layers',
    'norm': 'model.norm',
    'lm_head': 'lm_head',
}

GEMMA_KEYS = [
    'model.language_model.embed_tokens.weight',
    'model.language_model.layers.0.self_attn.q_proj.weight',
    'model.language_model.layers.0.experts.down_proj.weight',
    'model.language_model.layers.1.self_attn.q_proj.weight',
    'model.language_model.norm.weight',
    'lm_head.weight',
    'model.vision_tower.encoder.layers.0.attn.weight',
    'model.embed_vision.embedding_projection.weight',
]


class TestLayerPrefixMatching(unittest.TestCase):
    def test_model_layers_does_not_match_inside_language_model(self):
        ids = layer_ids_for_prefix(GEMMA_KEYS, 'model.layers')
        self.assertEqual(ids, set())

    def test_generic_names_retarget_to_the_language_model(self):
        names = dict(GENERIC)
        ids = align_layer_names(names, GEMMA_KEYS)
        self.assertEqual(ids, {0, 1})
        self.assertEqual(names['layer_prefix'], 'model.language_model.layers')
        self.assertEqual(names['embed'], 'model.language_model.embed_tokens')
        self.assertEqual(names['norm'], 'model.language_model.norm')
        self.assertEqual(names['lm_head'], 'lm_head')
        self.assertIn('model.vision_tower', names['resident'])
        self.assertIn('model.embed_vision', names['resident'])
        self.assertNotIn('model.vision_tower.encoder.layers', names['layer_prefix'])

    def test_standard_prefix_is_not_rewritten(self):
        names = dict(GENERIC)
        keys = [
            'model.embed_tokens.weight',
            'model.layers.0.self_attn.q_proj.weight',
            'model.layers.1.self_attn.q_proj.weight',
            'model.norm.weight',
            'lm_head.weight',
        ]
        ids = align_layer_names(names, keys)
        self.assertEqual(ids, {0, 1})
        self.assertEqual(names['layer_prefix'], 'model.layers')
        self.assertNotIn('resident', names)


class TestGemmaStyleSplit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shard = {
            'model.language_model.embed_tokens.weight': torch.randn(4, 4),
            'model.language_model.layers.0.self_attn.q_proj.weight': torch.randn(4, 4),
            'model.language_model.layers.0.experts.down_proj.weight': torch.randn(4, 4),
            'model.language_model.layers.1.self_attn.q_proj.weight': torch.randn(4, 4),
            'model.language_model.norm.weight': torch.randn(4),
            'lm_head.weight': torch.randn(4, 4),
            'model.vision_tower.encoder.layers.0.attn.weight': torch.randn(4, 4),
            'model.embed_vision.embedding_projection.weight': torch.randn(4, 4),
        }
        save_file(shard, str(self.root / 'model.safetensors'))
        self.original = shard
        self.out = Path(split_and_save_layers(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def test_decoder_and_side_modules_are_split(self):
        expected = [
            'model.language_model.embed_tokens',
            'model.language_model.layers.0',
            'model.language_model.layers.1',
            'model.language_model.norm',
            'lm_head',
            'model.vision_tower',
            'model.embed_vision',
        ]
        for module in expected:
            self.assertTrue((self.out / f'{module}.safetensors').exists(), module)

    def test_fused_expert_stays_on_its_decoder_layer(self):
        sd = load_file(str(self.out / 'model.language_model.layers.0.safetensors'))
        self.assertIn('model.language_model.layers.0.experts.down_proj.weight', sd)
        self.assertNotIn('model.language_model.layers.1.self_attn.q_proj.weight', sd)

    def test_round_trip(self):
        recovered = {}
        for path in self.out.glob('*.safetensors'):
            recovered.update(load_file(str(path)))
        self.assertEqual(set(self.original), set(recovered))
        for key, value in self.original.items():
            self.assertTrue(torch.equal(value, recovered[key]), key)


if __name__ == '__main__':
    unittest.main(verbosity=2)
