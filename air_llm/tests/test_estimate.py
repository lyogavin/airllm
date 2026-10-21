"""Disk / VRAM dry-run estimates against a tiny local checkpoint (no Hub download)."""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

_AIRLLM_DIR = Path(__file__).resolve().parents[1] / 'airllm'

if 'airllm' not in sys.modules:
    _pkg = types.ModuleType('airllm')
    _pkg.__path__ = [str(_AIRLLM_DIR)]
    sys.modules['airllm'] = _pkg

from airllm.persist import model_persister as _persister_mod
from airllm.persist.safetensor_model_persister import SafetensorModelPersister

_persister_mod.model_persister = SafetensorModelPersister()

from airllm.estimate import estimate_resources, format_resource_estimate
from airllm.utils import split_and_save_layers

N_LAYERS = 3


def build_fake_checkpoint(root):
    shards = {
        "model-00001-of-00002.safetensors": {
            "model.embed_tokens.weight": torch.randn(32, 16),
            "model.layers.0.input_layernorm.weight": torch.randn(16),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 16),
            "model.layers.1.input_layernorm.weight": torch.randn(16),
            "model.layers.1.self_attn.q_proj.weight": torch.randn(16, 16),
        },
        "model-00002-of-00002.safetensors": {
            "model.layers.2.input_layernorm.weight": torch.randn(16),
            "model.layers.2.self_attn.q_proj.weight": torch.randn(16, 16),
            "model.norm.weight": torch.randn(16),
            "lm_head.weight": torch.randn(32, 16),
        },
    }
    weight_map = {}
    for fname, sd in shards.items():
        save_file(sd, str(Path(root) / fname))
        for k in sd:
            weight_map[k] = fname
    with open(Path(root) / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 1}, "weight_map": weight_map}, f)
    with open(Path(root) / "config.json", "w") as f:
        json.dump({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": N_LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 4,
            "torch_dtype": "float16",
            "max_position_embeddings": 128,
        }, f)
    return shards


class TestEstimateResources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.shards = build_fake_checkpoint(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _weight_bytes(self):
        return sum(os.path.getsize(self.root / name) for name in self.shards)

    def test_local_estimate_matches_checkpoint_size_and_does_not_split(self):
        info = estimate_resources(str(self.root))
        self.assertEqual(info['source'], 'local')
        self.assertEqual(info['checkpoint_bytes'], self._weight_bytes())
        self.assertEqual(info['n_shards'], 2)
        self.assertEqual(info['n_layers'], N_LAYERS)
        self.assertEqual(info['n_modules'], 6)  # embed + 3 layers + norm + lm_head
        self.assertEqual(info['split_bytes'], info['checkpoint_bytes'])
        self.assertEqual(info['disk_keep_both_bytes'], 2 * info['checkpoint_bytes'])
        self.assertGreater(info['largest_layer_bytes'], 0)
        self.assertGreater(info['kv_cache_bytes'], 0)
        self.assertGreater(info['estimated_peak_vram_bytes'], info['largest_layer_bytes'])
        self.assertFalse(info['already_split'])
        self.assertIsNone(info['split_cache_path'])
        self.assertFalse((self.root / 'splitted_model').exists())

    def test_report_mentions_disk_and_vram(self):
        text = format_resource_estimate(estimate_resources(str(self.root)))
        self.assertIn('AirLLM resource estimate', text)
        self.assertIn('estimated peak VRAM', text)
        self.assertIn('checkpoint:', text)
        self.assertIn('layer cache:', text)

    def test_compression_shrinks_estimated_split_copy(self):
        raw = estimate_resources(str(self.root))
        four = estimate_resources(str(self.root), compression='4bit')
        eight = estimate_resources(str(self.root), compression='8bit')
        self.assertLess(four['split_bytes'], raw['split_bytes'])
        self.assertLess(eight['split_bytes'], raw['split_bytes'])
        self.assertGreater(eight['split_bytes'], four['split_bytes'])

    def test_existing_layer_cache_is_reported(self):
        split_and_save_layers(self.root)
        info = estimate_resources(str(self.root))
        self.assertTrue(info['already_split'])
        self.assertIsNotNone(info['split_cache_path'])
        self.assertGreater(info['split_cache_bytes'], 0)
        self.assertGreater(info['split_cache_files'], 0)
        text = format_resource_estimate(info)
        self.assertIn('splitted_model', text)

    def test_override_layer_names_match_streaming(self):
        from airllm.estimate import _arch_layer_names
        expected = {
            "Qwen3_5ForConditionalGeneration": "model.language_model.layers",
            "Qwen4ExpForConditionalGeneration": "model.language_model.layers",
            "KimiK3ForConditionalGeneration": "language_model.model.layers",
        }
        for arch, prefix in expected.items():
            names = _arch_layer_names(arch)
            self.assertEqual(names["layer_prefix"], prefix)
            self.assertNotEqual(prefix, "model.layers")

    def test_nested_vl_decoder_is_not_model_layers(self):
        root = self.root / "vl"
        root.mkdir()
        ngram = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
        shards = {
            "layer0.safetensors": {
                "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
                "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 32, 32),
            },
            "ngram.safetensors": {f"{ngram}.weight": torch.randn(256, 256)},
            "rest.safetensors": {
                "model.visual.patch_embed.proj.weight": torch.randn(16, 8),
                "model.language_model.embed_tokens.weight": torch.randn(8, 8),
                "model.language_model.layers.1.self_attn.q_proj.weight": torch.randn(8, 8),
                "model.language_model.hyper_connection_mixer.hc_norm.weight": torch.randn(8),
                "lm_head.weight": torch.randn(8, 8),
            },
        }
        weight_map = {}
        for fname, sd in shards.items():
            save_file(sd, str(root / fname))
            weight_map.update({k: fname for k in sd})
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (root / "config.json").write_text(json.dumps({
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "text_config": {
                "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 4, "num_experts": 4,
            },
            "torch_dtype": "float16",
        }))
        info = estimate_resources(str(root))
        largest = (info["largest_layer"] or "").rstrip(".")
        self.assertTrue(largest.startswith("model.language_model.layers."))
        self.assertFalse(largest.startswith("model.layers."))
        self.assertNotIn("ngram", largest)
        self.assertNotIn("visual", largest)
        self.assertEqual(info["n_layers"], 2)
        self.assertLess(info["largest_layer_bytes"], os.path.getsize(root / "ngram.safetensors"))
        self.assertGreater(
            info["estimated_peak_vram_bytes"],
            info["largest_layer_bytes"] + info["kv_cache_bytes"] + 256 * 1024 * 1024,
        )
        src = (_AIRLLM_DIR / "auto_model.py").read_text()
        self.assertIn("def estimate(", src)
        self.assertNotIn("dry_run", src)
        self.assertIn("estimate_resources", (_AIRLLM_DIR / "__init__.py").read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
