import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from ..airllm.preflight import inspect_model
from ..airllm.utils import checkpoint_total_size_bytes, estimated_split_size_bytes, split_and_save_layers


class TestPreflight(unittest.TestCase):
    def _write_model_metadata(self, root, architecture, prefix, total_size=10_000, num_hidden_layers=None):
        config = {
            "architectures": [architecture],
            "transformers_version": "5.12.0",
        }
        if num_hidden_layers is not None:
            config["num_hidden_layers"] = num_hidden_layers
        weight_map = {
            f"{prefix}.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            f"{prefix}.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}),
            encoding="utf-8",
        )

    def test_glm_standard_layout_and_size_estimates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_model_metadata(root, "GlmMoeDsaForCausalLM", "model.layers")
            report = inspect_model(str(root), compression="4bit", offload_dir=root)

        self.assertEqual(report.compatibility, "supported")
        self.assertEqual(report.layer_count, 2)
        self.assertEqual(report.checkpoint_size_bytes, 10_000)
        self.assertEqual(report.estimated_split_size_bytes, 2_813)
        self.assertIn("transformers>=5.12,<5.13", report.required_packages)

    def test_kimi_uses_nested_text_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_model_metadata(
                root,
                "KimiK25ForConditionalGeneration",
                "language_model.model.layers",
            )
            report = inspect_model(str(root))

        self.assertEqual(report.compatibility, "text-only")
        self.assertEqual(report.layer_layout["lm_head"], "language_model.lm_head")
        self.assertEqual(report.layer_count, 2)
        self.assertTrue(any("text-only" in warning for warning in report.warnings))
        self.assertIn("transformers>=4.57.1,<5.0.0", report.required_packages)

    def test_checkpoint_size_prefers_index_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_model_metadata(root, "LlamaForCausalLM", "model.layers", total_size=123_456)
            self.assertEqual(checkpoint_total_size_bytes(root), 123_456)

    def test_configured_layer_count_excludes_auxiliary_index_layers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_model_metadata(
                root,
                "GlmMoeDsaForCausalLM",
                "model.layers",
                num_hidden_layers=1,
            )
            report = inspect_model(str(root))

        self.assertEqual(report.layer_count, 1)

    def test_split_excludes_auxiliary_index_layers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text(
                json.dumps({"num_hidden_layers": 1}),
                encoding="utf-8",
            )
            save_file(
                {
                    "model.embed_tokens.weight": torch.ones(2, 2),
                    "model.layers.0.weight": torch.ones(2, 2),
                    "model.layers.1.weight": torch.ones(2, 2),
                    "model.norm.weight": torch.ones(2),
                    "lm_head.weight": torch.ones(2, 2),
                },
                root / "model.safetensors",
            )
            split_path = Path(
                split_and_save_layers(
                    root,
                    layer_names={
                        "embed": "model.embed_tokens",
                        "layer_prefix": "model.layers",
                        "norm": "model.norm",
                        "lm_head": "lm_head",
                    },
                )
            )

            saved_names = {path.name for path in split_path.glob("*.safetensors")}

        self.assertIn("model.layers.0.safetensors", saved_names)
        self.assertNotIn("model.layers.1.safetensors", saved_names)

    def test_delete_original_removes_final_multishard_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text(
                json.dumps({"num_hidden_layers": 1}), encoding="utf-8"
            )
            shard_one = root / "model-00001-of-00002.safetensors"
            shard_two = root / "model-00002-of-00002.safetensors"
            save_file(
                {
                    "model.embed_tokens.weight": torch.ones(2, 2),
                    "model.layers.0.weight": torch.ones(2, 2),
                },
                shard_one,
            )
            save_file(
                {
                    "model.norm.weight": torch.ones(2),
                    "lm_head.weight": torch.ones(2, 2),
                },
                shard_two,
            )
            weight_map = {
                "model.embed_tokens.weight": shard_one.name,
                "model.layers.0.weight": shard_one.name,
                "model.norm.weight": shard_two.name,
                "lm_head.weight": shard_two.name,
            }
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": shard_one.stat().st_size + shard_two.stat().st_size},
                        "weight_map": weight_map,
                    }
                ),
                encoding="utf-8",
            )

            split_and_save_layers(root, delete_original=True)

            self.assertFalse(shard_one.exists())
            self.assertFalse(shard_two.exists())

    def test_compression_ratios(self):
        self.assertEqual(estimated_split_size_bytes(10_000, None), 10_000)
        self.assertEqual(estimated_split_size_bytes(10_000, "8bit"), 5_000)
        self.assertEqual(estimated_split_size_bytes(10_000, "4bit"), 2_813)
        with self.assertRaises(ValueError):
            estimated_split_size_bytes(10_000, "3bit")

    def test_prequantized_checkpoint_rejects_stacked_compression(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_model_metadata(root, "GlmMoeDsaForCausalLM", "model.layers")
            config_path = root / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["quantization_config"] = {"quant_method": "fp8"}
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaises(ValueError):
                inspect_model(str(root), compression="4bit")
