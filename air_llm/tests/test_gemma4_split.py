"""Structural regression test for Gemma 4 multimodal MoE checkpoints."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from transformers.models.gemma4 import (
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)

import airllm
from airllm.auto_model import AutoConfig, AutoModel
from airllm.utils import split_and_save_layers


ARCHITECTURE = "Gemma4ForConditionalGeneration"
CLASS_NAME = "AirLLMGemma4"
LAYER_PREFIX = "model.language_model.layers"
SHARD_FILE = "model-00001-of-00001.safetensors"
INDEX_FILE = "model.safetensors.index.json"
MODEL_SIZE = 8
VOCAB_SIZE = 16
N_LAYERS = 2
N_HEADS = 2
N_KEY_VALUE_HEADS = 1
HEAD_SIZE = 4
INTERMEDIATE_SIZE = 16
MAX_POSITION = 32
SLIDING_WINDOW = 8
LAYER_TYPES = ["sliding_attention", "full_attention"]
N_EXPERTS = 2
TOP_K_EXPERTS = 1
MOE_INTERMEDIATE_SIZE = 8
HIDDEN_SIZE_PER_LAYER_INPUT = 4
VISION_LAYERS = 1
PATCH_SIZE = 2
POSITION_EMBEDDING_SIZE = 16
POOLING_KERNEL_SIZE = 1
IMAGE_TOKEN_ID = 15
BEGIN_IMAGE_TOKEN_ID = 14
END_IMAGE_TOKEN_ID = 13
MODULE_PATHS = (
    "model.language_model.embed_tokens",
    LAYER_PREFIX,
    "model.language_model.norm",
    "model.language_model.embed_tokens_per_layer",
    "model.language_model.per_layer_model_projection",
    "model.language_model.per_layer_projection_norm",
    "lm_head",
    "model.embed_vision",
    "model.vision_tower",
)


def build_tiny_model():
    text_config = Gemma4TextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=MODEL_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KEY_VALUE_HEADS,
        head_dim=HEAD_SIZE,
        max_position_embeddings=MAX_POSITION,
        sliding_window=SLIDING_WINDOW,
        layer_types=LAYER_TYPES,
        vocab_size_per_layer_input=VOCAB_SIZE,
        hidden_size_per_layer_input=HIDDEN_SIZE_PER_LAYER_INPUT,
        num_global_key_value_heads=N_KEY_VALUE_HEADS,
        global_head_dim=HEAD_SIZE,
        enable_moe_block=True,
        num_experts=N_EXPERTS,
        top_k_experts=TOP_K_EXPERTS,
        moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
    )
    vision_config = Gemma4VisionConfig(
        hidden_size=MODEL_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=VISION_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        head_dim=HEAD_SIZE,
        max_position_embeddings=POSITION_EMBEDDING_SIZE,
        patch_size=PATCH_SIZE,
        position_embedding_size=POSITION_EMBEDDING_SIZE,
        pooling_kernel_size=POOLING_KERNEL_SIZE,
    )
    config = Gemma4Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=IMAGE_TOKEN_ID,
        boi_token_id=BEGIN_IMAGE_TOKEN_ID,
        eoi_token_id=END_IMAGE_TOKEN_ID,
    )
    return Gemma4ForConditionalGeneration(config)


def build_fake_checkpoint(root, model):
    tensors = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    save_file(tensors, str(root / SHARD_FILE))
    weight_map = {name: SHARD_FILE for name in tensors}
    (root / INDEX_FILE).write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return tensors


class TestGemma4Split(unittest.TestCase):
    def test_nested_language_model_and_fused_experts_split(self):
        config = SimpleNamespace(architectures=[ARCHITECTURE])
        with patch.object(AutoConfig, "from_pretrained", return_value=config):
            module_name, class_name = AutoModel.get_module_class("unused")

        self.assertEqual((module_name, class_name), ("airllm", CLASS_NAME))
        model_class = getattr(airllm, class_name)
        model = model_class.__new__(model_class)
        model.set_layer_names_dict()

        runtime_model = build_tiny_model()
        for path in MODULE_PATHS:
            module = runtime_model
            for component in path.split("."):
                module = getattr(module, component)
        self.assertEqual(len(runtime_model.model.language_model.layers), N_LAYERS)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = build_fake_checkpoint(root, runtime_model)
            output = Path(split_and_save_layers(root, layer_names=model.layer_names_dict))
            recovered = {}
            for split_file in output.glob("*.safetensors"):
                recovered.update(load_file(str(split_file)))

        self.assertEqual(set(recovered), set(original))
        for name, tensor in original.items():
            self.assertTrue(torch.equal(recovered[name], tensor), f"tensor changed: {name}")

        load_result = build_tiny_model().load_state_dict(recovered, strict=False)
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
