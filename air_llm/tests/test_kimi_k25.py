import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ..airllm.airllm_base import AirLLMBaseModel
from ..airllm.airllm_kimi_k25 import AirLLMKimiK25


class TestKimiK25(unittest.TestCase):
    def test_layer_layout(self):
        model = AirLLMKimiK25.__new__(AirLLMKimiK25)
        model.set_layer_names_dict()
        self.assertEqual(model.layer_names_dict["embed"], "language_model.model.embed_tokens")
        self.assertEqual(model.layer_names_dict["layer_prefix"], "language_model.model.layers")
        self.assertEqual(model.layer_names_dict["norm"], "language_model.model.norm")
        self.assertEqual(model.layer_names_dict["lm_head"], "language_model.lm_head")

    def test_visual_inputs_are_rejected(self):
        with self.assertRaises(NotImplementedError):
            AirLLMKimiK25._reject_visual_inputs({"pixel_values": object()})

        AirLLMKimiK25._reject_visual_inputs({"pixel_values": None, "grid_thws": None})

    @patch.object(AirLLMBaseModel, "init_model")
    def test_vision_flash_attention_is_disabled_for_text_only_loader(self, base_init):
        model = AirLLMKimiK25.__new__(AirLLMKimiK25)
        model.config = SimpleNamespace(
            vision_config=SimpleNamespace(
                _attn_implementation="flash_attention_2",
                _attn_implementation_internal="flash_attention_2",
            )
        )

        model.init_model()

        self.assertEqual(model.config.vision_config._attn_implementation, "eager")
        self.assertEqual(model.config.vision_config._attn_implementation_internal, "eager")
        base_init.assert_called_once_with()
