import unittest
from types import SimpleNamespace

import torch

from airllm.airllm_base import AirLLMBaseModel


class TestPrequantizedBitsAndBytes(unittest.TestCase):
    def test_quantization_state_keys_collapse_to_weight(self):
        state_dict = {
            'model.layers.0.mlp.gate_proj.weight': object(),
            'model.layers.0.mlp.gate_proj.weight.absmax': object(),
            'model.layers.0.mlp.gate_proj.weight.nested_absmax': object(),
            'model.layers.0.mlp.gate_proj.weight.nested_quant_map': object(),
            'model.layers.0.mlp.gate_proj.weight.quant_map': object(),
            'model.layers.0.mlp.gate_proj.weight.quant_state.bitsandbytes__nf4': object(),
            'model.layers.0.input_layernorm.weight': object(),
        }

        model = AirLLMBaseModel.__new__(AirLLMBaseModel)
        self.assertEqual(
            model._param_names_from_state_dict(state_dict),
            [
                'model.layers.0.mlp.gate_proj.weight',
                'model.layers.0.input_layernorm.weight',
            ],
        )

    def test_transformers_5_weight_conversion_api(self):
        class FakeOperation:
            def __init__(self):
                self.received = None

            def convert(self, state, **kwargs):
                self.received = (state, kwargs)
                return {
                    'weight': torch.nn.Parameter(
                        torch.tensor([[7.0]]),
                        requires_grad=False,
                    ),
                }

        operation = FakeOperation()
        quantizer = SimpleNamespace(
            get_weight_conversions=lambda: [
                SimpleNamespace(target_patterns=('weight',), operations=(operation,)),
            ],
        )

        model = AirLLMBaseModel.__new__(AirLLMBaseModel)
        model.hf_quantizer = quantizer
        model.running_device = 'cpu'
        model.model = torch.nn.Module()
        model.model.linear = torch.nn.Linear(1, 1, bias=False)
        model.config = SimpleNamespace()

        model._create_quantized_param(
            'linear.weight',
            {
                'linear.weight': torch.tensor([[1]], dtype=torch.uint8),
                'linear.weight.absmax': torch.tensor([2.0]),
            },
        )

        state, kwargs = operation.received
        self.assertEqual(set(state), {'weight', 'weight.absmax'})
        self.assertEqual(kwargs['full_layer_name'], 'linear.weight')
        self.assertIs(kwargs['model'], model.model)
        self.assertIs(kwargs['config'], model.config)
        torch.testing.assert_close(model.model.linear.weight, torch.tensor([[7.0]]))


if __name__ == '__main__':
    unittest.main()
