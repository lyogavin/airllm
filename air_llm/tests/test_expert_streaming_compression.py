import tempfile
import unittest

import torch
import torch.nn as nn
from safetensors.torch import save_file

from airllm.airllm_base import AirLLMBaseModel
from airllm.utils import load_layer_subset


class _Expert(nn.Module):
    pass


class _MoE(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.experts = nn.ModuleList([_Expert() for _ in range(n)])


class _Layer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.block_sparse_moe = _MoE(n)


class TestExpertStreamingCompressionGuard(unittest.TestCase):
    """
    Per-expert streaming reads tensors out of a shard with load_layer_subset(), which -- unlike
    load_layer() -- never runs uncompress_layer_state_dict(). So it must not engage when the shards
    are compressed, or still-quantized weights reach move_layer_to_device().
    """

    LAYER = 'language_model.model.layers.0'

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_shard(self, compressed):
        """Write a layer shard holding two experts, optionally in airllm's compressed layout."""
        sd = {}
        for e in (0, 1):
            base = f'{self.LAYER}.block_sparse_moe.experts.{e}.w1.weight'
            if compressed:
                sd[base] = torch.zeros(64, dtype=torch.uint8)
                sd[base + '.4bit.absmax'] = torch.zeros(4)
            else:
                sd[base] = torch.zeros(8, 8)
        sd[f'{self.LAYER}.self_attn.q_proj.weight'] = torch.zeros(8, 8)
        save_file(sd, f'{self.tmpdir.name}/{self.LAYER}.safetensors')

    def _make_obj(self, compression):
        # Skip the heavy __init__; _setup_expert_streaming only needs these attributes.
        obj = AirLLMBaseModel.__new__(AirLLMBaseModel)
        obj.compression = compression
        obj.layer_names_dict = {'layer_prefix': 'language_model.model.layers',
                                'expert_prefix': 'block_sparse_moe.experts'}
        obj.layer_names = ['embed', self.LAYER, 'norm']
        obj._streamed_indices = [1]
        obj.checkpoint_path = self.tmpdir.name
        obj.layers = [None, _Layer(2), None]
        return obj

    def test_expert_streaming_engages_without_compression(self):
        # Control: proves the fixture really does enable expert streaming, so the guard test below
        # is meaningful rather than passing for an unrelated reason.
        self._write_shard(compressed=False)
        obj = self._make_obj(None)
        obj._setup_expert_streaming()
        self.assertTrue(obj._expert_streaming)
        self.assertEqual(sorted(obj._expert_keys[1]), [0, 1])

    def test_expert_streaming_disabled_when_compression_is_on(self):
        for compression in ('4bit', '8bit'):
            with self.subTest(compression=compression):
                self._write_shard(compressed=True)
                obj = self._make_obj(compression)
                obj._setup_expert_streaming()
                self.assertFalse(obj._expert_streaming,
                                 f"expert streaming must stay off with compression={compression}")
                self.assertEqual(obj._expert_keys, {})
                self.assertEqual(obj._non_expert_keys, {})

    def test_load_layer_subset_does_not_decompress(self):
        # Documents *why* the guard is needed.
        self._write_shard(compressed=True)
        base = f'{self.LAYER}.block_sparse_moe.experts.0.w1.weight'
        got = load_layer_subset(self.tmpdir.name, self.LAYER, [base, base + '.4bit.absmax'])
        self.assertEqual(got[base].dtype, torch.uint8)     # never dequantized
        self.assertIn(base + '.4bit.absmax', got)          # quant state still present


if __name__ == '__main__':
    unittest.main()
