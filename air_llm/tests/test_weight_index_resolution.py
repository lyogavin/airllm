import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from airllm.utils import _resolve_weight_map


class TestResolveWeightMap(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_index(self, name, weight_map):
        with open(os.path.join(self.root, name), 'w') as f:
            json.dump({'metadata': {}, 'weight_map': weight_map}, f)

    def _write_safetensors(self, name):
        save_file({'w': torch.zeros(2)}, os.path.join(self.root, name))

    def _write_bin(self, name):
        torch.save({'w': torch.zeros(2)}, os.path.join(self.root, name))

    def test_prefers_safetensors_index_when_both_indexes_present(self):
        # A repo that ships both formats must not be unpickled just because the .bin index exists.
        self._write_index('pytorch_model.bin.index.json',
                          {'w': 'pytorch_model-00001-of-00001.bin'})
        self._write_index('model.safetensors.index.json',
                          {'w': 'model-00001-of-00001.safetensors'})

        index, safetensors_format = _resolve_weight_map(self.root)

        self.assertTrue(safetensors_format)
        self.assertEqual(index['w'], 'model-00001-of-00001.safetensors')

    def test_falls_back_to_bin_index_when_only_bin_present(self):
        self._write_index('pytorch_model.bin.index.json',
                          {'w': 'pytorch_model-00001-of-00001.bin'})

        index, safetensors_format = _resolve_weight_map(self.root)

        self.assertFalse(safetensors_format)
        self.assertEqual(index['w'], 'pytorch_model-00001-of-00001.bin')

    def test_prefers_single_file_safetensors_over_single_file_bin(self):
        self._write_safetensors('model.safetensors')
        self._write_bin('pytorch_model.bin')

        index, safetensors_format = _resolve_weight_map(self.root)

        self.assertTrue(safetensors_format)
        self.assertEqual(set(index.values()), {'model.safetensors'})

    def test_single_file_bin_loads_with_weights_only(self):
        # The single-file .bin branch must read keys without executing pickle payloads.
        self._write_bin('pytorch_model.bin')

        real_load = torch.load
        seen = {}

        def spy(*args, **kwargs):
            seen['weights_only'] = kwargs.get('weights_only')
            return real_load(*args, **kwargs)

        torch.load = spy
        try:
            index, safetensors_format = _resolve_weight_map(self.root)
        finally:
            torch.load = real_load

        self.assertFalse(safetensors_format)
        self.assertEqual(seen.get('weights_only'), True)
        self.assertIn('w', index)

    def test_raises_when_no_weights_present(self):
        with self.assertRaises(FileNotFoundError):
            _resolve_weight_map(self.root)


if __name__ == '__main__':
    unittest.main()
