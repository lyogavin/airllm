import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

import airllm.utils as utils


class TestLocalModelDetection(unittest.TestCase):
    """
    find_or_create_local_splitted_path must recognize a local checkpoint (so it splits it in place)
    instead of falling through to snapshot_download(), which treats the local path as a repo id and
    fails. This must hold for single-file checkpoints, not just sharded (index.json) ones.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = self.tmpdir.name
        self._orig_split = utils.split_and_save_layers
        self._orig_download = utils.huggingface_hub.snapshot_download
        self.calls = {'split': False, 'download': False}

        def fake_split(*args, **kwargs):
            self.calls['split'] = True
            return 'SPLIT_PATH'

        def fake_download(*args, **kwargs):
            self.calls['download'] = True
            raise AssertionError('snapshot_download should not be called for a local checkpoint')

        utils.split_and_save_layers = fake_split
        utils.huggingface_hub.snapshot_download = fake_download

    def tearDown(self):
        utils.split_and_save_layers = self._orig_split
        utils.huggingface_hub.snapshot_download = self._orig_download
        self.tmpdir.cleanup()

    def _run(self):
        return utils.find_or_create_local_splitted_path(self.root, layer_names={
            'embed': 'model.embed_tokens', 'layer_prefix': 'model.layers',
            'norm': 'model.norm', 'lm_head': 'lm_head'})

    def test_single_file_safetensors_is_split_locally(self):
        save_file({'w': torch.zeros(2)}, os.path.join(self.root, 'model.safetensors'))
        local_path, saved = self._run()
        self.assertTrue(self.calls['split'])
        self.assertFalse(self.calls['download'])
        self.assertEqual(Path(local_path), Path(self.root))

    def test_single_file_bin_is_split_locally(self):
        torch.save({'w': torch.zeros(2)}, os.path.join(self.root, 'pytorch_model.bin'))
        self._run()
        self.assertTrue(self.calls['split'])
        self.assertFalse(self.calls['download'])

    def test_sharded_index_still_split_locally(self):
        with open(os.path.join(self.root, 'model.safetensors.index.json'), 'w') as f:
            f.write('{}')
        self._run()
        self.assertTrue(self.calls['split'])
        self.assertFalse(self.calls['download'])

    def test_directory_without_weights_falls_through_to_download(self):
        with open(os.path.join(self.root, 'config.json'), 'w') as f:
            f.write('{}')
        with self.assertRaises(AssertionError):
            self._run()
        self.assertFalse(self.calls['split'])
        self.assertTrue(self.calls['download'])


if __name__ == '__main__':
    unittest.main()
