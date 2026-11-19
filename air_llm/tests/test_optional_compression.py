"""Optional native compression libraries must not break uncompressed inference."""
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch


@unittest.skipIf(sys.platform == 'darwin', 'tests the PyTorch path; macOS uses MLX')
class OptionalCompressionTests(unittest.TestCase):
    def test_uncompressed_path_does_not_import_bitsandbytes(self):
        script_template = textwrap.dedent('''
        import importlib.abc
        import sys
        import torch
        import transformers

        attempts = []
        class BrokenBackend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'bitsandbytes' or fullname.startswith('bitsandbytes.'):
                    attempts.append(fullname)
                    raise ERROR('simulated incompatible native backend')

        sys.meta_path.insert(0, BrokenBackend())
        from airllm import AirLLMBaseModel
        from airllm.utils import compress_layer_state_dict, uncompress_layer_state_dict
        state = {'weight': torch.ones(2, 2)}
        assert compress_layer_state_dict(state, None) is state
        assert uncompress_layer_state_dict(state) is state
        assert 'bitsandbytes' not in sys.modules
        assert attempts == []

        # Fail before downloading or splitting a model; retain the underlying cause.
        for operation in (
            lambda: AirLLMBaseModel('unused-model', compression='4bit'),
            lambda: compress_layer_state_dict(state, '8bit'),
            lambda: uncompress_layer_state_dict({'weight.4bit.absmax': torch.ones(1)}),
            lambda: uncompress_layer_state_dict({'weight.8bit.absmax': torch.ones(1)}),
        ):
            try:
                operation()
            except ImportError as exc:
                assert 'compatible' in str(exc)
                assert isinstance(exc.__cause__, ERROR)
            else:
                raise AssertionError('compression must require a working backend')
        ''')
        for error in ('ImportError', 'OSError', 'RuntimeError'):
            with self.subTest(error=error):
                script = script_template.replace('ERROR', error)
                result = subprocess.run(
                    [sys.executable, '-c', script], capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_compression_backend_remains_available_when_installed(self):
        from airllm.utils import require_bitsandbytes

        backend = SimpleNamespace()
        with patch.dict(sys.modules, {'bitsandbytes': backend}):
            self.assertIs(require_bitsandbytes(), backend)


if __name__ == '__main__':
    unittest.main()
