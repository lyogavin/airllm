"""Optional native compression libraries must not break uncompressed inference."""
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch


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

    def test_8bit_dtype_metadata_survives_disk_roundtrip(self):
        import torch
        from safetensors.torch import load_file, save_file
        from airllm.utils import compress_layer_state_dict, uncompress_layer_state_dict

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                original = torch.ones(16, dtype=dtype)
                state = SimpleNamespace(
                    absmax=torch.ones(1), code=torch.ones(256), dtype=dtype,
                )
                functional = SimpleNamespace(
                    quantize_blockwise=Mock(return_value=(torch.zeros(16, dtype=torch.uint8), state)),
                    QuantState=Mock(side_effect=SimpleNamespace),
                    dequantize_blockwise=Mock(side_effect=lambda packed, quant_state: packed.to(quant_state.dtype)),
                )
                with patch.dict(sys.modules, {'bitsandbytes': SimpleNamespace(functional=functional)}), \
                        patch.object(torch.Tensor, 'cuda', lambda tensor: tensor):
                    packed = compress_layer_state_dict({'weight': original}, '8bit')
                    self.assertEqual(packed['weight.8bit.dtype'].numel(), 0)
                    self.assertEqual(packed['weight.8bit.dtype'].dtype, dtype)
                    with TemporaryDirectory() as directory:
                        path = Path(directory) / 'layer.safetensors'
                        save_file(packed, path)
                        loaded = {key: value.clone() for key, value in load_file(path).items()}
                    restored = uncompress_layer_state_dict(loaded)
                self.assertEqual(set(restored), {'weight'})
                self.assertEqual(restored['weight'].dtype, dtype)
                self.assertEqual(functional.QuantState.call_args.kwargs['dtype'], dtype)

    def test_legacy_8bit_metadata_defaults_to_float16(self):
        import torch
        from airllm.utils import uncompress_layer_state_dict

        functional = SimpleNamespace(
            QuantState=Mock(side_effect=SimpleNamespace),
            dequantize_blockwise=Mock(return_value=torch.ones(16, dtype=torch.float16)),
        )
        legacy = {
            'weight': torch.zeros(16, dtype=torch.uint8),
            'weight.8bit.absmax': torch.ones(1),
            'weight.8bit.code': torch.ones(256),
        }
        with patch.dict(sys.modules, {'bitsandbytes': SimpleNamespace(functional=functional)}), \
                patch.object(torch.Tensor, 'cuda', lambda tensor: tensor):
            restored = uncompress_layer_state_dict(legacy)
        self.assertEqual(set(restored), {'weight'})
        self.assertEqual(functional.QuantState.call_args.kwargs['dtype'], torch.float16)


if __name__ == '__main__':
    unittest.main()
