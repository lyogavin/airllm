"""Opt-in GPU compression tests, including persisted quantization metadata.

Run with AIRLLM_GPU_TESTS=1 in a CUDA/ROCm environment with bitsandbytes
installed. A missing GPU skips the suite; a broken or missing backend on a GPU
is a test failure. Default unittest discovery imports no GPU dependencies here.
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(os.environ.get("AIRLLM_GPU_TESTS") == "1", "set AIRLLM_GPU_TESTS=1")
class CompressionBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("requires a CUDA or ROCm GPU")
        from airllm.persist.safetensor_model_persister import SafetensorModelPersister
        from airllm.utils import (
            compress_layer_state_dict,
            require_bitsandbytes,
            uncompress_layer_state_dict,
        )

        cls.torch = torch
        cls.persister_class = SafetensorModelPersister
        cls.compress = staticmethod(compress_layer_state_dict)
        cls.uncompress = staticmethod(uncompress_layer_state_dict)
        cls.backend = require_bitsandbytes()

    def _reference(self, tensor, mode):
        tensor = tensor.cuda()
        if mode == "4bit":
            packed, state = self.backend.functional.quantize_nf4(tensor, blocksize=64)
            return self.backend.functional.dequantize_nf4(packed, state)
        packed, state = self.backend.functional.quantize_blockwise(tensor, blocksize=2048)
        return self.backend.functional.dequantize_blockwise(packed, state)

    def _disk_roundtrip(self, state):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            persister = self.persister_class()
            persister.persist_model(state, "layer.", path)
            self.assertTrue(persister.model_persist_exist("layer.", path))
            loaded = {
                key: value.clone() for key, value in persister.load_model("layer", path).items()
            }
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in loaded.values()))
        return self.uncompress(loaded)

    def test_compression_matches_backend_after_disk_roundtrip(self):
        torch = self.torch
        for dtype in (torch.float16, torch.bfloat16):
            for mode in ("4bit", "8bit"):
                with self.subTest(dtype=dtype, mode=mode):
                    generator = torch.Generator().manual_seed(318)
                    original = {
                        "layer.weight": torch.randn(64, 128, generator=generator).to(dtype),
                        "layer.bias": torch.randn(128, generator=generator).to(dtype),
                        "layer.norm": torch.ones(128, dtype=dtype),
                    }
                    expected = {key: self._reference(value, mode) for key, value in original.items()}
                    packed = self.compress(original, mode)
                    self.assertTrue(all(packed[key].dtype == torch.uint8 for key in original))
                    self.assertLess(
                        sum(value.numel() * value.element_size() for value in packed.values()),
                        sum(value.numel() * value.element_size() for value in original.values()),
                    )
                    restored = self._disk_roundtrip(packed)
                    torch.cuda.synchronize()
                    self.assertEqual(restored.keys(), original.keys())
                    for key, value in restored.items():
                        self.assertEqual(value.dtype, dtype)
                        self.assertEqual(value.shape, original[key].shape)
                        self.assertEqual(value.device.type, "cuda")
                        self.assertTrue(torch.isfinite(value).all())
                        # Serialization must introduce no additional quantization error.
                        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                        relative_rmse = (value.float().cpu() - original[key].float()).square().mean().sqrt()
                        relative_rmse /= original[key].float().square().mean().sqrt()
                        self.assertLess(relative_rmse.item(), 0.13 if mode == "4bit" else 0.025)

    def test_bfloat16_range_survives_compression(self):
        torch = self.torch
        for mode in ("4bit", "8bit"):
            with self.subTest(mode=mode):
                # BF16 values above FP16's maximum must not overflow during decompression.
                original = torch.linspace(-131072, 131072, 4096).to(torch.bfloat16)
                expected = self._reference(original, mode)
                restored = self._disk_roundtrip(self.compress({"weight": original}, mode))["weight"]
                torch.cuda.synchronize()
                self.assertTrue(torch.isfinite(restored).all())
                self.assertEqual(restored.dtype, torch.bfloat16)
                torch.testing.assert_close(restored, expected, rtol=0, atol=0)

    def test_legacy_8bit_shard_remains_readable(self):
        torch = self.torch
        # Older AirLLM shards contain only these three tensors and imply FP16.
        original = torch.linspace(-2, 2, 4096, dtype=torch.float16, device="cuda")
        packed, state = self.backend.functional.quantize_blockwise(original, blocksize=2048)
        legacy = {
            "weight": packed,
            "weight.8bit.absmax": state.absmax.clone(),
            "weight.8bit.code": state.code.clone(),
        }
        restored = self._disk_roundtrip(legacy)["weight"]
        torch.cuda.synchronize()
        self.assertEqual(restored.dtype, torch.float16)
        torch.testing.assert_close(
            restored, self.backend.functional.dequantize_blockwise(packed, state), rtol=0, atol=0,
        )


if __name__ == "__main__":
    unittest.main()
