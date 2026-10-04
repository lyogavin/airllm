"""Exercise real GPU compression, including persisted quantization metadata.

Run explicitly in a CUDA/ROCm environment with bitsandbytes installed. A missing
GPU skips this module; a broken or missing backend on a GPU is a test failure.
"""
import pytest
import torch

from airllm.compression import require_bitsandbytes
from airllm.persist.safetensor_model_persister import SafetensorModelPersister
from airllm.utils import compress_layer_state_dict, uncompress_layer_state_dict


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _reference(tensor, mode):
    backend = require_bitsandbytes()
    tensor = tensor.cuda()
    if mode == "4bit":
        packed, state = backend.functional.quantize_nf4(tensor, blocksize=64)
        return backend.functional.dequantize_nf4(packed, state)
    packed, state = backend.functional.quantize_blockwise(tensor, blocksize=2048)
    return backend.functional.dequantize_blockwise(packed, state)


def _disk_roundtrip(state, path):
    persister = SafetensorModelPersister()
    persister.persist_model(state, "layer.", path)
    assert persister.model_persist_exist("layer.", path)
    loaded = persister.load_model("layer", path)
    assert all(tensor.device.type == "cpu" for tensor in loaded.values())
    return uncompress_layer_state_dict(loaded)


@pytest.mark.parametrize("mode", ["4bit", "8bit"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compression_matches_backend_after_disk_roundtrip(tmp_path, mode, dtype):
    generator = torch.Generator().manual_seed(318)
    original = {
        "layer.weight": torch.randn(64, 128, generator=generator).to(dtype),
        "layer.bias": torch.randn(128, generator=generator).to(dtype),
        "layer.norm": torch.ones(128, dtype=dtype),
    }
    expected = {key: _reference(value, mode) for key, value in original.items()}
    packed = compress_layer_state_dict(original, mode)
    assert all(packed[key].dtype == torch.uint8 for key in original)
    assert sum(value.numel() * value.element_size() for value in packed.values()) < sum(
        value.numel() * value.element_size() for value in original.values()
    )
    restored = _disk_roundtrip(packed, tmp_path)
    torch.cuda.synchronize()
    assert restored.keys() == original.keys()
    for key, value in restored.items():
        assert value.dtype == dtype
        assert value.shape == original[key].shape
        assert value.device.type == "cuda"
        assert torch.isfinite(value).all()
        # Serialization must introduce no additional quantization error.
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        relative_rmse = (value.float().cpu() - original[key].float()).square().mean().sqrt()
        relative_rmse /= original[key].float().square().mean().sqrt()
        assert relative_rmse.item() < (0.13 if mode == "4bit" else 0.025)


@pytest.mark.parametrize("mode", ["4bit", "8bit"])
def test_bfloat16_range_survives_compression(tmp_path, mode):
    # BF16 values above FP16's maximum must not overflow during decompression.
    original = torch.linspace(-131072, 131072, 4096).to(torch.bfloat16)
    expected = _reference(original, mode)
    restored = _disk_roundtrip(compress_layer_state_dict({"weight": original}, mode), tmp_path)["weight"]
    torch.cuda.synchronize()
    assert torch.isfinite(restored).all()
    assert restored.dtype == torch.bfloat16
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def test_legacy_8bit_shard_remains_readable(tmp_path):
    # Older AirLLM shards contain only these three tensors and imply FP16.
    backend = require_bitsandbytes()
    original = torch.linspace(-2, 2, 4096, dtype=torch.float16, device="cuda")
    packed, state = backend.functional.quantize_blockwise(original, blocksize=2048)
    legacy = {
        "weight": packed,
        "weight.8bit.absmax": state.absmax.clone(),
        "weight.8bit.code": state.code.clone(),
    }
    restored = _disk_roundtrip(legacy, tmp_path)["weight"]
    torch.cuda.synchronize()
    assert restored.dtype == torch.float16
    torch.testing.assert_close(
        restored, backend.functional.dequantize_blockwise(packed, state), rtol=0, atol=0
    )
