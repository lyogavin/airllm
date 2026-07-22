import importlib.util
import unittest

import torch
import torch.nn as nn

from ..airllm.airllm_packed import AirLLMPackedLinear, replace_with_airllm_packed_linears


class _PackedSkeleton(nn.Linear):
    def __init__(self):
        super().__init__(32, 32, bias=False, device="meta")
        del self._parameters["weight"]
        self.weight_packed = nn.Parameter(
            torch.empty(32, 4, dtype=torch.int32, device="meta"), requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(32, 1, dtype=torch.float32, device="meta"), requires_grad=False
        )
        self.weight_shape = nn.Parameter(
            torch.empty(2, dtype=torch.int64, device="meta"), requires_grad=False
        )
        self.quantization_scheme = object()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _PackedSkeleton()
        self.ct_decompress_hook = self.register_forward_pre_hook(lambda model, args: None)


class TestPackedLinear(unittest.TestCase):
    def test_replacement_preserves_checkpoint_state_names(self):
        model = _Model()
        count = replace_with_airllm_packed_linears(model)

        self.assertEqual(count, 1)
        self.assertIsInstance(model.proj, AirLLMPackedLinear)
        self.assertFalse(hasattr(model, "ct_decompress_hook"))
        self.assertEqual(
            set(model.proj.state_dict()),
            {"weight_packed", "weight_scale", "weight_shape"},
        )

    @unittest.skipUnless(importlib.util.find_spec("compressed_tensors"), "requires compressed-tensors")
    def test_packed_projection_matches_explicit_decompression(self):
        from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
        from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

        scheme = QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="group",
                group_size=32,
            ),
        )
        weight = torch.randint(-7, 8, (32, 32), dtype=torch.int8).float()
        scale = torch.ones(32, 1)
        compressed = PackedQuantizationCompressor.compress(
            {"weight": weight, "weight_scale": scale}, scheme
        )

        original = _PackedSkeleton()
        original.quantization_scheme = scheme
        module = AirLLMPackedLinear(original).to_empty(device="cpu")
        module.weight_packed.data.copy_(compressed["weight_packed"])
        module.weight_scale.data.copy_(compressed["weight_scale"])
        module.weight_shape.data.copy_(compressed["weight_shape"])
        inputs = torch.randn(2, 32)

        explicit = PackedQuantizationCompressor.decompress(compressed, scheme)["weight"]
        expected = torch.nn.functional.linear(inputs, explicit)
        torch.testing.assert_close(module(inputs), expected)

    @unittest.skipUnless(
        importlib.util.find_spec("compressed_tensors") and torch.cuda.is_available(),
        "requires compressed-tensors and a CUDA/ROCm device",
    )
    def test_packed_projection_runs_on_gpu(self):
        from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
        from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

        scheme = QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="group",
                group_size=32,
            ),
        )
        compressed = PackedQuantizationCompressor.compress(
            {
                "weight": torch.randint(-7, 8, (32, 32), dtype=torch.int8).float(),
                "weight_scale": torch.ones(32, 1),
            },
            scheme,
        )
        original = _PackedSkeleton()
        original.quantization_scheme = scheme
        module = AirLLMPackedLinear(original).to_empty(device="cuda")
        for name, value in compressed.items():
            getattr(module, name).data.copy_(value.to("cuda"))
        inputs = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)

        explicit = PackedQuantizationCompressor.decompress(
            {name: value.to("cuda") for name, value in compressed.items()}, scheme
        )["weight"].to(torch.bfloat16)
        expected = torch.nn.functional.linear(inputs, explicit)
        torch.testing.assert_close(module(inputs), expected)
