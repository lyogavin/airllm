"""Regression test for the AirLLMLlamaMlx input-coercion fix.

Before this fix, ``AirLLMLlamaMlx.generate(input_ids)`` raised

    ValueError: Cannot index mlx array using the given type.

whenever ``input_ids`` was a ``torch.LongTensor`` (the natural output
of ``tokenizer(text, return_tensors="pt").input_ids``). The downstream
``mlx.nn.Embedding`` call cannot index an mlx array using a torch
tensor.

The fix introduces ``_coerce_to_mlx_array`` and calls it at the
``generate()`` entry point so callers don't need to know about
``mlx.core``. This test exercises the helper on every input shape the
generate boundary realistically receives.
"""

import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from ..airllm.airllm_llama_mlx import _coerce_to_mlx_array


class TestCoerceToMlxArray(unittest.TestCase):
    """Direct tests on the helper."""

    def test_mlx_array_is_passthrough(self):
        x = mx.array([[1, 2, 3]])
        out = _coerce_to_mlx_array(x)
        self.assertIs(out, x)

    def test_torch_long_tensor_is_converted(self):
        # The exact shape that triggers the original bug:
        # tokenizer(..., return_tensors="pt").input_ids is a 2D
        # torch.LongTensor of shape (batch, seq_len).
        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        out = _coerce_to_mlx_array(x)
        self.assertIsInstance(out, mx.array)
        self.assertEqual(out.shape, (1, 4))

    def test_torch_grad_tensor_is_handled(self):
        # input_ids don't normally have grad, but the helper should
        # not crash if a caller passes a grad-attached tensor.
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = _coerce_to_mlx_array(x)
        self.assertIsInstance(out, mx.array)

    def test_numpy_array_is_converted(self):
        x = np.array([[1, 2, 3]], dtype=np.int64)
        out = _coerce_to_mlx_array(x)
        self.assertIsInstance(out, mx.array)
        self.assertEqual(out.shape, (1, 3))

    def test_python_list_is_converted(self):
        x = [[1, 2, 3]]
        out = _coerce_to_mlx_array(x)
        self.assertIsInstance(out, mx.array)
        self.assertEqual(out.shape, (1, 3))


class TestEmbeddingIntegration(unittest.TestCase):
    """End-to-end regression: the original failure was in
    ``self.tok_embeddings(x)`` inside model_generate. Verify the
    coerced input flows through an mlx.nn.Embedding without raising.
    """

    def test_torch_tensor_through_embedding_after_coerce(self):
        emb = nn.Embedding(100, 4)

        # Confirm the original failure mode is real (sanity check —
        # if this ever stops failing, something changed in mlx).
        raw_torch = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with self.assertRaises((ValueError, TypeError)):
            emb(raw_torch)

        # The fix: coerce, then use.
        coerced = _coerce_to_mlx_array(raw_torch)
        out = emb(coerced)
        self.assertEqual(out.shape, (1, 3, 4))


if __name__ == "__main__":
    unittest.main()
