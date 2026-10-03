import sys
import unittest


@unittest.skipUnless(sys.platform == "darwin", "MLX is only available on macOS")
class TestMlxInputTypes(unittest.TestCase):
    def test_torch_tensor_is_converted_for_embedding_lookup(self):
        import mlx.core as mx
        import torch

        from ..airllm.airllm_llama_mlx import _coerce_to_mlx_array

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        converted = _coerce_to_mlx_array(input_ids)

        self.assertIsInstance(converted, mx.array)
        self.assertEqual(converted.shape, (1, 3))
        self.assertEqual(converted.tolist(), [[1, 2, 3]])

    def test_existing_mlx_array_is_reused(self):
        import mlx.core as mx

        from ..airllm.airllm_llama_mlx import _coerce_to_mlx_array

        input_ids = mx.array([[1, 2, 3]])

        self.assertIs(_coerce_to_mlx_array(input_ids), input_ids)

    def test_python_sequence_is_converted(self):
        import mlx.core as mx

        from ..airllm.airllm_llama_mlx import _coerce_to_mlx_array

        converted = _coerce_to_mlx_array([[1, 2, 3]])

        self.assertIsInstance(converted, mx.array)
        self.assertEqual(converted.tolist(), [[1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
