import unittest

try:
    import mlx.core  # noqa: F401
    _HAVE_MLX = True
except Exception:
    _HAVE_MLX = False


class _Cfg:
    """Minimal stand-in for a transformers config."""
    hidden_size = 64
    intermediate_size = 128
    num_attention_heads = 8
    num_key_value_heads = 8
    num_hidden_layers = 2
    vocab_size = 100
    rms_norm_eps = 1e-5

    def __init__(self, rope_theta=None):
        if rope_theta is not None:
            self.rope_theta = rope_theta


@unittest.skipUnless(_HAVE_MLX, "MLX backend not installed (macOS / Apple silicon only)")
class TestMlxRopeTheta(unittest.TestCase):
    """
    The MLX Llama path must carry rope_theta from the model config. Otherwise sanitize_config()
    falls back to the Llama-2 default of 10000, but Llama 3.x uses 500000 — output silently
    degrades on prompts past ~1k tokens (issue #368).
    """

    def test_rope_theta_carried_from_config(self):
        from airllm.airllm_llama_mlx import get_model_args_from_config
        args = get_model_args_from_config(_Cfg(rope_theta=500000.0))
        self.assertEqual(args.rope_theta, 500000.0)

    def test_rope_theta_falls_back_when_config_lacks_it(self):
        from airllm.airllm_llama_mlx import get_model_args_from_config
        args = get_model_args_from_config(_Cfg(rope_theta=None))
        self.assertEqual(args.rope_theta, 10000)


if __name__ == '__main__':
    unittest.main()
