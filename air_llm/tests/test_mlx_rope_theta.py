"""Offline test for #368: MLX path must respect rope_theta from HF config.

No GPU, no network, no mlx needed (mocked before import).
Run from air_llm/: python -m pytest tests/test_mlx_rope_theta.py -q
"""

import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from types import SimpleNamespace


def _mod(name, is_pkg=False):
    m = types.ModuleType(name)
    m.__spec__ = ModuleSpec(name, loader=None, is_package=is_pkg)
    if is_pkg:
        m.__path__ = []
    return m


def _ensure_mlx_mocked():
    if "mlx.core" in sys.modules and getattr(
        sys.modules["mlx"], "__spec__", None
    ) is not None:
        return
    mlx = _mod("mlx", is_pkg=True)
    core = _mod("mlx.core")

    class _Arr:
        pass

    core.array = _Arr
    core.ones = lambda *a, **k: None
    mlx.core = core
    nn_mod = _mod("mlx.nn")

    class _M:
        def __init__(self, *a, **k):
            pass

    nn_mod.Module = _M
    nn_mod.Linear = _M
    nn_mod.RoPE = _M
    mlx.nn = nn_mod
    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = core
    sys.modules["mlx.nn"] = nn_mod


_ensure_mlx_mocked()

from ..airllm.airllm_llama_mlx import get_model_args_from_config


def _base_config(**overrides):
    cfg = dict(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=32,
        vocab_size=128256,
        rms_norm_eps=1e-5,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class TestMlxRopeTheta(unittest.TestCase):
    def test_llama3_theta_is_respected(self):
        """Llama 3.x rope_theta=500000 must survive (issue #368)."""
        args = get_model_args_from_config(_base_config(rope_theta=500000))
        self.assertEqual(args.rope_theta, 500000)

    def test_missing_theta_falls_back_to_10000(self):
        """Llama 2 style configs without theta keep legacy default."""
        args = get_model_args_from_config(_base_config())
        self.assertEqual(args.rope_theta, 10000)

    def test_explicit_10000_stays_10000(self):
        args = get_model_args_from_config(_base_config(rope_theta=10000))
        self.assertEqual(args.rope_theta, 10000)


if __name__ == "__main__":
    unittest.main()
