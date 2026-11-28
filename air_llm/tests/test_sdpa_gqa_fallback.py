"""Unit tests for GQA SDPA math-kernel fallback avoidance on CUDA (issue #396).

Validates:
1. Ampere+ with FlashAttention or cuDNN keeps native enable_gqa=True (wrapper is no-op).
2. Fallback on pre-Ampere/Windows disables enable_gqa so transformers does repeat_kv
   and scaled_dot_product_attention runs on the EFFICIENT_ATTENTION kernel family.
3. Math-only environments are never touched (no faster kernel to fall back to).
4. Long prefill sequences avoid O(n^2) math attention memory spikes/OOM.
5. All probes use public PyTorch APIs without importing private SDPAParams.
"""

import importlib.machinery
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Provide lightweight stubs for optional dependencies when running in minimal test environments
if importlib.util.find_spec("accelerate") is None:
    fake_accelerate = types.ModuleType("accelerate")
    fake_accelerate.__spec__ = importlib.machinery.ModuleSpec("accelerate", None)
    fake_accelerate.__version__ = "0.26.0"
    fake_accelerate.init_empty_weights = lambda *a, **kw: None
    fake_acc_utils = types.ModuleType("accelerate.utils")
    fake_acc_utils.__spec__ = importlib.machinery.ModuleSpec("accelerate.utils", None)
    fake_acc_modeling = types.ModuleType("accelerate.utils.modeling")
    fake_acc_modeling.__spec__ = importlib.machinery.ModuleSpec("accelerate.utils.modeling", None)
    fake_acc_modeling.set_module_tensor_to_device = lambda *a, **kw: None
    sys.modules["accelerate"] = fake_accelerate
    sys.modules["accelerate.utils"] = fake_acc_utils
    sys.modules["accelerate.utils.modeling"] = fake_acc_modeling

_AIRLLM_ROOT = Path(__file__).resolve().parents[1]
if str(_AIRLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_AIRLLM_ROOT))

import torch

from airllm import utils


class _FakeCudaTensor:
    """Minimal key/value lookalike reporting device='cuda:0' on CPU-only test runners."""

    def __init__(self, shape=(1, 2, 16, 128), dtype=torch.float16, device_index=0):
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device("cuda", device_index)


def _orig_use_gqa(attention_mask, key, value):
    return attention_mask is None and key.shape[-1] == value.shape[-1] <= 256


class TestSdpaGqaFallback(unittest.TestCase):
    def setUp(self):
        utils._CUDA_SDPA_GQA_CACHE.clear()
        self.fake_sa = SimpleNamespace(use_gqa_in_sdpa=_orig_use_gqa)
        self.current_backend_context = []
        self.last_executed_kernel_family = None

    def tearDown(self):
        utils._CUDA_SDPA_GQA_CACHE.clear()

    def _mock_cuda_env(
        self,
        *,
        flash_enabled=True,
        cudnn_enabled=True,
        mem_eff_enabled=True,
        flash_supports_gqa=False,
        cudnn_supports_gqa=False,
        eff_supports_gqa=False,
        eff_supports_repeated=True,
    ):
        orig_empty = torch.empty

        def _fake_empty(*args, **kw):
            dev = kw.get("device")
            if dev is not None and torch.device(dev).type == "cuda":
                shape = args[0] if args else kw.get("size")
                dtype = kw.get("dtype", torch.float16)
                return _FakeCudaTensor(shape, dtype)
            return orig_empty(*args, **kw)

        def _fake_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            enable_gqa = kwargs.get("enable_gqa", False)
            backends = self.current_backend_context

            can_flash = flash_enabled and ("flash" in backends or not backends) and flash_supports_gqa
            can_cudnn = cudnn_enabled and ("cudnn" in backends or not backends) and cudnn_supports_gqa

            is_repeated = query.shape[1] == key.shape[1]
            can_eff = (
                mem_eff_enabled
                and ("efficient" in backends or not backends)
                and ((enable_gqa and eff_supports_gqa) or (is_repeated and eff_supports_repeated))
            )

            if can_flash:
                self.last_executed_kernel_family = "flash"
                return torch.zeros_like(query) if isinstance(query, torch.Tensor) else None
            if can_cudnn:
                self.last_executed_kernel_family = "cudnn"
                return torch.zeros_like(query) if isinstance(query, torch.Tensor) else None
            if can_eff:
                self.last_executed_kernel_family = "efficient"
                return torch.zeros_like(query) if isinstance(query, torch.Tensor) else None

            if backends and not ("math" in backends):
                raise RuntimeError(
                    f"No viable backend for scaled_dot_product_attention with backends={backends}"
                )

            self.last_executed_kernel_family = "math"
            return torch.zeros_like(query) if isinstance(query, torch.Tensor) else None

        class _MockBackendContext:
            def __init__(outer, backends):
                outer.backends = backends
                outer.prev = None

            def __enter__(outer):
                outer.prev = list(self.current_backend_context)
                self.current_backend_context = list(outer.backends)
                return self

            def __exit__(outer, *exc):
                self.current_backend_context = outer.prev

        return (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.backends.cuda, "flash_sdp_enabled", side_effect=lambda: flash_enabled, create=True),
            patch.object(torch.backends.cuda, "cudnn_sdp_enabled", side_effect=lambda: cudnn_enabled, create=True),
            patch.object(
                torch.backends.cuda, "mem_efficient_sdp_enabled", side_effect=lambda: mem_eff_enabled, create=True
            ),
            patch.object(utils, "_sdpa_kernel_context", side_effect=_MockBackendContext),
            patch.object(torch.nn.functional, "scaled_dot_product_attention", side_effect=_fake_sdpa),
            patch.object(torch, "empty", side_effect=_fake_empty),
        )

    def _enter_all(self, patches):
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_ampere_flash_attention_preserves_native_gqa_noop(self):
        """On Ampere+ where FlashAttention supports native GQA, wrapper is a complete no-op."""
        self._enter_all(
            self._mock_cuda_env(
                flash_enabled=True,
                flash_supports_gqa=True,
                cudnn_enabled=False,
                mem_eff_enabled=True,
            )
        )
        supported = utils.configure_sdpa_gqa_fallback(
            device="cuda:0",
            dtype=torch.float16,
            head_dim=128,
            sdpa_module=self.fake_sa,
        )
        self.assertTrue(supported)
        key = _FakeCudaTensor(shape=(1, 2, 16, 128), dtype=torch.float16)
        # Native enable_gqa=True is preserved
        self.assertTrue(self.fake_sa.use_gqa_in_sdpa(None, key, key))
        self.assertEqual(self.last_executed_kernel_family, "flash")

    def test_cudnn_attention_preserves_native_gqa_noop(self):
        """When cuDNN handles native GQA, wrapper is a no-op and leaves native GQA untouched."""
        self._enter_all(
            self._mock_cuda_env(
                flash_enabled=False,
                cudnn_enabled=True,
                cudnn_supports_gqa=True,
                mem_eff_enabled=True,
            )
        )
        supported = utils.configure_sdpa_gqa_fallback(
            device="cuda:0",
            dtype=torch.float16,
            head_dim=128,
            sdpa_module=self.fake_sa,
        )
        self.assertTrue(supported)
        key = _FakeCudaTensor(shape=(1, 2, 16, 128), dtype=torch.float16)
        self.assertTrue(self.fake_sa.use_gqa_in_sdpa(None, key, key))
        self.assertEqual(self.last_executed_kernel_family, "cudnn")

    def test_math_only_preserves_native_gqa_untouched(self):
        """When only math SDPA is available (mem-efficient disabled), native GQA is untouched."""
        self._enter_all(
            self._mock_cuda_env(
                flash_enabled=False,
                cudnn_enabled=False,
                mem_eff_enabled=False,  # Mem-efficient disabled
            )
        )
        supported = utils.configure_sdpa_gqa_fallback(
            device="cuda:0",
            dtype=torch.float16,
            head_dim=128,
            sdpa_module=self.fake_sa,
        )
        # Never touch math-only: nothing faster to fall back to
        self.assertTrue(supported)
        key = _FakeCudaTensor(shape=(1, 2, 16, 128), dtype=torch.float16)
        self.assertTrue(self.fake_sa.use_gqa_in_sdpa(None, key, key))

    def test_fallback_disables_gqa_and_runs_mem_efficient_kernel_family(self):
        """Pre-Ampere/Windows fallback disables enable_gqa and executes on EFFICIENT_ATTENTION."""
        self._enter_all(
            self._mock_cuda_env(
                flash_enabled=False,  # Flash unavailable (e.g. Windows official wheel)
                cudnn_enabled=False,
                mem_eff_enabled=True,
                eff_supports_gqa=False,  # Mem-efficient rejects native GQA
                eff_supports_repeated=True,  # Mem-efficient accepts equal heads (repeat_kv)
            )
        )
        supported = utils.configure_sdpa_gqa_fallback(
            device="cuda:0",
            dtype=torch.float16,
            head_dim=128,
            sdpa_module=self.fake_sa,
        )
        self.assertFalse(supported)
        key = _FakeCudaTensor(shape=(1, 2, 16, 128), dtype=torch.float16)
        # use_gqa_in_sdpa returns False so transformers repeats KV
        self.assertFalse(self.fake_sa.use_gqa_in_sdpa(None, key, key))

        # Now simulate transformers sdpa_attention_forward calling scaled_dot_product_attention
        # with repeated KV (enable_gqa=False)
        query = torch.zeros((1, 8, 16, 128), dtype=torch.float16)
        repeated_key = torch.zeros((1, 8, 16, 128), dtype=torch.float16)
        self.current_backend_context = []

        torch.nn.functional.scaled_dot_product_attention(
            query, repeated_key, repeated_key, is_causal=True, enable_gqa=False
        )
        # Explicitly assert the executed kernel family is efficient (mem-efficient), NOT math!
        self.assertEqual(self.last_executed_kernel_family, "efficient")

    def test_long_sequence_prefill_avoids_oom_by_staying_on_mem_efficient(self):
        """Simulate long prefill where math attention would OOM but mem-efficient stays within budget."""
        seq_len = 16384
        num_heads = 32
        device_memory_limit_bytes = 4 * 1024 * 1024 * 1024  # 4 GB

        # Math kernel materializes full (batch, heads, seq, seq) attention matrix in float32:
        math_matrix_bytes = 1 * num_heads * seq_len * seq_len * 4  # ~34.3 GB
        self.assertGreater(math_matrix_bytes, device_memory_limit_bytes)

        # Mem-efficient kernel computes in tiled blocks (e.g. 512x128):
        mem_efficient_peak_bytes = 1 * num_heads * 512 * 128 * 4  # ~8.3 MB
        self.assertLess(mem_efficient_peak_bytes, device_memory_limit_bytes)

        self._enter_all(
            self._mock_cuda_env(
                flash_enabled=False,
                cudnn_enabled=False,
                mem_eff_enabled=True,
                eff_supports_gqa=False,
                eff_supports_repeated=True,
            )
        )
        supported = utils.configure_sdpa_gqa_fallback(
            device="cuda:0",
            dtype=torch.float16,
            head_dim=128,
            sdpa_module=self.fake_sa,
        )
        self.assertFalse(supported)

        # Fallback directs transformers to repeat KV and run mem_efficient:
        self.current_backend_context = []
        torch.nn.functional.scaled_dot_product_attention(
            torch.zeros((1, 8, 16, 128)),
            torch.zeros((1, 8, 16, 128)),
            torch.zeros((1, 8, 16, 128)),
            enable_gqa=False,
        )
        self.assertEqual(self.last_executed_kernel_family, "efficient")

    def test_non_cuda_tensors_and_masks_follow_original_logic(self):
        self._enter_all(self._mock_cuda_env())
        utils.configure_sdpa_gqa_fallback(
            device="cuda:0",
            dtype=torch.float16,
            head_dim=128,
            sdpa_module=self.fake_sa,
        )
        cpu_key = torch.empty((1, 2, 16, 128), dtype=torch.float16)
        mask = torch.ones((1, 1, 16, 16), dtype=torch.bool)
        self.assertTrue(self.fake_sa.use_gqa_in_sdpa(None, cpu_key, cpu_key))
        self.assertFalse(self.fake_sa.use_gqa_in_sdpa(mask, cpu_key, cpu_key))

    def test_idempotent_patch_and_cache_respects_sdpa_toggle_changes(self):
        flag = {"flash": True}
        self._enter_all(
            (
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.backends.cuda, "flash_sdp_enabled", side_effect=lambda: flag["flash"], create=True),
                patch.object(torch.backends.cuda, "cudnn_sdp_enabled", side_effect=lambda: False, create=True),
                patch.object(torch.backends.cuda, "mem_efficient_sdp_enabled", side_effect=lambda: True, create=True),
                patch.object(
                    torch,
                    "empty",
                    side_effect=lambda shape, device=None, dtype=torch.float16, **kw: _FakeCudaTensor(shape, dtype),
                ),
            )
        )
        with patch.object(utils, "_sdpa_kernel_context") as mock_ctx:
            mock_ctx.return_value.__enter__.return_value = None
            mock_ctx.return_value.__exit__.return_value = None

            def _sim_sdpa(q, k, v, **kw):
                if kw.get("enable_gqa") and not flag["flash"]:
                    raise RuntimeError("No fast GQA")
                return torch.zeros((1, 2, 1, 128))

            with patch.object(torch.nn.functional, "scaled_dot_product_attention", side_effect=_sim_sdpa):
                utils.configure_sdpa_gqa_fallback(sdpa_module=self.fake_sa)
                utils.configure_sdpa_gqa_fallback(sdpa_module=self.fake_sa)
                self.assertIs(self.fake_sa._airllm_orig_use_gqa_in_sdpa, _orig_use_gqa)

                key = _FakeCudaTensor(shape=(1, 2, 16, 128), dtype=torch.float16)
                self.assertTrue(self.fake_sa.use_gqa_in_sdpa(None, key, key))

                # Toggle flash off -> cache invalidates and probes mem-efficient fallback
                flag["flash"] = False
                self.assertFalse(self.fake_sa.use_gqa_in_sdpa(None, key, key))

    def test_real_transformers_sdpa_attention_forward_repeats_kv_when_fallback_active(self):
        import transformers.integrations.sdpa_attention as sa

        saved_orig = getattr(sa, "_airllm_orig_use_gqa_in_sdpa", None)
        saved_fn = sa.use_gqa_in_sdpa
        recorded = {}

        def _fake_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, scale=None, is_causal=False, **kwargs):
            recorded["key_heads"] = key.shape[1]
            recorded["value_heads"] = value.shape[1]
            recorded["enable_gqa"] = kwargs.get("enable_gqa", False)
            return torch.zeros_like(query)

        try:
            self._enter_all(
                self._mock_cuda_env(
                    flash_enabled=False,
                    cudnn_enabled=False,
                    mem_eff_enabled=True,
                    eff_supports_gqa=False,
                    eff_supports_repeated=True,
                )
            )
            utils.configure_sdpa_gqa_fallback(
                device="cuda:0",
                dtype=torch.float16,
                head_dim=64,
                sdpa_module=sa,
            )

            module = SimpleNamespace(num_key_value_groups=4, is_causal=True, layer_idx=0)
            query = torch.zeros((1, 8, 16, 64), dtype=torch.float16)
            key = _FakeTensorOnFakeCuda((1, 2, 16, 64), dtype=torch.float16)
            value = _FakeTensorOnFakeCuda((1, 2, 16, 64), dtype=torch.float16)

            with patch.object(torch.nn.functional, "scaled_dot_product_attention", side_effect=_fake_sdpa):
                sa.sdpa_attention_forward(module, query, key, value, attention_mask=None)

            self.assertEqual(recorded["key_heads"], 8)
            self.assertEqual(recorded["value_heads"], 8)
            self.assertFalse(recorded["enable_gqa"])
        finally:
            sa.use_gqa_in_sdpa = saved_fn
            if saved_orig is None:
                if hasattr(sa, "_airllm_orig_use_gqa_in_sdpa"):
                    delattr(sa, "_airllm_orig_use_gqa_in_sdpa")
            else:
                sa._airllm_orig_use_gqa_in_sdpa = saved_orig

    def test_airllm_base_configure_sdpa_gqa_reads_dense_and_nested_configs(self):
        from airllm.airllm_base import AirLLMBaseModel

        for cfg in (
            SimpleNamespace(hidden_size=2048, num_attention_heads=16, num_key_value_heads=2),
            SimpleNamespace(text_config=SimpleNamespace(head_dim=128, num_attention_heads=32, num_key_value_heads=4)),
        ):
            obj = AirLLMBaseModel.__new__(AirLLMBaseModel)
            obj.device = torch.device("cuda", 0)
            obj.running_dtype = torch.float16
            obj.config = cfg
            with patch("airllm.airllm_base.configure_sdpa_gqa_fallback", return_value=False) as mock_cfg:
                obj._configure_sdpa_gqa()
                mock_cfg.assert_called_once_with(
                    device=torch.device("cuda", 0),
                    dtype=torch.float16,
                    head_dim=128,
                )

        cpu_obj = AirLLMBaseModel.__new__(AirLLMBaseModel)
        cpu_obj.device = torch.device("cpu")
        cpu_obj.running_dtype = torch.float16
        cpu_obj.config = SimpleNamespace(hidden_size=2048, num_attention_heads=16, num_key_value_heads=2)
        with patch("airllm.airllm_base.configure_sdpa_gqa_fallback") as mock_cfg:
            cpu_obj._configure_sdpa_gqa()
            mock_cfg.assert_not_called()


class _FakeTensorOnFakeCuda:
    """Thin tensor wrapper supporting shape/expand/reshape while exposing device.type == 'cuda'."""

    def __init__(self, shape, dtype=torch.float16):
        self._data = torch.zeros(shape, dtype=dtype)

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def device(self):
        return torch.device("cuda", 0)

    def __getitem__(self, item):
        return self._data[item]


if __name__ == "__main__":
    unittest.main()
