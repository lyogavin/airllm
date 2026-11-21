"""Unit tests for GQA SDPA math-kernel fallback avoidance on CUDA (issue #396)."""

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

_AIRLLM_DIR = Path(__file__).resolve().parents[1] / "airllm"

if "airllm" not in sys.modules:
    _pkg = types.ModuleType("airllm")
    _pkg.__path__ = [str(_AIRLLM_DIR)]
    sys.modules["airllm"] = _pkg

from airllm import utils


class _FakeCudaTensor:
    """Minimal key/value lookalike reporting device='cuda:0' on CPU-only test runners."""

    def __init__(self, shape=(1, 2, 16, 128), dtype=torch.float16, device_index=0):
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device("cuda", device_index)


class _FakeSDPAParams:
    def __init__(self, query, key, value, attn_mask, dropout_p, is_causal, enable_gqa):
        self.query = query
        self.key = key
        self.value = value
        self.attn_mask = attn_mask
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.enable_gqa = enable_gqa


def _orig_use_gqa(attention_mask, key, value):
    return attention_mask is None and key.shape[-1] == value.shape[-1] <= 256


class TestSdpaGqaFallback(unittest.TestCase):
    def setUp(self):
        utils._CUDA_SDPA_GQA_CACHE.clear()
        self.fake_sa = SimpleNamespace(use_gqa_in_sdpa=_orig_use_gqa)

    def tearDown(self):
        utils._CUDA_SDPA_GQA_CACHE.clear()

    def _mock_cuda_env(
        self,
        *,
        flash_enabled=True,
        cudnn_enabled=True,
        mem_eff_enabled=True,
        can_flash=lambda p: False,
        can_cudnn=lambda p: False,
        can_eff=lambda p: not p.enable_gqa,
    ):
        orig_empty = torch.empty

        def _fake_empty(*args, **kw):
            dev = kw.get("device")
            if dev is not None and torch.device(dev).type == "cuda":
                shape = args[0] if args else kw.get("size")
                dtype = kw.get("dtype", torch.float16)
                return _FakeCudaTensor(shape, dtype)
            return orig_empty(*args, **kw)

        return (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.backends.cuda, "SDPAParams", _FakeSDPAParams, create=True),
            patch.object(torch.backends.cuda, "flash_sdp_enabled", side_effect=lambda: flash_enabled, create=True),
            patch.object(torch.backends.cuda, "cudnn_sdp_enabled", side_effect=lambda: cudnn_enabled, create=True),
            patch.object(
                torch.backends.cuda, "mem_efficient_sdp_enabled", side_effect=lambda: mem_eff_enabled, create=True
            ),
            patch.object(torch.backends.cuda, "can_use_flash_attention", side_effect=can_flash, create=True),
            patch.object(torch.backends.cuda, "can_use_cudnn_attention", side_effect=can_cudnn, create=True),
            patch.object(torch.backends.cuda, "can_use_efficient_attention", side_effect=can_eff, create=True),
            patch.object(torch, "empty", side_effect=_fake_empty),
        )

    def _enter_all(self, patches):
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_disables_gqa_in_sdpa_when_flash_unavailable_and_mem_efficient_works(self):
        self._enter_all(
            self._mock_cuda_env(
                can_flash=lambda p: False,
                can_cudnn=lambda p: False,
                can_eff=lambda p: not p.enable_gqa,
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
        self.assertFalse(self.fake_sa.use_gqa_in_sdpa(None, key, key))

    def test_preserves_native_gqa_when_flash_accepts_gqa(self):
        self._enter_all(
            self._mock_cuda_env(
                can_flash=lambda p: p.enable_gqa,
                can_cudnn=lambda p: False,
                can_eff=lambda p: not p.enable_gqa,
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

    def test_preserves_native_gqa_when_only_math_is_available(self):
        self._enter_all(
            self._mock_cuda_env(
                can_flash=lambda p: False,
                can_cudnn=lambda p: False,
                can_eff=lambda p: False,
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
                patch.object(torch.backends.cuda, "SDPAParams", _FakeSDPAParams, create=True),
                patch.object(torch.backends.cuda, "flash_sdp_enabled", side_effect=lambda: flag["flash"], create=True),
                patch.object(torch.backends.cuda, "cudnn_sdp_enabled", side_effect=lambda: False, create=True),
                patch.object(torch.backends.cuda, "mem_efficient_sdp_enabled", side_effect=lambda: True, create=True),
                patch.object(torch.backends.cuda, "can_use_flash_attention", side_effect=lambda p: p.enable_gqa, create=True),
                patch.object(torch.backends.cuda, "can_use_cudnn_attention", side_effect=lambda p: False, create=True),
                patch.object(
                    torch.backends.cuda, "can_use_efficient_attention", side_effect=lambda p: not p.enable_gqa, create=True
                ),
                patch.object(
                    torch,
                    "empty",
                    side_effect=lambda shape, device=None, dtype=torch.float16, **kw: _FakeCudaTensor(shape, dtype),
                ),
            )
        )
        utils.configure_sdpa_gqa_fallback(sdpa_module=self.fake_sa)
        utils.configure_sdpa_gqa_fallback(sdpa_module=self.fake_sa)
        self.assertIs(self.fake_sa._airllm_orig_use_gqa_in_sdpa, _orig_use_gqa)

        key = _FakeCudaTensor(shape=(1, 2, 16, 128), dtype=torch.float16)
        self.assertTrue(self.fake_sa.use_gqa_in_sdpa(None, key, key))
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
                    can_flash=lambda p: False,
                    can_cudnn=lambda p: False,
                    can_eff=lambda p: not p.enable_gqa,
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
        import importlib.machinery
        import importlib.util

        stub_mods = {}
        if importlib.util.find_spec("accelerate") is None:
            import transformers  # noqa: F401 - let transformers finish dependency_versions_check first

            fake_accelerate = types.ModuleType("accelerate")
            fake_accelerate.__spec__ = importlib.machinery.ModuleSpec("accelerate", None)
            fake_accelerate.init_empty_weights = lambda *a, **kw: None
            fake_acc_utils = types.ModuleType("accelerate.utils")
            fake_acc_modeling = types.ModuleType("accelerate.utils.modeling")
            fake_acc_modeling.set_module_tensor_to_device = lambda *a, **kw: None
            stub_mods = {
                "accelerate": fake_accelerate,
                "accelerate.utils": fake_acc_utils,
                "accelerate.utils.modeling": fake_acc_modeling,
            }

        # Only track temporary stub keys so lazy PyTorch/Transformers submodules
        # loaded during import stay registered in sys.modules across later tests.
        saved_stubs = {name: sys.modules.get(name) for name in stub_mods}
        sys.modules.update(stub_mods)
        try:
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
        finally:
            for name, prev in saved_stubs.items():
                if prev is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prev


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
