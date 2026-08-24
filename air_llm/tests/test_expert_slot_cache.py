import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from airllm.airllm_base import AirLLMBaseModel
from airllm.airllm_phimoe import AirLLMPhiMoE
from airllm.expert_slot_cache import ExpertHostBank, ExpertSlotRuntime, FixedSlotLRU


def _checkpoint_tensors(second_w1_shape=(3, 4)):
    tensors = {}
    for expert in range(2):
        prefix = f"model.layers.0.block_sparse_moe.experts.{expert}"
        w1_shape = (3, 4) if expert == 0 else second_w1_shape
        tensors[f"{prefix}.w1.weight"] = torch.arange(
            torch.tensor(w1_shape).prod(), dtype=torch.bfloat16
        ).reshape(w1_shape) + expert
        tensors[f"{prefix}.w2.weight"] = (
            torch.arange(12, dtype=torch.bfloat16).reshape(4, 3) + expert
        )
        tensors[f"{prefix}.w3.weight"] = (
            torch.arange(12, dtype=torch.bfloat16).reshape(3, 4) + expert
        )
    return tensors


def _bank_args(root):
    keys = {
        0: {
            expert: [
                f"model.layers.0.block_sparse_moe.experts.{expert}.{projection}.weight"
                for projection in ("w1", "w2", "w3")
            ]
            for expert in range(2)
        }
    }
    prefixes = {
        (0, expert): f"model.layers.0.block_sparse_moe.experts.{expert}"
        for expert in range(2)
    }
    return root, ["model.layers.0"], keys, prefixes, torch.float16


class TestFixedSlotLRU(unittest.TestCase):
    def test_hit_refreshes_recency_and_reservation_reuses_fixed_slot(self):
        lru = FixedSlotLRU(2)
        first = lru.reserve((0, 0))
        second = lru.reserve((0, 1))
        self.assertEqual((first.slot, second.slot), (0, 1))

        self.assertEqual(lru.lookup((0, 0)), 0)
        third = lru.reserve((1, 0))
        self.assertEqual(third.slot, 1)
        self.assertEqual(third.evicted_key, (0, 1))
        self.assertEqual(lru.keys(), ((0, 0), (1, 0)))

    def test_active_experts_are_not_selected_as_victims(self):
        lru = FixedSlotLRU(2)
        lru.reserve((0, 0))
        lru.reserve((0, 1))
        self.assertIsNone(lru.reserve((0, 2), protected=((0, 0), (0, 1))))

    def test_clear_reuses_all_preallocated_slot_ids(self):
        lru = FixedSlotLRU(2)
        lru.reserve((0, 0))
        lru.reserve((0, 1))
        lru.clear()
        self.assertEqual(lru.resident_count, 0)
        self.assertEqual(lru.reserve((1, 0)).slot, 0)


class TestSlotBackendConfiguration(unittest.TestCase):
    def test_unknown_backend_is_rejected_before_checkpoint_access(self):
        with self.assertRaisesRegex(ValueError, "must be 'none' or 'slot'"):
            AirLLMBaseModel("unused", expert_cache_backend="unknown")

    def test_budget_requires_explicit_slot_backend(self):
        with self.assertRaisesRegex(ValueError, "requires expert_cache_backend='slot'"):
            AirLLMPhiMoE("unused", expert_cache_gb=0.75)

    def test_slot_backend_requires_positive_budget(self):
        with self.assertRaisesRegex(ValueError, "requires expert_cache_gb > 0"):
            AirLLMPhiMoE("unused", expert_cache_backend="slot")

    def test_slot_backend_rejects_cpu_device_before_checkpoint_access(self):
        with self.assertRaisesRegex(ValueError, "requires a CUDA-compatible device"):
            AirLLMPhiMoE(
                "unused",
                device="cpu",
                expert_cache_gb=0.75,
                expert_cache_backend="slot",
            )

    def test_slot_backend_rejects_unavailable_accelerator_before_checkpoint_access(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "requires a CUDA-compatible device"):
                AirLLMPhiMoE(
                    "unused",
                    device="cuda:0",
                    expert_cache_gb=0.75,
                    expert_cache_backend="slot",
                )


class TestExpertHostBank(unittest.TestCase):
    def test_normalizes_checkpoint_experts_into_contiguous_runtime_dtype_banks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _checkpoint_tensors()
            save_file(source, root / "model.layers.0.safetensors")

            bank = ExpertHostBank(*_bank_args(root))

            self.assertEqual(bank.expert_count, 2)
            self.assertEqual(bank.source_tensor_reads, 6)
            self.assertEqual(bank.projection_names, ("w1.weight", "w2.weight", "w3.weight"))
            self.assertTrue(all(tensor.is_contiguous() for tensor in bank.banks.values()))
            self.assertTrue(all(tensor.dtype == torch.float16 for tensor in bank.banks.values()))
            actual = bank.tensors((0, 1))
            for projection, tensor in zip(bank.projection_names, actual):
                expected = source[
                    f"model.layers.0.block_sparse_moe.experts.1.{projection}"
                ].to(torch.float16)
                torch.testing.assert_close(tensor, expected)

    def test_rejects_non_uniform_expert_shapes_before_building_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file(_checkpoint_tensors(second_w1_shape=(2, 4)),
                      root / "model.layers.0.safetensors")
            with self.assertRaisesRegex(ValueError, "non-uniform w1.weight shape"):
                ExpertHostBank(*_bank_args(root))

    def test_rejects_missing_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _checkpoint_tensors()
            source.pop("model.layers.0.block_sparse_moe.experts.1.w3.weight")
            save_file(source, root / "model.layers.0.safetensors")
            args = list(_bank_args(root))
            args[2][0][1].remove(
                "model.layers.0.block_sparse_moe.experts.1.w3.weight"
            )
            with self.assertRaisesRegex(ValueError, "must contain exactly"):
                ExpertHostBank(*args)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA-compatible GPU")
class TestExpertSlotRuntimeGPU(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        save_file(_checkpoint_tensors(), root / "model.layers.0.safetensors")
        self.bank = ExpertHostBank(*_bank_args(root))
        self.runtime = ExpertSlotRuntime(
            self.bank,
            self.bank.expert_bytes,
            torch.device("cuda:0"),
        )

    def tearDown(self):
        self.runtime.close()
        del self.runtime
        torch.cuda.empty_cache()
        self.tmp.cleanup()

    def _capture(self, expert_indices, admit):
        captured = {}

        def callback(expert_idx, weights):
            captured[expert_idx] = tuple(weight.clone() for weight in weights)

        self.runtime.execute(0, expert_indices, admit=admit, callback=callback)
        torch.cuda.synchronize()
        return captured

    def test_evictions_reuse_fixed_addresses_and_preserve_values(self):
        pointers = tuple(bank.data_ptr() for bank in self.runtime.cache_banks.values())
        for expert_idx in (0, 1, 0):
            captured = self._capture([expert_idx], admit=True)
            for actual, expected in zip(captured[expert_idx], self.bank.tensors((0, expert_idx))):
                torch.testing.assert_close(actual.cpu(), expected)
        self.assertEqual(
            pointers,
            tuple(bank.data_ptr() for bank in self.runtime.cache_banks.values()),
        )
        self.assertEqual(self.runtime.stats()["evictions"], 2)

    def test_two_lane_prefill_does_not_admit_scratch_experts(self):
        captured = self._capture([0, 1], admit=False)
        self.assertEqual(set(captured), {0, 1})
        stats = self.runtime.stats()
        self.assertEqual(stats["prefill_skips"], 2)
        self.assertEqual(stats["resident_experts"], 0)
        self.assertEqual(stats["generation_disk_expert_reads"], 0)

    def test_reset_preserves_slots_and_clear_only_invalidates_residency(self):
        pointers = tuple(bank.data_ptr() for bank in self.runtime.cache_banks.values())
        self._capture([0], admit=True)
        self._capture([0], admit=True)
        self.assertEqual(self.runtime.stats()["hits"], 1)

        self.runtime.reset_stats()
        self.assertEqual(self.runtime.stats()["hits"], 0)
        self.assertEqual(self.runtime.stats()["resident_experts"], 1)
        self.runtime.clear()
        self.assertEqual(self.runtime.stats()["resident_experts"], 0)
        self.assertEqual(
            pointers,
            tuple(bank.data_ptr() for bank in self.runtime.cache_banks.values()),
        )

    def test_callback_failure_invalidates_residency_and_runtime_remains_usable(self):
        def fail(_expert_idx, _weights):
            raise RuntimeError("synthetic callback failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic callback failure"):
            self.runtime.execute(0, [0], admit=True, callback=fail)

        self.assertFalse(self.runtime._in_execute)
        self.assertEqual(self.runtime.stats()["resident_experts"], 0)
        captured = self._capture([1], admit=True)
        self.assertEqual(set(captured), {1})


if __name__ == "__main__":
    unittest.main()
