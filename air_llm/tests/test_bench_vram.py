"""Offline tests for examples/bench_vram.py — no GPU, no network."""

import sys
import unittest

sys.path.insert(0, "examples")

import torch

from bench_vram import default_device, run_bench


class _Fake:
    device = "cpu"

    def generate(self, ids, max_new_tokens=5):
        import time

        time.sleep(0.01)
        b, s = ids.shape
        return torch.zeros(b, s + max_new_tokens, dtype=torch.long)


class TestBenchVram(unittest.TestCase):
    def test_default_device_returns_string(self):
        self.assertIn(default_device(), ("cuda:0", "mps", "cpu"))

    def test_report_keys_and_math(self):
        rep = run_bench(_Fake(), torch.zeros(1, 4, dtype=torch.long),
                        max_new_tokens=5, slow_threshold_s=100.0)
        self.assertEqual(rep["device"], "cpu")
        self.assertEqual(rep["new_tokens"], 5)
        self.assertGreater(rep["tok_per_s"], 0)
        self.assertFalse(rep["slow"])
        for k in ("elapsed_s", "cuda_peak_mb", "rss_before_mb"):
            self.assertIn(k, rep)

    def test_slow_flag(self):
        rep = run_bench(_Fake(), torch.zeros(1, 2, dtype=torch.long),
                        max_new_tokens=2, slow_threshold_s=0.001)
        self.assertTrue(rep["slow"])


if __name__ == "__main__":
    unittest.main()
