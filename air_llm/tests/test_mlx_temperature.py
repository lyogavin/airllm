"""Dependency-free regression test: model_generate must pass temperature to every sample() call.

It used to call sample(logits) without it, so temperature > 0 was silently ignored and output
was always greedy.
"""
import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "airllm" / "airllm_llama_mlx.py"


def _sample_calls_in(func):
    return [node for node in ast.walk(func)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "sample"]


class TestMlxTemperature(unittest.TestCase):
    def test_model_generate_passes_temperature_to_sample(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "model_generate"]
        self.assertEqual(len(funcs), 1, "expected exactly one model_generate")

        calls = _sample_calls_in(funcs[0])
        self.assertTrue(calls, "model_generate should sample tokens")
        for call in calls:
            passed = [ast.unparse(a) for a in call.args[1:]] + \
                     [ast.unparse(k.value) for k in call.keywords if k.arg == "temperature"]
            self.assertIn("temperature", passed, f"sample() call ignores temperature: {ast.unparse(call)}")


if __name__ == "__main__":
    unittest.main()
