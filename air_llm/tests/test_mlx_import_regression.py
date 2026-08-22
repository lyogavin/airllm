"""Dependency-free regression test for issue #326."""
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "airllm" / "airllm_llama_mlx.py"


def test_mlx_module_does_not_eagerly_import_unused_llama_class():
    source = SOURCE.read_text(encoding="utf-8")
    import_lines = [line for line in source.splitlines() if line.startswith("from transformers import")]
    assert import_lines, "expected the MLX module's transformers import"
    assert all("LlamaForCausalLM" not in line for line in import_lines)
