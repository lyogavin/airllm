"""Repro VRAM / stall bench for AirLLM (issue #295).

Logs verifiable numbers per run so headline VRAM claims can be checked
per commit instead of trusted: device, wall time, tok/s, peak CUDA mem
(if any), host RSS delta, and a slow-call flag (issues #364/#374).

Works with a real ``AutoModel`` wrapper or any object exposing
``generate()`` + ``device`` (see tests for the fake).

Real run (needs model download, ~1GB for Qwen2-0.5B):
    python air_llm/examples/bench_vram.py --repo Qwen/Qwen2-0.5B \\
        --prompt "What is the capital of France?" --max-new-tokens 20

Offline (no GPU, no network): see air_llm/tests/test_bench_vram.py.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, Optional

import torch


def default_device() -> str:
    """CUDA -> Apple MPS -> CPU. Local copy so the script runs on main too."""
    if torch.cuda.is_available():
        return "cuda:0"
    try:
        if hasattr(torch, "backends") and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 — old torch without MPS
        pass
    return "cpu"


def _rss_mb() -> Optional[float]:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**2
    except Exception:  # noqa: BLE001 — psutil optional
        return None


def run_bench(
    wrapper: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    slow_threshold_s: float = 5.0,
) -> Dict[str, Any]:
    """Time one generate() call and report. Never raises on missing CUDA."""
    device = str(getattr(wrapper, "device", None) or default_device())
    try:
        dev_type = torch.device(device).type
    except Exception:  # noqa: BLE001
        dev_type = "cpu"

    rss_before = _rss_mb()
    cuda_before = None
    if dev_type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device)
            cuda_before = torch.cuda.memory_allocated(device) / 1024**2
        except Exception:  # noqa: BLE001
            cuda_before = None

    t0 = time.time()
    try:
        out = wrapper.generate(input_ids, max_new_tokens=max_new_tokens)
    except TypeError:
        # fake wrappers / older signatures: generate(ids)
        out = wrapper.generate(input_ids)
    elapsed = time.time() - t0

    new_tokens = max_new_tokens
    try:
        seq = getattr(out, "sequences", out)
        if isinstance(seq, torch.Tensor) and seq.dim() == 2:
            new_tokens = max(1, seq.shape[1] - input_ids.shape[1])
    except Exception:  # noqa: BLE001
        pass

    peak_cuda = None
    if dev_type == "cuda" and torch.cuda.is_available():
        try:
            peak_cuda = torch.cuda.max_memory_allocated(device) / 1024**2
        except Exception:  # noqa: BLE001
            peak_cuda = None

    rss_after = _rss_mb()
    return {
        "device": device,
        "elapsed_s": round(elapsed, 3),
        "max_new_tokens": max_new_tokens,
        "new_tokens": new_tokens,
        "tok_per_s": round(new_tokens / max(elapsed, 1e-6), 2),
        "slow": elapsed >= slow_threshold_s,
        "slow_threshold_s": slow_threshold_s,
        "cuda_allocated_before_mb": cuda_before,
        "cuda_peak_mb": peak_cuda,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": (
            round(rss_after - rss_before, 1)
            if rss_before is not None and rss_after is not None
            else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--slow-threshold-s", type=float, default=5.0)
    args = ap.parse_args()

    from air_llm.airllm.auto_model import AutoModel

    device = args.device or default_device()
    print(f"auto-detected device: {device}", flush=True)
    model = AutoModel.from_pretrained(args.repo, device=device)
    dev = getattr(model, "device", device)
    toks = model.tokenizer(
        [args.prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=512,
        padding=False,
    )
    ids = toks["input_ids"].to(dev)
    report = run_bench(
        model, ids,
        max_new_tokens=args.max_new_tokens,
        slow_threshold_s=args.slow_threshold_s,
    )
    report["repo"] = args.repo
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
