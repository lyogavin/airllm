"""Benchmark AirLLM's fixed-slot PhiMoE runtime on a reproducible public model."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from airllm import AutoModel


DEFAULT_MODEL = "microsoft/Phi-tiny-MoE-instruct"
DEFAULT_REVISION = "2fe50e88d0e2a5a132563815686ea0dcc8e252b5"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--expert-cache-gb", type=float, default=0.0)
    parser.add_argument(
        "--expert-cache-backend",
        choices=("none", "slot"),
        default="none",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument(
        "--prompt",
        default="Explain mixture-of-experts routing in one short sentence.",
    )
    parser.add_argument("--shard-dir", type=Path, default=Path("models/airllm-shards/phimoe"))
    parser.add_argument("--hf-cache-dir", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optional path for the machine-readable result (raw results should stay untracked)",
    )
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument(
        "--no-host-warmup",
        action="store_true",
        help="do not read split safetensors once before timing",
    )
    return parser.parse_args()


def resolve_model_path(model, revision, cache_dir):
    local = Path(model)
    if local.exists():
        return local.resolve()
    kwargs = {"repo_id": model, "revision": revision}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return Path(snapshot_download(**kwargs))


def warm_host_pages(checkpoint_path):
    """Populate the OS page cache so timed runs isolate GPU expert residency."""
    total = 0
    for shard in sorted(Path(checkpoint_path).glob("*.safetensors")):
        with shard.open("rb", buffering=0) as handle:
            while True:
                chunk = handle.read(8 * 1024 ** 2)
                if not chunk:
                    break
                total += len(chunk)
    return total


def sequences_from(output):
    return output.sequences if hasattr(output, "sequences") else output


def main():
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least one")
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-compatible PyTorch device (CUDA or ROCm)")

    model_path = resolve_model_path(args.model, args.revision, args.hf_cache_dir)
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(model_path),
        device=args.device,
        dtype=dtype,
        max_seq_len=args.max_seq_len,
        layer_shards_saving_path=str(args.shard_dir),
        prefetching=not args.no_prefetch,
        expert_cache_gb=args.expert_cache_gb,
        expert_cache_backend=args.expert_cache_backend,
    )
    load_seconds = time.perf_counter() - load_started

    encoded = model.tokenizer(
        args.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_len,
        padding=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    warmed_host_bytes = 0
    if not args.no_host_warmup:
        warmed_host_bytes = warm_host_pages(model.checkpoint_path)

    # Compile/initialize the real kernels and generation path outside the timed repetitions.
    model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        min_new_tokens=1,
        do_sample=False,
        use_cache=True,
    )
    torch.cuda.synchronize(device)
    model.clear_expert_cache()

    repetitions = []
    expected_ids = None
    for repetition in range(args.repetitions):
        model.clear_expert_cache()
        model.reset_expert_cache_stats()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        started = time.perf_counter()
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        sequences = sequences_from(output)
        generated = sequences[0, input_ids.shape[-1]:].detach().cpu().tolist()
        if len(generated) != args.max_new_tokens:
            raise RuntimeError(
                f"expected exactly {args.max_new_tokens} generated tokens, got {len(generated)}"
            )
        if expected_ids is None:
            expected_ids = generated
        elif generated != expected_ids:
            raise RuntimeError(
                f"non-deterministic output in repetition {repetition}: "
                f"expected {expected_ids}, got {generated}"
            )
        repetitions.append({
            "repetition": repetition + 1,
            "generation_seconds": elapsed,
            "generated_tokens": len(generated),
            "tokens_per_second": len(generated) / elapsed,
            "generated_token_ids": generated,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024 ** 2,
            "expert_cache": model.get_expert_cache_stats(),
        })
        print(
            f"repetition {repetition + 1}/{args.repetitions}: "
            f"{elapsed:.3f}s ({len(generated) / elapsed:.5f} tok/s)",
            flush=True,
        )

    result = {
        "model": args.model,
        "model_path": str(model_path),
        "revision": args.revision,
        "device": torch.cuda.get_device_name(device),
        "architecture": getattr(torch.cuda.get_device_properties(device), "gcnArchName", None),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "dtype": args.dtype,
        "prefetch_requested": not args.no_prefetch,
        "prefetch": model.prefetching,
        "expert_cache_gb": args.expert_cache_gb,
        "expert_cache_backend": args.expert_cache_backend,
        "prompt": args.prompt,
        "input_tokens": input_ids.shape[-1],
        "max_new_tokens": args.max_new_tokens,
        "load_seconds": load_seconds,
        "host_warmup_bytes": warmed_host_bytes,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "median_generation_seconds": statistics.median(
            item["generation_seconds"] for item in repetitions
        ),
        "median_tokens_per_second": statistics.median(
            item["tokens_per_second"] for item in repetitions
        ),
        "repetitions": repetitions,
        "generated_text": model.tokenizer.decode(expected_ids, skip_special_tokens=True),
    }
    payload = json.dumps(result, ensure_ascii=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print("AIRLLM_EXPERT_CACHE_BENCHMARK=" + payload)
    model.clear_expert_cache()


if __name__ == "__main__":
    main()
