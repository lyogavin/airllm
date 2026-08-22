"""Smoke-test and benchmark a pre-quantized Hugging Face checkpoint with AirLLM."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from airllm import AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    )
    parser.add_argument("--engine", choices=("airllm", "resident"), default="airllm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--no-prefetch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    using_cuda = torch.device(args.device).type == "cuda"
    started = time.perf_counter()
    if args.engine == "airllm":
        shard_dir = args.shard_dir
        if shard_dir is None:
            shard_dir = Path("models/airllm-shards") / args.model.replace("/", "--")
        shard_dir.mkdir(parents=True, exist_ok=True)
        model = AutoModel.from_pretrained(
            args.model,
            device=args.device,
            max_seq_len=args.max_seq_len,
            layer_shards_saving_path=str(shard_dir),
            profiling_mode=True,
            prefetching=not args.no_prefetch,
        )
        tokenizer = model.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map={"": args.device},
            low_cpu_mem_usage=True,
        )
        model.eval()
    load_seconds = time.perf_counter() - started

    prompt = "Answer with one word: what color is a clear daytime sky?"
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_len,
        padding=False,
    )
    input_ids = encoded["input_ids"].to(args.device)

    if using_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()
    output = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    if using_cuda:
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started

    sequences = output.sequences if hasattr(output, "sequences") else output
    generated_tokens = sequences.shape[-1] - input_ids.shape[-1]
    result = {
        "model": args.model,
        "engine": args.engine,
        "device": args.device,
        "prefetch": not args.no_prefetch,
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(generated_tokens / generation_seconds, 4),
        "peak_process_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "output": tokenizer.decode(sequences[0], skip_special_tokens=True),
    }
    if args.engine == "airllm" and getattr(model, "profiler", None) is not None:
        result["profile_seconds"] = {
            name: round(sum(samples), 3)
            for name, samples in model.profiler.profiling_time_dict.items()
        }
    if using_cuda:
        result["peak_accelerator_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    print("AIRLLM_BENCHMARK=" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
