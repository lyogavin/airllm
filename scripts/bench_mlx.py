"""Time AirLLM's MLX path and check that two versions produce the same tokens.

Works against both the old and the new code, so a reviewer can compare them directly:

    # 1. old code (e.g. a checkout of the commit before the change)
    PYTHONPATH=/path/to/old/air_llm python scripts/bench_mlx.py --save-ids old.json

    # 2. new code
    PYTHONPATH=air_llm python scripts/bench_mlx.py --compare-ids old.json
    PYTHONPATH=air_llm python scripts/bench_mlx.py --compare-ids old.json --layer-cache-bytes auto

Greedy decoding (temperature 0) is deterministic, so the token ids must match exactly.
--layer-cache-bytes only exists in the new code: 0 (the default) streams every layer from disk
on every token like the old code did; "auto" keeps layers in memory while they fit.
"""
import argparse
import json
import os
import resource
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")   # the old code drew progress bars for every token

import mlx.core as mx


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct",
                   help="HF repo id or local path (default: %(default)s)")
    p.add_argument("--tokens", type=int, default=30, help="tokens to generate (default: %(default)s)")
    p.add_argument("--prompt", default="Explain why the sky is blue in two sentences.")
    p.add_argument("--layer-cache-bytes", default=None,
                   help='new code only: "auto", an integer, or "none" for unlimited')
    p.add_argument("--save-ids", help="write the generated token ids to this JSON file")
    p.add_argument("--compare-ids", help="compare the generated token ids with this JSON file")
    a = p.parse_args()

    from airllm import AutoModel
    import airllm
    print(f"airllm: {os.path.dirname(airllm.__file__)}")

    kwargs = {}
    if a.layer_cache_bytes is not None:
        v = a.layer_cache_bytes
        kwargs["layer_cache_bytes"] = None if v == "none" else (v if v == "auto" else int(v))

    model = AutoModel.from_pretrained(a.model, **kwargs)
    tok = model.tokenizer
    prompt = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok([prompt], return_tensors="np", add_special_tokens=False)["input_ids"]

    out, times = [], []
    t = time.perf_counter()
    for y in model.model_generate(mx.array(ids), temperature=0):
        out.append(y.item())
        now = time.perf_counter()
        times.append(now - t)
        t = now
        if len(out) >= a.tokens:
            break

    decode = times[1:]
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss /= 1e9 if sys.platform == "darwin" else 1e6      # bytes on macOS, KiB on Linux
    get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    print(f"prompt tokens:   {ids.shape[1]}")
    print(f"first token:     {times[0]:.3f} s")
    if decode:
        per = sum(decode) / len(decode)
        print(f"decode:          {per:.4f} s/token ({1 / per:.1f} tokens/s)")
    print(f"peak MLX memory: {get_peak() / 1e9:.2f} GB")
    print(f"peak process RSS:{peak_rss:6.2f} GB")
    print(f"text:            {tok.decode(out)!r}")

    if a.save_ids:
        with open(a.save_ids, "w") as f:
            json.dump(out, f)
        print(f"saved ids to {a.save_ids}")
    if a.compare_ids:
        with open(a.compare_ids) as f:
            ref = json.load(f)
        same = sum(x == y for x, y in zip(ref, out))
        ok = ref == out
        print(f"same tokens as {a.compare_ids}: {'YES' if ok else 'NO'} ({same}/{max(len(ref), len(out))})")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
