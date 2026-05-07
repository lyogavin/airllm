import sys
import time
import os
import gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "air_llm"))

import mlx.core as mx
from airllm.airllm_llama_mlx import AirLLMLlamaMlx
from airllm.persist import ModelPersister

mp = ModelPersister.get_model_persister()
orig_load = mp.load_model
load_timings = []
def timed_load(layer_name, path):
    t0 = time.perf_counter()
    out = orig_load(layer_name, path)
    dt = time.perf_counter() - t0
    load_timings.append((layer_name, dt))
    return out
mp.load_model = timed_load

MODEL = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 3

def run(label, reuse_modules, compile_block=False):
    print(f"\n========== RUN: {label}  reuse_modules={reuse_modules}  compile_block={compile_block} ==========")
    load_timings.clear()
    gc.collect()

    t0 = time.perf_counter()
    model = AirLLMLlamaMlx(MODEL, show_memory_util=False,
                            reuse_modules=reuse_modules, compile_block=compile_block)
    t_init = time.perf_counter() - t0
    print(f"[init]  from_pretrained: {t_init:6.2f}s   "
          f"(n_layers={model.model_args.n_layers} dim={model.model_args.dim})")

    init_loads = list(load_timings)
    load_timings.clear()

    input_tokens = model.tokenizer(["I like"], return_tensors="np",
        return_attention_mask=False, truncation=True, max_length=128, padding=False)

    t0 = time.perf_counter()
    out = model.generate(mx.array(input_tokens["input_ids"]), max_new_tokens=MAX_NEW,
                         use_cache=True, return_dict_in_generate=True)
    t_gen = time.perf_counter() - t0

    print(f"[gen]   total: {t_gen:6.2f}s  ({t_gen/MAX_NEW:.3f}s/token)  output: {out!r}")

    total_load = sum(dt for _, dt in load_timings)
    by_kind = {}
    for name, dt in load_timings:
        if "embed" in name:    kind = "embed"
        elif "layers." in name: kind = "block"
        elif "norm" in name:   kind = "norm"
        elif "lm_head" in name or "output" in name: kind = "lm_head"
        else: kind = name
        by_kind.setdefault(kind, []).append(dt)

    print(f"[disk]  during gen: {total_load:.2f}s = {total_load/t_gen*100:.1f}% of gen")
    for k, ts in by_kind.items():
        print(f"        {k:8s}  count={len(ts):3d}  total={sum(ts):.2f}s  avg={sum(ts)/len(ts)*1000:.1f}ms")

    del model
    gc.collect()
    return t_init, t_gen, total_load

print(f"=== model: {MODEL}, max_new_tokens={MAX_NEW} ===")
# Run baseline first then progressively-optimized variants
t1_init, t1_gen, t1_disk = run("BASELINE", reuse_modules=False)
t2_init, t2_gen, t2_disk = run("REUSE_MODULES", reuse_modules=True)
t3_init, t3_gen, t3_disk = run("REUSE+COMPILE", reuse_modules=True, compile_block=True)

print("\n========== A/B/C SUMMARY ==========")
print(f"               baseline       reuse       reuse+compile")
print(f"  gen total:   {t1_gen:7.2f}s     {t2_gen:7.2f}s     {t3_gen:7.2f}s")
print(f"  gen/token:   {t1_gen/MAX_NEW:7.3f}s     {t2_gen/MAX_NEW:7.3f}s     {t3_gen/MAX_NEW:7.3f}s")
print(f"  vs baseline:                  {(t2_gen-t1_gen)/t1_gen*100:+5.1f}%        {(t3_gen-t1_gen)/t1_gen*100:+5.1f}%")
print(f"  vs reuse:                                   {(t3_gen-t2_gen)/t2_gen*100:+5.1f}%")
print(f"  disk gen:    {t1_disk:7.2f}s     {t2_disk:7.2f}s     {t3_disk:7.2f}s")