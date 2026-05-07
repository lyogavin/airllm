"""Smoke test for MLX-native 4bit quantization on AirLLM.

Reads same model in fp16 and 4bit modes, compares:
  - shard size on disk
  - generation correctness (4bit should produce sensible output, not necessarily identical to fp16)
  - generation speed
"""
import sys
import time
import os
import gc
import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "air_llm"))

import mlx.core as mx
from airllm.airllm_llama_mlx import AirLLMLlamaMlx

MODEL = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 5

def shard_size(model):
    pat = os.path.join(str(model.checkpoint_path), "*.mlx.npz")
    files = glob.glob(pat)
    return sum(os.path.getsize(f) for f in files), len(files)

def run(label, compression):
    print(f"\n========== {label}  compression={compression!r} ==========")
    gc.collect()
    t0 = time.perf_counter()
    model = AirLLMLlamaMlx(MODEL, show_memory_util=False, compression=compression)
    t_init = time.perf_counter() - t0
    size_b, n_files = shard_size(model)
    print(f"[init] {t_init:6.2f}s  shards: {n_files} files, {size_b/1024/1024:.1f} MB total")

    input_tokens = model.tokenizer(["I like"], return_tensors="np",
        return_attention_mask=False, truncation=True, max_length=128, padding=False)
    t0 = time.perf_counter()
    out = model.generate(mx.array(input_tokens["input_ids"]), max_new_tokens=MAX_NEW,
                         use_cache=True, return_dict_in_generate=True)
    t_gen = time.perf_counter() - t0
    print(f"[gen]  {t_gen:6.2f}s  ({t_gen/MAX_NEW:.2f}s/token)  output: {out!r}")
    del model
    gc.collect()
    return size_b, t_gen, out

print(f"=== model: {MODEL}, max_new_tokens={MAX_NEW} ===")
sz_fp16, t_fp16, out_fp16 = run("FP16 (default)", compression=None)
sz_4bit, t_4bit, out_4bit = run("4BIT (mlx)",     compression='4bit')

print("\n========== SUMMARY ==========")
print(f"  fp16:  {sz_fp16/1024/1024:7.1f} MB on disk    {t_fp16:.2f}s gen   output: {out_fp16!r}")
print(f"  4bit:  {sz_4bit/1024/1024:7.1f} MB on disk    {t_4bit:.2f}s gen   output: {out_4bit!r}")
print(f"  ratio: {sz_fp16/sz_4bit:.2f}x smaller on disk")