"""Smoke-test streamed LoRA on Qwen3.8-27B and print peak VRAM.

v1 is text-only, batch=1. Start at ``--seq-len 512`` on a 4GB card; 2048 is the
8GB target. This is not Hugging Face Trainer / bitsandbytes QLoRA — see
``airllm.airllm_lora.AirLLMLoRA``.

Example::

    python examples/train_qwen38_lora.py --model Qwen/Qwen3.8-27B --seq-len 512 --steps 1
"""

import argparse
import os
import sys

import torch

_AIRLLM = os.path.join(os.path.dirname(__file__), "..")
if _AIRLLM not in sys.path:
    sys.path.insert(0, os.path.abspath(_AIRLLM))

from airllm.airllm_lora import AirLLMLoRA


def parse_args():
    p = argparse.ArgumentParser(description="AirLLM streamed LoRA smoke test (Qwen3.8-27B)")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ce-chunk-size", type=int, default=4096)
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="shard/runtime cap; defaults to --seq-len")
    p.add_argument("--save-adapter", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required for this smoke test.")
        sys.exit(1)

    max_seq_len = args.max_seq_len or args.seq_len
    trainer = AirLLMLoRA(
        args.model,
        device=args.device,
        max_seq_len=max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        ce_chunk_size=args.ce_chunk_size,
    )

    tok = trainer.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    text = "AirLLM trains LoRA with one decoder layer on the GPU at a time. " * 256
    encoded = tok(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
    input_ids = encoded["input_ids"].to(trainer.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(trainer.device)
    print(f"tokens={input_ids.shape[1]} requested={args.seq_len}", flush=True)

    torch.cuda.reset_peak_memory_stats(trainer.device)
    for step in range(args.steps):
        loss = trainer.train_step(
            input_ids, attention_mask=attention_mask, verbose=args.verbose)
        peak = torch.cuda.max_memory_allocated(trainer.device) / 1024 ** 3
        print(f"step {step} loss={loss:.4f} peak_vram={peak:.2f}GB seq={input_ids.shape[1]}", flush=True)

    if args.save_adapter:
        trainer.save_adapter(args.save_adapter)
        print(f"saved adapter to {args.save_adapter}")


if __name__ == "__main__":
    main()
