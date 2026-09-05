"""Streamed LoRA on Qwen3.8-Flash-Next (125B MoE + 51B PLE).

Shared-expert / attention / hyper-connection / PLE Linears train; packed-expert
LoRA is off by default (48 layers of 512-expert adapters plus Adam will not
fit). Measured peak is under 6GB at seq 512 (RTX 3060 Ti / 4090). The n-gram
table stays mmap'd.

Needs a transformers build with in-tree ``qwen4_exp``::

    pip install git+https://github.com/huggingface/transformers.git
    python air_llm/examples/train_qwen38_flash_next_lora.py --data my_data.jsonl --save-adapter out.pt
"""

import argparse
import os
import sys
import time

import torch

_AIRLLM = os.path.join(os.path.dirname(__file__), "..")
if _AIRLLM not in sys.path:
    sys.path.insert(0, os.path.abspath(_AIRLLM))

from airllm.airllm_lora import AirLLMLoRAQwen4Exp
from airllm.lora_data import encode_record, iter_train_items, load_records

_FALLBACK = (
    "AirLLM trains LoRA with one decoder layer on the GPU at a time. "
    "Frozen Qwen3.8-Flash-Next experts stay on disk until a layer is needed. "
    "The n-gram PLE table is file-mapped on the host and never copied into VRAM. "
    "Hyper-connections mix four residual streams before the language-model head. "
    "Gated DeltaNet layers use a recurrent state; sparse-attention layers use QSA. "
    "Adam keeps only the adapter matrices resident, about 168 megabytes at rank 16. "
    "A packed MoE layer is five gigabytes of bfloat16; streaming it is the whole trick. "
    "Loss should fall as the adapters fit this short supervised sequence. "
) * 32


def parse_args():
    p = argparse.ArgumentParser(description="AirLLM streamed LoRA (Qwen3.8-Flash-Next)")
    p.add_argument("--data", default=None,
                   help="JSONL/JSON/TXT dataset (see README Training). Omit to overfit a built-in snippet.")
    p.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=None,
                   help="stop after N steps (default: all examples × epochs; 1 if --data is omitted)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ce-chunk-size", type=int, default=4096)
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="shard/runtime cap; defaults to --seq-len")
    p.add_argument("--save-adapter", default=None)
    p.add_argument("--packed-expert-lora", action="store_true",
                   help="also LoRA the 512 packed experts (VRAM-heavy)")
    p.add_argument("--keep-original", action="store_true",
                   help="keep the Hugging Face shard files after splitting")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-csv", default=None,
                   help="write step,loss,peak_vram_gb,elapsed_s")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required for this trainer.")
        sys.exit(1)

    if args.data:
        records = load_records(args.data)
        epochs, steps = args.epochs, args.steps
    else:
        records = [{"text": _FALLBACK, "mask_prefix": None}]
        epochs, steps = (args.steps or 1), None

    max_seq_len = args.max_seq_len or args.seq_len
    trainer = AirLLMLoRAQwen4Exp(
        args.model,
        device=args.device,
        max_seq_len=max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        ce_chunk_size=args.ce_chunk_size,
        delete_original=not args.keep_original,
        packed_experts=args.packed_expert_lora,
    )

    tok = trainer.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"examples={len(records)} epochs={epochs} seq_len={args.seq_len}", flush=True)

    csv_f = None
    if args.log_csv:
        csv_f = open(args.log_csv, "w", encoding="utf-8")
        csv_f.write("step,loss,peak_vram_gb,elapsed_s\n")
        csv_f.flush()

    torch.cuda.reset_peak_memory_stats(trainer.device)
    t0 = time.time()
    n = 0
    for step, epoch, rec in iter_train_items(records, epochs, steps):
        input_ids, attention_mask, labels = encode_record(
            tok, rec, args.seq_len, trainer.device)
        loss = trainer.train_step(
            input_ids, labels=labels, attention_mask=attention_mask, verbose=args.verbose)
        peak = torch.cuda.max_memory_allocated(trainer.device) / 1024 ** 3
        elapsed = time.time() - t0
        print(
            f"step {step} epoch {epoch} loss={loss:.6f} peak_vram={peak:.2f}GB "
            f"seq={input_ids.shape[1]} elapsed_s={elapsed:.1f}",
            flush=True,
        )
        if csv_f is not None:
            csv_f.write(f"{step},{loss:.8f},{peak:.4f},{elapsed:.2f}\n")
            csv_f.flush()
        n += 1
    if csv_f is not None:
        csv_f.close()

    if n == 0:
        print("no training steps ran")
        sys.exit(1)

    if args.save_adapter:
        trainer.save_adapter(args.save_adapter)
        print(f"saved adapter to {args.save_adapter}")


if __name__ == "__main__":
    main()
