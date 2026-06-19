#!/usr/bin/env python3
"""
FP8 quantization validation harness for AMD MI300X (gfx942).

Measures perplexity regression and throughput uplift of FP8 (e4m3fnuz)
vs FP16 baseline on MI300X. Target: <0.5% PPL regression, 1.7-2.0x
throughput uplift at high batch.

Run: python3 bench/fp8_validate.py --model meta-llama/Llama-3.1-8B
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def perplexity(model_logits_fn, text_chunks: list[str], tokenizer, stride: int = 512) -> float:
    """Standard sliding-window perplexity (WikiText-103 style)."""
    nlls, count = 0.0, 0
    for chunk in text_chunks:
        enc = tokenizer(chunk, return_tensors="pt").input_ids
        max_len = enc.size(1)
        for begin in range(0, max_len, stride):
            end = min(begin + stride, max_len)
            input_ids = enc[:, begin:end]
            target_ids = input_ids.clone()
            with torch.no_grad():
                logits = model_logits_fn(input_ids)
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = target_ids[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            nlls += loss.item()
            count += shift_labels.numel()
    return math.exp(nlls / count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--num-chunks", type=int, default=20)
    parser.add_argument("--out", default="results/fp8_validation.json")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading WikiText-103 test split ({args.num_chunks} chunks)...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    chunks = [t for t in ds["text"] if len(t) > 500][: args.num_chunks]

    print(f"[INFO] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # FP16 baseline
    print("\n[1/2] FP16 baseline perplexity...")
    from vllm import LLM, SamplingParams
    llm_fp16 = LLM(model=args.model, dtype="float16", gpu_memory_utilization=0.85, max_model_len=4096)

    def fp16_logits(input_ids):
        # vLLM forward — returns prompt logits only, used for offline PPL
        return llm_fp16.get_model().forward(input_ids.cuda()).logits

    t0 = time.perf_counter()
    ppl_fp16 = perplexity(fp16_logits, chunks, tokenizer)
    fp16_time = time.perf_counter() - t0
    print(f"   PPL(fp16)  = {ppl_fp16:.4f}  ({fp16_time:.1f}s)")

    # FP8 quantized
    print("\n[2/2] FP8 quantized perplexity...")
    llm_fp8 = LLM(
        model=args.model,
        dtype="float16",
        quantization="fp8",
        kv_cache_dtype="fp8_e5m2",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )

    def fp8_logits(input_ids):
        return llm_fp8.get_model().forward(input_ids.cuda()).logits

    t0 = time.perf_counter()
    ppl_fp8 = perplexity(fp8_logits, chunks, tokenizer)
    fp8_time = time.perf_counter() - t0
    print(f"   PPL(fp8)   = {ppl_fp8:.4f}  ({fp8_time:.1f}s)")

    ppl_delta = (ppl_fp8 - ppl_fp16) / ppl_fp16 * 100
    time_reduction = (fp16_time - fp8_time) / fp16_time * 100

    result = {
        "model": args.model,
        "fp16": {"perplexity": round(ppl_fp16, 4), "wallclock_s": round(fp16_time, 1)},
        "fp8":  {"perplexity": round(ppl_fp8, 4),  "wallclock_s": round(fp8_time, 1)},
        "ppl_delta_pct": round(ppl_delta, 3),
        "wallclock_delta_pct": round(time_reduction, 2),
        "verdict": "PASS" if abs(ppl_delta) < 0.5 else "FAIL",
    }

    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  FP8 PPL regression : {ppl_delta:+.3f}%")
    print(f"  Wallclock reduction: {time_reduction:+.2f}%")
    print(f"  Verdict            : {result['verdict']}  (target: <0.5%)")
    print(f"  Saved              : {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
