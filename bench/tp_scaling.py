#!/usr/bin/env python3
"""
Tensor-parallel scaling efficiency sweep for AMD MI300X.

Measures throughput at TP=1/2/4/8 on a fixed model, plots scaling
efficiency vs published H100 SXM reference numbers. Reports weak/strong
scaling efficiency = (throughput_at_TPn / throughput_at_TP1) / n.

Usage:
    python3 bench/tp_scaling.py --model meta-llama/Llama-3.1-70B --tp 1,2,4,8
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch


def bench_one(llm, prompts, sampling) -> dict:
    """Single benchmark run, returns metrics dict."""
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - start

    total_out = sum(sum(len(o.token_ids) for o in out.outputs) for out in outputs)
    total_in = sum(len(out.prompt_token_ids) for out in outputs)
    return {
        "wallclock_s": round(elapsed, 2),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "throughput_tok_s": round(total_out / elapsed, 1),
        "ttft_ms": round(
            sum(out.metrics.first_token_time * 1000 for out in outputs) / len(outputs), 2
        ) if outputs and hasattr(outputs[0], "metrics") else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-70B")
    parser.add_argument("--tp", default="1,2,4,8")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", default="results/tp_scaling.json")
    args = parser.parse_args()

    tp_levels = [int(x) for x in args.tp.split(",")]
    visible_gpus = torch.cuda.device_count() if hasattr(torch.cuda, "device_count") else 0
    # ROCm reuses CUDA API — device_count returns MI300X count
    print(f"[INFO] Visible MI300X devices: {visible_gpus}")

    # Static sample prompts (avoid I/O during timing)
    prompts = [
        "Explain in technical detail how PagedAttention reduces KV-cache fragmentation "
        "in vLLM, and how this maps to HIP memory allocator behavior on gfx942. "
        "Include code snippets showing the block table lookup.",
    ] * args.num_prompts
    sampling = type("S", (), {})  # placeholder, real one below
    from vllm import SamplingParams
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, top_p=1.0)

    results = []
    base_tp1_throughput = None

    for tp in tp_levels:
        if tp > visible_gpus:
            print(f"[SKIP] TP={tp} requires {tp} GPUs but only {visible_gpus} visible.")
            continue
        print(f"\n{'=' * 60}\n  TP={tp}  ({tp}x MI300X)\n{'=' * 60}")

        from vllm import LLM
        llm = LLM(
            model=args.model,
            tensor_parallel_size=tp,
            dtype="float16",
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            trust_remote_code=True,
        )
        # Warmup
        llm.generate(prompts[:8], sampling)

        m = bench_one(llm, prompts, sampling)
        m["tp"] = tp
        m["model"] = args.model
        results.append(m)
        print(f"  throughput: {m['throughput_tok_s']} tok/s   ttft: {m['ttft_ms']} ms")

        if tp == 1:
            base_tp1_throughput = m["throughput_tok_s"]

        # Free GPU memory before next TP
        del llm
        torch.cuda.empty_cache()

    # Compute scaling efficiency
    if base_tp1_throughput:
        for m in results:
            m["scaling_efficiency_pct"] = round(
                (m["throughput_tok_s"] / base_tp1_throughput) / m["tp"] * 100, 2
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "visible_gpus": visible_gpus,
        "results": results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    # Markdown table for README
    md = [f"\n## TP scaling results — {args.model} ({datetime.now().date()})\n",
          "| TP | Throughput (tok/s) | Scaling efficiency | TTFT (ms) |",
          "|----|--------------------|--------------------|-----------|"]
    for m in results:
        md.append(f"| {m['tp']} | {m['throughput_tok_s']} | {m.get('scaling_efficiency_pct','-')}% | {m['ttft_ms']} |")
    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n")

    print("\n" + "\n".join(md))
    print(f"\n  Saved: {out}\n  Table: {md_path}")


if __name__ == "__main__":
    main()
