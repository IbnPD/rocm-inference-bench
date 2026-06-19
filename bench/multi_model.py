#!/usr/bin/env python3
"""
Multi-model benchmark suite covering Llama 3.1 / Mistral / Qwen 2.5
at 7B/13B/70B scales on a single MI300X.

Run: python3 bench/multi_model.py
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams

MODELS_7B_13B = [
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-v0.3",
    "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B-Instruct",
]
MODELS_70B_PLUS = [
    "meta-llama/Llama-3.1-70B-Instruct",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "Qwen/Qwen2.5-72B-Instruct",
]

CONCURRENCY = [1, 8, 32, 128]
PROMPTS_PER_RUN = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-class", choices=["small", "large", "all"], dest="model_class", default="small")
    parser.add_argument("--out", default="results/multi_model.json")
    args = parser.parse_args()

    models = MODELS_7B_13B if args.model_class != "large" else MODELS_70B_PLUS
    if args.model_class == "all":
        models = MODELS_7B_13B + MODELS_70B_PLUS

    prompts = [
        "Describe a distributed training architecture for trillion-parameter models, "
        "covering tensor parallelism, pipeline parallelism, ZeRO-3 sharding, "
        "and gradient compression strategies. Include the trade-offs of each.",
    ] * PROMPTS_PER_RUN

    sampling = SamplingParams(temperature=0.0, max_tokens=256, top_p=1.0)
    results = []

    for model in models:
        tp = 1 if model in MODELS_7B_13B else 4
        print(f"\n{'=' * 60}\n  {model}  TP={tp}\n{'=' * 60}")
        llm = LLM(
            model=model,
            tensor_parallel_size=tp,
            dtype="float16",
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            trust_remote_code=True,
        )
        llm.generate(prompts[:8], sampling)  # warmup

        for c in CONCURRENCY:
            t0 = time.perf_counter()
            outputs = llm.generate(prompts[:c], sampling)
            elapsed = time.perf_counter() - t0
            total_out = sum(sum(len(o.token_ids) for o in out.outputs) for out in outputs)
            results.append({
                "model": model,
                "concurrency": c,
                "tp": tp,
                "wallclock_s": round(elapsed, 2),
                "total_output_tokens": total_out,
                "throughput_tok_s": round(total_out / elapsed, 1),
                "tokens_per_request": round(total_out / c, 1),
            })
            print(f"  c={c:3d}  {total_out} tokens  {total_out/elapsed:.0f} tok/s")

        del llm
        import torch
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
