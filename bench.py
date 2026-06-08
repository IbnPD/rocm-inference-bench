#!/usr/bin/env python3
"""
ROCm Inference Benchmark Suite
Benchmark vLLM inference throughput on AMD Instinct MI300X GPUs.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("[WARN] vLLM not installed. Running in dry-run mode.")


# Sample prompts from ShareGPT-style conversations
SAMPLE_PROMPTS = [
    "Explain the difference between PagedAttention and standard attention mechanisms in transformer models.",
    "Write a Python function to compute the softmax of a tensor using HIP/ROCm kernels.",
    "What are the key architectural differences between AMD MI300X and NVIDIA H100 for LLM inference?",
    "Describe how KV-cache memory management works in vLLM and why it matters for throughput.",
    "How does the ROCm memory allocator differ from CUDA's unified memory for large language models?",
    "Explain tensor parallelism across multiple GPUs and how NCCL handles collective communications.",
    "Write a benchmark script that measures time-to-first-token for a vLLM serving endpoint.",
    "What optimizations does AMD provide for running transformer models on Instinct GPUs?",
    "Compare FP16 vs BF16 precision for LLM inference on MI300X — which gives better throughput?",
    "How does PagedAttention reduce memory fragmentation in long-context inference scenarios?",
]


def load_prompts(num_prompts: int) -> list[str]:
    """Load or generate prompts for benchmarking."""
    prompts = []
    while len(prompts) < num_prompts:
        prompts.extend(SAMPLE_PROMPTS)
    return prompts[:num_prompts]


def run_benchmark(
    model: str,
    num_prompts: int,
    tensor_parallel: int,
    max_tokens: int = 256,
    concurrency_levels: list[int] | None = None,
) -> list[dict]:
    """Run inference benchmark and collect metrics."""
    if not VLLM_AVAILABLE:
        return dry_run(model, num_prompts, tensor_parallel)

    prompts = load_prompts(num_prompts)
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, top_p=1.0)

    print(f"\n{'='*60}")
    print(f"  ROCm Inference Benchmark")
    print(f"  Model: {model}")
    print(f"  GPUs: {tensor_parallel}x MI300X")
    print(f"  Prompts: {num_prompts}")
    print(f"  Max tokens: {max_tokens}")
    print(f"{'='*60}\n")

    # Initialize vLLM engine
    print("[1/3] Loading model...")
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel,
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        trust_remote_code=True,
    )

    # Warmup
    print("[2/3] Warming up (10 prompts)...")
    warmup_prompts = load_prompts(10)
    llm.generate(warmup_prompts[:10], sampling)

    # Benchmark
    print(f"[3/3] Benchmarking {num_prompts} prompts...")
    results = []
    total_start = time.perf_counter()

    outputs = llm.generate(prompts, sampling)

    total_elapsed = time.perf_counter() - total_start

    for i, output in enumerate(outputs):
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = sum(len(o.token_ids) for o in output.outputs)
        ttft = output.metrics.first_token_time - output.metrics.arrival_time if hasattr(output, 'metrics') else 0

        results.append({
            "prompt_idx": i,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "ttft_ms": round(ttft * 1000, 2) if ttft else None,
            "throughput_tok_s": round(completion_tokens / (total_elapsed / num_prompts), 1),
        })

    total_tokens = sum(r["completion_tokens"] for r in results)
    avg_throughput = total_tokens / total_elapsed

    summary = {
        "model": model,
        "gpus": tensor_parallel,
        "num_prompts": num_prompts,
        "total_time_s": round(total_elapsed, 2),
        "total_output_tokens": total_tokens,
        "avg_throughput_tok_s": round(avg_throughput, 1),
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"  Total time: {summary['total_time_s']}s")
    print(f"  Output tokens: {summary['total_output_tokens']}")
    print(f"  Throughput: {summary['avg_throughput_tok_s']} tok/s")
    print(f"{'='*60}\n")

    return results, summary


def dry_run(model, num_prompts, tp):
    """Simulate benchmark when vLLM is not installed."""
    print(f"[DRY RUN] Model={model}, Prompts={num_prompts}, TP={tp}")
    print("[DRY RUN] Install vLLM + ROCm to run actual benchmarks.")
    print("[DRY RUN] pip install vllm")
    return [], {"model": model, "gpus": tp, "status": "dry_run"}


def save_results(results: list[dict], summary: dict, output_dir: str):
    """Save benchmark results to CSV and JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(output_dir, f"bench_{ts}.csv")
    json_path = os.path.join(output_dir, f"bench_{ts}.json")

    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV: {csv_path}")

    with open(json_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="ROCm Inference Benchmark Suite")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="HuggingFace model name")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of prompts to benchmark")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens per prompt")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--suite", choices=["quick", "full"], help="Run preset benchmark suite")

    args = parser.parse_args()

    if args.suite == "quick":
        args.model = "meta-llama/Llama-3.1-8B"
        args.num_prompts = 50
        args.tp = 1
    elif args.suite == "full":
        # Run multiple configs
        configs = [
            ("meta-llama/Llama-3.1-8B", 100, 1),
            ("meta-llama/Llama-3.1-8B", 100, 2),
            ("meta-llama/Llama-3.1-70B", 50, 4),
        ]
        all_results = []
        for model, n, tp in configs:
            results, summary = run_benchmark(model, n, tp, args.max_tokens)
            save_results(results, summary, args.output)
            all_results.append(summary)
        print("\n=== Full Suite Complete ===")
        for s in all_results:
            print(f"  {s.get('model','?')} | {s.get('gpus','?')} GPU | {s.get('avg_throughput_tok_s','?')} tok/s")
        return

    results, summary = run_benchmark(args.model, args.num_prompts, args.tp, args.max_tokens)
    save_results(results, summary, args.output)


if __name__ == "__main__":
    main()
