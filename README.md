# ROCm Inference Benchmark Suite

Benchmarking LLM inference throughput on AMD Instinct MI300X GPUs using vLLM with ROCm backend.

## Overview

This project measures and compares inference performance (tokens/sec, latency, throughput) of popular LLM models running on AMD Instinct MI300X via the ROCm software stack. The goal is to validate ROCm as a production-ready alternative to CUDA for LLM serving.

## Goals

- Benchmark vLLM serving throughput on MI300X (192GB HBM3)
- Compare PagedAttention kernel performance: ROCm vs CUDA baseline
- Profile KV-cache memory management under high-concurrency workloads
- Document optimal vLLM configuration flags for MI300X

## Hardware Requirements

- AMD Instinct MI300X (tested on 1x and 8x configurations)
- ROCm 6.2+
- Ubuntu 22.04 / 24.04

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run single-model benchmark
python3 bench.py --model meta-llama/Llama-3.1-70B --num-prompts 100 --tp 1

# Run full suite (multiple models, concurrency levels)
python3 bench.py --suite full --output results/
```

## Benchmark Results

| Model           | GPUs | Concurrency | Throughput (tok/s) | Latency P50 (ms) |
|-----------------|------|-------------|-------------------|-------------------|
| Llama-3.1-8B    | 1    | 1           | 4,200             | 12                |
| Llama-3.1-8B    | 1    | 32          | 28,500            | 45                |
| Llama-3.1-70B   | 4    | 1           | 680               | 78                |
| Llama-3.1-70B   | 4    | 16          | 8,200             | 120               |

_Results on MI300X with ROCm 6.2, vLLM 0.6.x_

## Methodology

- **Workload**: ShareGPT dataset (real-world conversation traces)
- **Metrics**: Time-to-first-token (TTFT), inter-token latency (ITL), throughput (tokens/sec)
- **Config**: FP16, PagedAttention enabled, KV-cache quantization off
- **Warmup**: 10 prompts discarded before measurement

## Project Structure

```
rocm-inference-bench/
├── bench.py              # Main benchmark runner
├── config.yaml           # Model + hardware config
├── requirements.txt
├── results/              # Benchmark output (CSV + plots)
└── README.md
```

## License

MIT
