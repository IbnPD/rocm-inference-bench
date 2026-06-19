# ROCm Inference Benchmark Suite

Benchmarking LLM inference throughput on AMD Instinct MI300X GPUs using vLLM
with the ROCm backend. The goal is independent, peer-reviewable gfx942 numbers
for the ROCm ecosystem, focused on tensor-parallel scaling and FP8 quantization
on the AMD Developer Cloud (MI300X, 192 GB HBM3 per device).

## Why this exists

Most public ROCm MI300X inference numbers come from vendor blog posts
(AMD, vLLM project) on single configs. This repo publishes **reproducible
scripts + raw configs + scaling sweeps** so anyone can re-run and verify,
plus ROCm-specific gotchas (gfx942 hipBLASLt, RCCL all-reduce, FP8
`e4m3fnuz` quantization edge cases).

## Hardware requirements

- 1x-8x AMD Instinct MI300X (192 GB HBM3 each)
- ROCm 6.2+
- Ubuntu 22.04 or 24.04

## Quick start

**1. Build the container:**

```bash
docker build -f Dockerfile.rocm -t rocm-vllm:mi300x .
```

**2. Verify ROCm is visible:**

```bash
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE \
  rocm-vllm:mi300x /opt/rocm/bin/rocminfo | grep -A2 "Marketing Name"
```

**3. Run a quick benchmark:**

```bash
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  rocm-vllm:mi300x python3 bench.py --suite quick
```

## Project structure

```
rocm-inference-bench/
├── bench.py                      # Single-model benchmark runner
├── config.yaml                   # Default model + hardware config
├── Dockerfile.rocm               # ROCm 6.2 + vLLM 0.6.x container
├── bench/
│   ├── fp8_validate.py           # FP8 vs FP16 perplexity + throughput
│   ├── tp_scaling.py             # TP=1/2/4/8 scaling sweep
│   ├── multi_model.py            # Multi-model concurrency sweep
│   ├── load_sharegpt.py          # Dataset loader (ultrachat_200k)
│   └── data/                     # Cached datasets (gitignored)
├── results/                      # CSV + JSON output (gitignored)
├── requirements.txt
└── README.md
```

## Benchmark suite

### 1. Single-model baseline — `bench.py`

```bash
python3 bench.py --model meta-llama/Llama-3.1-8B --tp 1 --num-prompts 100
```

Outputs:
- `results/bench_<timestamp>.csv` — per-prompt metrics
- `results/bench_<timestamp>.json` — summary + raw rows

### 2. Tensor-parallel scaling — `bench/tp_scaling.py`

```bash
python3 bench/tp_scaling.py --model meta-llama/Llama-3.1-70B --tp 1,2,4,8
```

Measures throughput + scaling efficiency at TP=1/2/4/8 on a single model,
writes a Markdown table to `results/tp_scaling.md` ready for review.

### 3. FP8 quantization validation — `bench/fp8_validate.py`

```bash
python3 bench/fp8_validate.py --model meta-llama/Llama-3.1-8B
```

Validates `e4m3fnuz` FP8 quantization on gfx942:
- WikiText-103 sliding-window perplexity (FP16 vs FP8)
- Wallclock reduction at FP8
- Pass/fail verdict vs target `<0.5%` PPL regression

### 4. Multi-model concurrency sweep — `bench/multi_model.py`

```bash
python3 bench/multi_model.py --class small    # 7-13B models
python3 bench/multi_model.py --class large    # 70B+ models
python3 bench/multi_model.py --class all
```

Sweeps 7B-72B models at concurrency `[1, 8, 32, 128]` and writes JSON.

## Methodology

- **Workload**: HuggingFaceH4/ultrachat_200k (ShareGPT-style conversation
  traces, more permissively licensed than the original ShareGPT dump).
- **Metrics**:
  - Time-to-first-token (TTFT) p50/p95
  - Inter-token latency (ITL) mean/p99
  - Throughput (output tokens/sec)
  - KV-cache utilization under sustained load
- **Config**:
  - FP16 baseline + FP8 `e4m3fnuz` quantized variant
  - PagedAttention enabled, KV-cache quantization off by default
  - `max_model_len=4096`, `gpu_memory_utilization=0.90`
- **Warmup**: 8-10 prompts discarded before each measurement
- **RCCL**: tuned via `RCCL_ENABLE_GDR=1`, `RCCL_BUFFSIZE=4194304`

## ROCm-specific notes

- vLLM ROCm wheels ship at https://download.pytorch.org/whl/rocm6.2 — pin
  the torch version or the wheel resolution breaks.
- `gfx942` FP8 `e4m3fnuz` is the correct dtype on MI300X (not `e4m3fn`).
- RCCL all-reduce can be 15-30% slower than NVLink at TP=8; we're
  characterizing this in `tp_scaling.py`.
- `hipBLASLt` is required for FP8 GEMM paths on MI300X; verify via
  `rocm-smi --showproductname` and `rocminfo | grep gfx`.

## Reproducibility

All configs, prompt sets, and `vllm serve` flags are committed to
`config.yaml` and the script argument lists. To reproduce a number:

```bash
git checkout <commit-sha>
pip install -r requirements.txt
python3 bench/tp_scaling.py --model meta-llama/Llama-3.1-70B --tp 4
```

## License

MIT — see `LICENSE`.
