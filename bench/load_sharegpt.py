#!/usr/bin/env python3
"""
Download + cache ShareGPT-style dataset for benchmarking.

Uses HuggingFaceH4/ultrachat_200k as a ShareGPT-style replacement
(more permissively licensed than the original ShareGPT dump).

Run: python3 bench/load_sharegpt.py --num 1000 --out bench/data/ultrachat_1k.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--out", default="bench/data/ultrachat_1k.jsonl")
    args = parser.parse_args()

    print(f"[INFO] Loading ultrachat_200k ({args.num} samples)...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{args.num}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for row in ds:
            f.write(json.dumps({"messages": row["messages"]}) + "\n")

    print(f"[OK] {args.num} samples → {out}")


if __name__ == "__main__":
    main()
