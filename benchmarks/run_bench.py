#!/usr/bin/env python3
"""Benchmark prefill + decode throughput and peak memory for FP16 vs INT8.

Usage:
    python benchmarks/run_bench.py              # auto-detect device (MPS/CUDA/CPU)
    python benchmarks/run_bench.py --device cpu  # force CPU

Prints a markdown table suitable for pasting into the README.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

# Add project root to path so we can import lolama
sys.path.insert(0, str(Path(__file__).parent.parent))

from lolama.data import load_model, load_tokenizer
from lolama.model import TextGenerator, GenerationConfig, quantize_model_int8, get_model_size_mb
from lolama.utils.device import resolve_device


def _sync(device: str) -> None:
    """Synchronize device to get accurate timings."""
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _peak_memory_mb(device: str) -> float:
    """Get peak memory usage in MB."""
    if device == "mps":
        # MPS doesn't expose peak memory; use driver-allocated as proxy
        return torch.mps.driver_allocated_memory() / 1e6
    elif device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0  # CPU — not tracked


def _reset_memory_stats(device: str) -> None:
    """Reset peak memory tracking."""
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def benchmark_model(
    model,
    tokenizer,
    device: str,
    prompt: str,
    prefill_tokens: int,
    decode_tokens: int,
    warmup_runs: int = 2,
    bench_runs: int = 5,
) -> dict[str, float]:
    """Benchmark a single model configuration.

    Returns dict with: prefill_tok_s, decode_tok_s, peak_mem_mb, model_size_mb
    """
    gen = TextGenerator(model)

    # Build input: use the prompt, pad/truncate to exactly prefill_tokens
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if ids.shape[1] < prefill_tokens:
        # Repeat prompt tokens to reach desired prefill length
        repeats = (prefill_tokens // ids.shape[1]) + 1
        ids = ids.repeat(1, repeats)[:, :prefill_tokens]
    else:
        ids = ids[:, :prefill_tokens]

    config = GenerationConfig.greedy(max_new_tokens=decode_tokens)

    # Warmup
    for _ in range(warmup_runs):
        gen.generate(ids, config)
        _sync(device)

    # Benchmark
    _reset_memory_stats(device)
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()

    prefill_times = []
    decode_times = []

    for _ in range(bench_runs):
        model.eval()
        kv_caches = model.create_kv_caches(batch_size=1, max_seq_len=prefill_tokens + decode_tokens)

        # Prefill
        _sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(ids, kv_caches=kv_caches)
        _sync(device)
        prefill_times.append(time.perf_counter() - t0)

        # Decode
        _sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            next_tok = logits[:, -1:, :].argmax(dim=-1)
            for _ in range(decode_tokens - 1):
                logits = model(next_tok, kv_caches=kv_caches)
                next_tok = logits[:, -1:, :].argmax(dim=-1)
        _sync(device)
        decode_times.append(time.perf_counter() - t0)

    avg_prefill = sum(prefill_times) / len(prefill_times)
    avg_decode = sum(decode_times) / len(decode_times)

    return {
        "prefill_tok_s": prefill_tokens / avg_prefill,
        "decode_tok_s": decode_tokens / avg_decode,
        "peak_mem_mb": _peak_memory_mb(device),
        "model_size_mb": get_model_size_mb(model),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark lolama inference")
    parser.add_argument("--model", default="tinyllama", help="Model alias (default: tinyllama)")
    parser.add_argument("--device", default=None, help="Device: cpu, mps, cuda (default: auto)")
    parser.add_argument("--prefill", type=int, default=128, help="Prefill token count")
    parser.add_argument("--decode", type=int, default=64, help="Decode token count")
    parser.add_argument("--runs", type=int, default=5, help="Benchmark iterations")
    args = parser.parse_args()

    device = args.device or resolve_device()
    dtype = torch.float16 if device != "cpu" else torch.float32
    prompt = "The meaning of life is a question that has been asked by philosophers for centuries."

    # Print hardware info
    import platform
    import subprocess
    hw = platform.processor() or platform.machine()
    try:
        hw = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        pass
    ram_gb = 0
    try:
        import os
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except Exception:
        pass
    if device.startswith("cuda") and torch.cuda.is_available():
        hw = torch.cuda.get_device_name(0)
        ram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print(f"Hardware: {hw} / {ram_gb:.0f} GB")
    print(f"Device: {device} | dtype: {dtype} | model: {args.model}")
    print(f"Prefill: {args.prefill} tokens | Decode: {args.decode} tokens | Runs: {args.runs}")
    print()

    # ── FP16 ──
    print("Loading FP16 model...")
    model = load_model(args.model, device=device, dtype=dtype)
    tokenizer = load_tokenizer(args.model)

    fp16_results = benchmark_model(
        model, tokenizer, device, prompt,
        prefill_tokens=args.prefill,
        decode_tokens=args.decode,
        bench_runs=args.runs,
    )

    # ── INT8 ──
    print("Quantizing to INT8...")
    quantize_model_int8(model, skip_layers=["lm_head", "embed_tokens"])

    int8_results = benchmark_model(
        model, tokenizer, device, prompt,
        prefill_tokens=args.prefill,
        decode_tokens=args.decode,
        bench_runs=args.runs,
    )

    # ── Print results ──
    print()
    print(f"## Benchmark: {args.model} on {device} (prefill={args.prefill}, decode={args.decode})")
    print()
    print("| Config | Prefill (tok/s) | Decode (tok/s) | Model Size (MB) | Peak Memory (MB) |")
    print("|--------|----------------:|---------------:|----------------:|-----------------:|")
    for label, r in [("FP16", fp16_results), ("INT8", int8_results)]:
        mem_str = f'{r["peak_mem_mb"]:.0f}' if r["peak_mem_mb"] > 0 else "n/a"
        print(
            f'| {label:6s} | {r["prefill_tok_s"]:>15,.0f} | {r["decode_tok_s"]:>14,.0f} '
            f'| {r["model_size_mb"]:>15,.0f} | {mem_str:>16s} |'
        )

    # Speedup / savings
    print()
    size_reduction = (1 - int8_results["model_size_mb"] / fp16_results["model_size_mb"]) * 100
    decode_ratio = int8_results["decode_tok_s"] / fp16_results["decode_tok_s"]
    print(f"INT8 model size: {size_reduction:.0f}% smaller")
    print(f"INT8 decode throughput: {decode_ratio:.2f}x vs FP16")


if __name__ == "__main__":
    main()
