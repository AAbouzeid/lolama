"""Correctness test + benchmark for fused quantized matmul paths.

Usage:
    python -m lolama.metal.benchmark
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def naive_dequant_matmul(
    x: torch.Tensor, w_int8: torch.Tensor, scales: torch.Tensor,
) -> torch.Tensor:
    """Naive dequant + matmul (the baseline)."""
    weight_f32 = w_int8.float() * scales.float().unsqueeze(1)
    weight = weight_f32.to(x.dtype)
    return F.linear(x, weight)


def int_mm_matmul(
    x: torch.Tensor, w_int8: torch.Tensor, scales: torch.Tensor,
) -> torch.Tensor:
    """torch._int_mm W8A8 path."""
    x_scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    x_int8 = (x / x_scale).round().clamp(-127, 127).to(torch.int8)
    out_int32 = torch._int_mm(x_int8, w_int8.t())
    return (out_int32.float() * x_scale * scales.float().unsqueeze(0)).to(x.dtype)


def benchmark_fn(fn, *args, warmup: int = 5, iters: int = 50, device: str = "cpu"):
    """Time a function, returning median ms."""
    for _ in range(warmup):
        fn(*args)
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        if device == "mps":
            torch.mps.synchronize()
        elif device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def make_test_data(M: int, N: int, K: int, device: str, dtype: torch.dtype = torch.float16):
    """Create random test matrices."""
    x = torch.randn(M, K, device=device, dtype=dtype)
    weight_fp = torch.randn(N, K, device=device, dtype=dtype)
    # Quantize weights
    absmax = weight_fp.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    scale = absmax / 127.0
    w_int8 = (weight_fp / scale).round().clamp(-127, 127).to(torch.int8)
    scales = scale.squeeze(1).to(dtype)
    return x, w_int8, scales


def test_correctness(device: str, dtype: torch.dtype = torch.float16):
    """Test all available paths against naive baseline."""
    print(f"\n{'='*60}")
    print(f"Correctness test on {device}")
    print(f"{'='*60}")

    M, N, K = 32, 128, 256
    x, w_int8, scales = make_test_data(M, N, K, device, dtype)

    # Baseline
    ref = naive_dequant_matmul(x, w_int8, scales)

    # Metal path
    if device == "mps":
        try:
            from lolama.metal import dequant_matmul, is_available
            if is_available():
                out_metal = dequant_matmul(x, w_int8, scales)
                diff = (out_metal - ref).abs().max().item()
                ok = diff < 0.05  # fp16 tolerance
                status = "PASS" if ok else "FAIL"
                print(f"  Metal fused:   max_diff={diff:.6f}  [{status}]")
            else:
                print("  Metal fused:   [SKIP - not available]")
        except Exception as e:
            print(f"  Metal fused:   [ERROR - {e}]")

    # _int_mm path (looser tolerance due to W8A8 double quantization)
    if device.startswith("cuda") and hasattr(torch, "_int_mm"):
        try:
            out_intmm = int_mm_matmul(x, w_int8, scales)
            diff = (out_intmm - ref).abs().max().item()
            ok = diff < 1.0  # looser tolerance for W8A8
            status = "PASS" if ok else "FAIL"
            print(f"  torch._int_mm: max_diff={diff:.6f}  [{status}]")
        except Exception as e:
            print(f"  torch._int_mm: [ERROR - {e}]")

    # Batched input test
    print("\n  Batched input test (3D):")
    x_batched = torch.randn(2, M, K, device=device, dtype=dtype)
    ref_batched = naive_dequant_matmul(
        x_batched.reshape(-1, K), w_int8, scales,
    ).reshape(2, M, N)

    if device == "mps":
        try:
            from lolama.metal import dequant_matmul, is_available
            if is_available():
                out = dequant_matmul(x_batched, w_int8, scales)
                diff = (out - ref_batched).abs().max().item()
                ok = diff < 0.05
                status = "PASS" if ok else "FAIL"
                print(f"    Metal batched: max_diff={diff:.6f}  [{status}]")
        except Exception as e:
            print(f"    Metal batched: [ERROR - {e}]")


def run_benchmarks(device: str, dtype: torch.dtype = torch.float16):
    """Benchmark all available paths."""
    print(f"\n{'='*60}")
    print(f"Benchmark on {device}")
    print(f"{'='*60}")

    configs = [
        (1, 4096, 4096, "Single token (1, 4096, 4096)"),
        (32, 4096, 4096, "Small batch  (32, 4096, 4096)"),
        (128, 4096, 4096, "Medium batch (128, 4096, 4096)"),
    ]

    for M, N, K, label in configs:
        print(f"\n  {label}:")
        x, w_int8, scales = make_test_data(M, N, K, device, dtype)

        # Naive dequant
        t_naive = benchmark_fn(naive_dequant_matmul, x, w_int8, scales, device=device)
        print(f"    Naive dequant:   {t_naive:.3f} ms")

        # Pre-cached fp16 (best case for memory)
        weight_fp16 = (w_int8.float() * scales.float().unsqueeze(1)).to(dtype)
        t_cached = benchmark_fn(F.linear, x, weight_fp16, device=device)
        print(f"    Cached fp16:     {t_cached:.3f} ms  ({t_naive/t_cached:.2f}x vs naive)")

        # Metal
        if device == "mps":
            try:
                from lolama.metal import dequant_matmul, is_available
                if is_available():
                    t_metal = benchmark_fn(dequant_matmul, x, w_int8, scales, device=device)
                    print(f"    Metal fused:     {t_metal:.3f} ms  ({t_naive/t_metal:.2f}x vs naive)")
            except Exception as e:
                print(f"    Metal fused:     [ERROR - {e}]")

        # _int_mm
        if device.startswith("cuda") and hasattr(torch, "_int_mm"):
            try:
                t_intmm = benchmark_fn(int_mm_matmul, x, w_int8, scales, device=device)
                print(f"    torch._int_mm:   {t_intmm:.3f} ms  ({t_naive/t_intmm:.2f}x vs naive)")
            except Exception as e:
                print(f"    torch._int_mm:   [ERROR - {e}]")


def main():
    print("Fused Quantized Matmul — Correctness & Benchmark")
    print(f"PyTorch {torch.__version__}")

    devices: list[str] = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    print(f"Available devices: {devices}")

    for device in devices:
        # CPU doesn't have accelerated paths, only test naive
        if device == "cpu":
            print(f"\n{'='*60}")
            print("CPU: naive path only (baseline)")
            print(f"{'='*60}")
            x, w_int8, scales = make_test_data(32, 4096, 4096, device)
            t = benchmark_fn(naive_dequant_matmul, x, w_int8, scales, device=device)
            print(f"  Naive dequant (32, 4096, 4096): {t:.3f} ms")
            continue

        test_correctness(device)
        run_benchmarks(device)


if __name__ == "__main__":
    main()
