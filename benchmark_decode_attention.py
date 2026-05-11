#!/usr/bin/env python3

import argparse
import csv
import subprocess
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass
class Result:
    gpu: str
    batch: int
    kv_len: int
    heads: int
    head_dim: int
    dtype: str
    latency_ms: float
    tokens_per_s: float
    approx_tflops: float
    max_memory_gb: float
    temperature_c: float
    power_w: float


def query_gpu_telemetry(gpu_index: int):
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    temp, power = out.split(",")
    return float(temp.strip()), float(power.strip())


def parse_dtype(name: str):
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def decode_attention_flops(batch, heads, kv_len, head_dim):
    # One-token decode:
    # QK^T:     2 * B * H * 1 * KV * D
    # Attn @ V: 2 * B * H * 1 * KV * D
    return 4.0 * batch * heads * kv_len * head_dim


@torch.inference_mode()
def run_one(
    gpu_name,
    gpu_index,
    batch,
    kv_len,
    heads,
    head_dim,
    dtype,
    warmup,
    iters,
):
    device = "cuda"

    # Decode phase: one new query token attends over an existing KV cache.
    q = torch.randn(batch, heads, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, kv_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, kv_len, head_dim, device=device, dtype=dtype)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        for _ in range(warmup):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        end.record()

        torch.cuda.synchronize()

    temperature_c, power_w = query_gpu_telemetry(gpu_index)

    latency_ms = start.elapsed_time(end) / iters
    tokens_per_s = batch / (latency_ms / 1e3)

    flops = decode_attention_flops(batch, heads, kv_len, head_dim)
    approx_tflops = flops / (latency_ms / 1e3) / 1e12

    max_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    return Result(
        gpu=gpu_name,
        batch=batch,
        kv_len=kv_len,
        heads=heads,
        head_dim=head_dim,
        dtype=str(dtype).replace("torch.", ""),
        latency_ms=latency_ms,
        tokens_per_s=tokens_per_s,
        approx_tflops=approx_tflops,
        max_memory_gb=max_memory_gb,
        temperature_c=temperature_c,
        power_w=power_w,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--kv-lens", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--csv", type=str, default="decode_attention_results.csv")
    parser.add_argument("--gpu-index", type=int, default=0)

    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU not available"

    torch.cuda.set_device(args.gpu_index)

    dtype = parse_dtype(args.dtype)
    gpu_name = torch.cuda.get_device_name(args.gpu_index)

    print("GPU:", gpu_name)
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print()

    results = []

    for batch in args.batch_sizes:
        for kv_len in args.kv_lens:
            try:
                result = run_one(
                    gpu_name=gpu_name,
                    gpu_index=args.gpu_index,
                    batch=batch,
                    kv_len=kv_len,
                    heads=args.heads,
                    head_dim=args.head_dim,
                    dtype=dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                )

                results.append(result)

                print(
                    f"B={result.batch:2d} "
                    f"KV={result.kv_len:7d} "
                    f"H={result.heads:3d} "
                    f"D={result.head_dim:3d} "
                    f"lat={result.latency_ms:9.4f} ms "
                    f"tok/s={result.tokens_per_s:12.1f} "
                    f"TFLOP/s≈{result.approx_tflops:9.2f} "
                    f"mem={result.max_memory_gb:7.2f} GB "
                    f"temp={result.temperature_c:.0f}C "
                    f"power={result.power_w:.1f}W"
                )

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"B={batch} KV={kv_len}: OOM")

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=Result.__dataclass_fields__.keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)

    print()
    print(f"Wrote results to {args.csv}")


if __name__ == "__main__":
    main()

