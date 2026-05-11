# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Standalone microbenchmarks measuring scaled-dot-product attention (SDPA) throughput on a single CUDA GPU. The goal is to compare **GH200** (Alps supercomputer at CSCS, where this directory lives) against **DGX Spark** (rented on enverge.ai) to reason about whether DGX Spark is a viable host for *locally hosted* LLMs powering coding agents. Memory-bandwidth differences are known on paper; what these benchmarks are meant to surface is (a) real-world latency in both attention regimes, and (b) whether DGX Spark thermally throttles under sustained load — hence the long sweeps in `run_benchmarks.sh` and the per-row `temperature_c` / `power_w` telemetry.

Two scripts cover the two attention regimes in LLM inference:

- `benchmark_prefill_attention.py` — causal self-attention over a full sequence (Q, K, V all length `seq_len`); FLOPs scale as `4·B·H·S²·D`.
- `benchmark_decode_attention.py` — single-token query against an existing KV cache of length `kv_len`; FLOPs scale as `4·B·H·KV·D`.

Both scripts share the same shape: parse args → sweep `(batch × seq/kv)` grid → for each cell, allocate random Q/K/V, warm up, time `iters` SDPA calls with `torch.cuda.Event`, query `nvidia-smi` for temp/power, append a `Result` row → write CSV. SDPA is forced through `sdpa_kernel([FLASH_ATTENTION, EFFICIENT_ATTENTION])`; the math backend is excluded so a missing flash/efficient kernel fails loudly rather than silently falling back. OOM at a `(B, S)` cell is caught and logged so the sweep continues.

## Running

`run_benchmarks.sh <context-tag>` is the generic entrypoint that runs both benchmarks back-to-back with the canonical sweep (heads=64, head_dim=128, bf16) and writes all outputs to `results/${tag}_*.{csv,log}`. The tag is host-context (e.g. `slurm`, `spark`) and lets results from different machines coexist.

- **On Alps (GH200, CSCS)**: submit via `sbatch slurm_submit.sbatch` — this calls `run_benchmarks.sh slurm` inside the NVIDIA PyTorch container declared in `slurm_environment.toml` (`nvcr.io#nvidia/pytorch:26.04-py3`). Slurm stdout/stderr land in `results/slurm_job-<jobid>.{out,err}`.
- **On DGX Spark (enverge.ai)**: run directly inside whatever PyTorch environment the host provides: `./run_benchmarks.sh spark`.

Note: `slurm_environment.toml` has `workdir = /capstor/scratch/cscs/stefsch/dgx-spark-test` (missing the `u` in `stefschu`) — fix before submitting if the job can't find the scripts.

Direct invocation (inside the container, GPU available) for a quick check:

```bash
python benchmark_prefill_attention.py --batch-sizes 1 --seq-lens 1024 --warmup 2 --iters 5
python benchmark_decode_attention.py  --batch-sizes 1 --kv-lens  1024 --warmup 2 --iters 5
```

Results are written to `prefill_attention_results.csv` / `decode_attention_results.csv` (override with `--csv`). `--gpu-index` selects the device for both `torch.cuda.set_device` and the `nvidia-smi` query — keep them in sync.

## Conventions

- No build, lint, or test infrastructure — these are single-file scripts. Keep them dependency-light (stdlib + torch).
- The two scripts are deliberately parallel in structure; when changing one (e.g., adding a CSV field or telemetry source), mirror the change in the other.
