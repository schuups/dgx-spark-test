#!/usr/bin/env bash
set -euo pipefail

# Usage: run_benchmarks.sh <context-tag>
#   e.g. ./run_benchmarks.sh slurm   (GH200 on Alps)
#        ./run_benchmarks.sh spark   (DGX Spark on enverge.ai)
# The tag is prefixed to every output filename so results from different hosts
# coexist in results/. If unset, it falls back to "slurm" when SLURM_JOB_ID is
# set, otherwise to the short hostname.
if [[ $# -ge 1 ]]; then
  CONTEXT="$1"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  CONTEXT="slurm"
else
  CONTEXT="$(hostname -s)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

cd "${SCRIPT_DIR}"

echo "Running benchmarks with context=${CONTEXT}"
echo "Results dir: ${RESULTS_DIR}"

# heads/head_dim/dtype are the canonical config; keep identical across hosts so
# overlapping shapes are directly comparable. Large-S/large-B cells may OOM on
# smaller GPUs (e.g. DGX Spark); the benchmark scripts catch OOM and continue.

python benchmark_prefill_attention.py \
  --dtype bf16 \
  --batch-sizes 1 2 4 8 16 \
  --seq-lens 1024 2048 4096 8192 16384 32768 65536 \
  --heads 64 \
  --head-dim 128 \
  --warmup 10 \
  --iters 30 \
  --csv "${RESULTS_DIR}/${CONTEXT}_prefill_attention_results.csv" \
  2>&1 | tee "${RESULTS_DIR}/${CONTEXT}_prefill_attention.log"

python benchmark_decode_attention.py \
  --dtype bf16 \
  --batch-sizes 1 2 4 8 16 \
  --kv-lens 1024 2048 4096 8192 16384 32768 65536 \
  --heads 64 \
  --head-dim 128 \
  --warmup 50 \
  --iters 200 \
  --csv "${RESULTS_DIR}/${CONTEXT}_decode_attention_results.csv" \
  2>&1 | tee "${RESULTS_DIR}/${CONTEXT}_decode_attention.log"

