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

BATCH_SIZES="1 2 4 8"
SEQ_LENS="1024 4096 16384 65536"
KV_LENS="1024 4096 16384 65536"

echo ""
echo "=== prefill | dtype=fp16 ==="
python -u benchmark_prefill_attention.py \
  --dtype fp16 \
  --batch-sizes ${BATCH_SIZES} \
  --seq-lens ${SEQ_LENS} \
  --heads 64 \
  --head-dim 128 \
  --warmup 5 \
  --iters 15 \
  --csv "${RESULTS_DIR}/${CONTEXT}_fp16_prefill_attention_results.csv" \
  2>&1 | tee "${RESULTS_DIR}/${CONTEXT}_fp16_prefill_attention.log"

echo ""
echo "=== decode | dtype=fp16 ==="
python -u benchmark_decode_attention.py \
  --dtype fp16 \
  --batch-sizes ${BATCH_SIZES} \
  --kv-lens ${KV_LENS} \
  --heads 64 \
  --head-dim 128 \
  --warmup 20 \
  --iters 50 \
  --csv "${RESULTS_DIR}/${CONTEXT}_fp16_decode_attention_results.csv" \
  2>&1 | tee "${RESULTS_DIR}/${CONTEXT}_fp16_decode_attention.log"
