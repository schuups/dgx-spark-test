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

cd "${SCRIPT_DIR}"

echo "Running benchmarks with context=${CONTEXT}"

# Install the right Flash Attention build for this GPU.
# FA4 (sm100+, Blackwell) handles FP8; FA3 (sm90, Hopper) also handles FP8.
# pip caches built wheels in ~/.cache/pip so subsequent runs skip recompilation.
GPU_SM=$(python3 -c "import torch; print(torch.cuda.get_device_capability()[0])")
if [[ "${GPU_SM}" -ge 10 ]]; then
  echo "Blackwell GPU (sm${GPU_SM}0): building Flash Attention 4 (SM121 fork)..."
#  rm -rf \
#    /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl \
#    /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl-*.dist-info
  export TORCH_CUDA_ARCH_LIST="12.1"
  export FLASH_ATTN_CUDA_ARCHS="121"
  export MAX_JOBS=16
  export NVCC_THREADS=4
  export CUDA_HOME=/usr/local/cuda
  export PIP_NO_CACHE_DIR=0
  export NINJA_STATUS="[%f/%t %p | %es] "
  pip wheel --no-build-isolation \
    -w /workspace/flash-wheels \
    "git+https://github.com/askliar/flash-attention.git@add-sm121-support"
  pip install /workspace/flash-wheels/flash_attn*.whl
else
  echo "Hopper GPU (sm${GPU_SM}0): installing Flash Attention 3 (pre-built wheel)..."
  python -m pip install --no-cache-dir --no-deps --force-reinstall \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.11/flash_attn_3-3.0.0%2Bcu130torch2.11gite2743ab-cp39-abi3-linux_aarch64.whl"
fi
echo "Flash Attention ready."

BATCH_SIZES="1 2 4 8"
SEQ_LENS="1024 4096 16384 65536"
KV_LENS="1024 4096 16384 65536"

# Results are organised by dtype: results/{dtype}/
for DTYPE in fp8; do
  RESULTS_DIR="${SCRIPT_DIR}/results/${DTYPE}"
  mkdir -p "${RESULTS_DIR}"

  echo ""
  echo "=== prefill | dtype=${DTYPE} ==="
  python -u benchmark_prefill_attention.py \
    --dtype "${DTYPE}" \
    --batch-sizes ${BATCH_SIZES} \
    --seq-lens ${SEQ_LENS} \
    --heads 64 \
    --head-dim 128 \
    --warmup 5 \
    --iters 15 \
    --csv "${RESULTS_DIR}/${CONTEXT}_${DTYPE}_prefill_attention_results.csv" \
    2>&1 | tee "${RESULTS_DIR}/${CONTEXT}_${DTYPE}_prefill_attention.log"

  echo ""
  echo "=== decode | dtype=${DTYPE} ==="
  python -u benchmark_decode_attention.py \
    --dtype "${DTYPE}" \
    --batch-sizes ${BATCH_SIZES} \
    --kv-lens ${KV_LENS} \
    --heads 64 \
    --head-dim 128 \
    --warmup 20 \
    --iters 50 \
    --csv "${RESULTS_DIR}/${CONTEXT}_${DTYPE}_decode_attention_results.csv" \
    2>&1 | tee "${RESULTS_DIR}/${CONTEXT}_${DTYPE}_decode_attention.log"
done
