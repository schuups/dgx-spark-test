#!/usr/bin/env bash
# Run the benchmark suite on DGX Spark using the same NGC image as the SLURM job.
#
# Prerequisites:
#   NVIDIA Container Toolkit installed and docker configured with --gpus support
#
# Usage:
#   ./run_benchmarks_on_dgxspark.sh
set -euo pipefail

IMAGE="nvcr.io/nvidia/pytorch:26.04-py3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${SCRIPT_DIR}/results"
nvidia-smi -q -d POWER > "${SCRIPT_DIR}/results/spark_power_caps.txt"

docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${SCRIPT_DIR}:/workspace" \
  -v "${HOME}/.cache/pip:/root/.cache/pip" \
  -w /workspace \
  "${IMAGE}" \
  bash run_benchmarks.sh spark
