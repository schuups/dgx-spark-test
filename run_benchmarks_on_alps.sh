#!/bin/bash
#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --job-name=dgx-spark-test
#SBATCH --chdir=/capstor/scratch/cscs/stefschu/dgx-spark-test
#SBATCH --output=results/fp8/slurm_job-%j.out
#SBATCH --error=results/fp8/slurm_job-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

cd "$SLURM_SUBMIT_DIR"

mkdir -p results/fp8
srun --environment="${SLURM_SUBMIT_DIR}/slurm_environment.toml" \
     nvidia-smi -q -d POWER > results/fp8/slurm_power_caps.txt

srun --environment="${SLURM_SUBMIT_DIR}/slurm_environment.toml" \
     bash run_benchmarks.sh slurm
