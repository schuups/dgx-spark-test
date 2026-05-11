#!/bin/bash
#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --job-name=dgx-spark-test
#SBATCH --chdir=/capstor/scratch/cscs/stefschu/dgx-spark-test
#SBATCH --output=results/slurm_job-%j.out
#SBATCH --error=results/slurm_job-%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

cd "$SLURM_SUBMIT_DIR"

srun --environment="${SLURM_SUBMIT_DIR}/slurm_environment.toml" \
     bash run_benchmarks.sh slurm
