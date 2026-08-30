#!/bin/bash
# Run one `jet-surrogate` command in the CUDA environment on the ouheptmp GPU node.
#   sbatch slurm/gpu.sbatch train-tagger --epochs 30
#   sbatch slurm/gpu.sbatch apply-tagger
#   sbatch slurm/gpu.sbatch train-surrogate
#   sbatch slurm/gpu.sbatch evaluate
#SBATCH --job-name=js-gpu
#SBATCH --partition=ouheptmp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/%x_%j.out
#SBATCH --error=/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/%x_%j.err
set -euo pipefail
cd /home/jburzyns/jet-surrogate
export WORKDIR=/home/jburzyns/jet-surrogate   # the container wrapper does `cd $WORKDIR` before each srun
export PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache
export CONDA_OVERRIDE_CUDA=12.4
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "host=$(hostname) job=$SLURM_JOB_ID task=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date
srun --chdir=/home/jburzyns/jet-surrogate pixi run --manifest-path /home/jburzyns/jet-surrogate/pixi.toml -e gpu jet-surrogate "$@"
date; echo done
