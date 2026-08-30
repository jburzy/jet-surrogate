#!/bin/bash
# Submit the GPU stages as a dependency chain on ouheptmp:
#   train-tagger -> apply-tagger -> train-surrogate -> evaluate
# Usage: ./slurm/ml_chain.sh [--after <jobid[:jobid...]>] [tagger args, e.g. --epochs 30]
# Then, from anywhere:  pixi run jet-surrogate visualize
set -euo pipefail
REPO=/home/jburzyns/jet-surrogate
cd "$REPO"
AFTER=""
if [[ ${1:-} == --after ]]; then AFTER="--dependency=afterany:$2"; shift 2; fi
j1=$(sbatch --parsable $AFTER --job-name=js-tagger    slurm/gpu.sbatch train-tagger "$@")
j2=$(sbatch --parsable --dependency=afterok:$j1 --job-name=js-apply     slurm/gpu.sbatch apply-tagger --force)
j3=$(sbatch --parsable --dependency=afterok:$j2 --job-name=js-surrogate slurm/gpu.sbatch train-surrogate)
j4=$(sbatch --parsable --dependency=afterok:$j3 --job-name=js-evaluate  slurm/gpu.sbatch evaluate)
echo "tagger $j1 -> apply $j2 -> surrogate $j3 -> evaluate $j4"
