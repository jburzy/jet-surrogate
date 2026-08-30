#!/bin/bash
# Re-skim every Delphes file (one array task per ROOT file): ./slurm/reskim.sh
#SBATCH --partition=sooner_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/%x_%A_%a.out
#SBATCH --error=/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/%x_%A_%a.err
set -euo pipefail
REPO=/home/jburzyns/jet-surrogate
export WORKDIR=$REPO PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache PYTHONUNBUFFERED=1
LIST=${LIST:?set LIST}
ROOT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[[ -n $ROOT ]] || { echo "no list line $SLURM_ARRAY_TASK_ID"; exit 1; }
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID root=$ROOT"; date
srun --chdir="$REPO" pixi run --manifest-path "$REPO/pixi.toml" jet-surrogate skim "$ROOT" --out "$REPO/data/skim"
date; echo done
