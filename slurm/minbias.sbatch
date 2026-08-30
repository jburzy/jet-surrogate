#!/bin/bash
# One array task = 25k minimum-bias events as HepMC2 (input of hepmc2pileup).
# Submit:  sbatch --array=1-20 slurm/minbias.sbatch
# Then:    sbatch --dependency=afterok:<jobid> slurm/minbias_convert.sbatch
#SBATCH --partition=sooner_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --job-name=gen-minbias
#SBATCH --output=/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/%x_%A_%a.out
#SBATCH --error=/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/%x_%A_%a.err
set -euo pipefail
REPO=/home/jburzyns/jet-surrogate
cd "$REPO"
export WORKDIR=$REPO PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache PYTHONUNBUFFERED=1
srun --chdir="$REPO" pixi run --manifest-path "$REPO/pixi.toml" jet-surrogate generate \
    --card cards/pythia/minbias.cmnd --format hepmc --hepmc2 \
    --nevents "${NEVENTS:-25000}" --seed "$SLURM_ARRAY_TASK_ID" --out "$REPO/data/minbias"
