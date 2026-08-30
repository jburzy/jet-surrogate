#!/bin/bash
# One array task = one 10k-event Delphes ROOT file + its HDF5 skim.
# Submit through ./slurm/submit.sh (sets the array range from the manifest).
#SBATCH --partition=sooner_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/%x_%A_%a.out
#SBATCH --error=/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/%x_%A_%a.err
set -euo pipefail
REPO=/home/jburzyns/jet-surrogate
cd "$REPO"
export WORKDIR=/home/jburzyns/jet-surrogate   # the container wrapper does `cd $WORKDIR` before each srun
export PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache
export PYTHONUNBUFFERED=1
NEVENTS=${NEVENTS:-10000}
MANIFEST=${MANIFEST:?set MANIFEST}
# manifest line: <sample> <mpid|-> <ctau|-> <seed> [extra generate args, e.g. --lambda 25 or --nflav 2]
read -r SAMPLE MPID CTAU SEED EXTRA < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[[ -n ${SAMPLE:-} ]] || { echo "no manifest line $SLURM_ARRAY_TASK_ID"; exit 1; }
ARGS=(--sample "$SAMPLE" --nevents "$NEVENTS" --seed "$SEED" --out "$REPO/data/delphes" --quiet)
[[ $CTAU != - ]] && ARGS+=(--ctau "$CTAU" --mpid "$MPID")
[[ -n ${EXTRA:-} && $EXTRA != - ]] && ARGS+=($EXTRA)
echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID sample=$SAMPLE mpid=$MPID ctau=$CTAU seed=$SEED extra=${EXTRA:-}"
date
GENLOG=$(mktemp -p "$REPO/data/delphes" .gen.XXXXXX)
srun --chdir="$REPO" pixi run --manifest-path "$REPO/pixi.toml" jet-surrogate generate "${ARGS[@]}" | tee "$GENLOG"
ROOT=$(sed -n 's/^wrote \(.*\.root\).*/\1/p' "$GENLOG" | tail -1); rm -f "$GENLOG"
[[ -n $ROOT ]] || { echo "generate did not report an output file"; exit 1; }
srun --chdir="$REPO" pixi run --manifest-path "$REPO/pixi.toml" jet-surrogate skim "$ROOT" --out "$REPO/data/skim"
date; echo done
