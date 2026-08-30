#!/bin/bash
# Merge the minimum-bias HepMC2 files into the binary pileup library read by
# Delphes' PileUpMerger, then delete the intermediates.
#SBATCH --partition=sooner_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --job-name=minbias-convert
#SBATCH --output=/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/%x_%j.out
#SBATCH --error=/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/%x_%j.err
set -euo pipefail
REPO=/home/jburzyns/jet-surrogate
cd "$REPO"
export WORKDIR=$REPO
srun --chdir="$REPO" "$REPO/.pixi/envs/default/bin/hepmc2pileup" \
    "$REPO/data/minbias/MinBias.pileup" "$REPO"/data/minbias/minbias_seed*.hepmc
ls -la "$REPO/data/minbias/MinBias.pileup"
rm -f "$REPO"/data/minbias/minbias_seed*.hepmc
