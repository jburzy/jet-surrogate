#!/bin/bash
# Merge the minimum-bias HepMC2 files into the binary pileup libraries read
# by Delphes' PileUpMerger, disjoint between the tagger's train / val / test
# events so no pileup event is shared across roles, then delete the
# intermediates. Seeds: 1-160 train, 161-170 val, 171-200 test.
#SBATCH --partition=sooner_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --job-name=minbias-convert
#SBATCH --output=/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/%x_%j.out
#SBATCH --error=/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/%x_%j.err
set -euo pipefail
REPO=/home/jburzyns/jet-surrogate
cd "$REPO"; export WORKDIR=$REPO
D=$REPO/data/minbias
train=(); val=(); test=()
for s in $(seq 1 200); do
  f=$D/minbias_seed$s.hepmc
  [[ -s $f ]] || { echo "missing $f"; exit 1; }
  if   (( s <= 160 )); then train+=("$f")
  elif (( s <= 170 )); then val+=("$f")
  else                      test+=("$f"); fi
done
B=$REPO/.pixi/envs/default/bin/hepmc2pileup
srun --chdir="$REPO" "$B" "$D/MinBias_train.pileup" "${train[@]}"
srun --chdir="$REPO" "$B" "$D/MinBias_val.pileup"   "${val[@]}"
srun --chdir="$REPO" "$B" "$D/MinBias_test.pileup"  "${test[@]}"
ls -la "$D"/MinBias_*.pileup
rm -f "$D"/minbias_seed*.hepmc
