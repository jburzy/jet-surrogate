#!/bin/bash
# Re-skim all Delphes files after a change to the skim/feature code.  MAXRUN=150 ./slurm/reskim.sh
set -euo pipefail
REPO=/home/jburzyns/jet-surrogate
LIST=$REPO/slurm/skim_list.txt
ls "$REPO"/data/delphes/*.root > "$LIST"
n=$(wc -l < "$LIST"); echo -n "reskim: $n files -> "
sbatch --parsable --job-name=skim --array="1-${n}%${MAXRUN:-120}" --export=ALL,LIST="$LIST" "$REPO/slurm/skim.sbatch"
