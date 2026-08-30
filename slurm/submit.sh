#!/bin/bash
# Submit one array per manifest.  ./slurm/submit.sh [signal|qcd ...]   MAXRUN=150 to raise concurrency
set -euo pipefail
REPO=/home/jburzyns/jet-surrogate
DIR=$REPO/slurm
mkdir -p /ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs
MAXRUN=${MAXRUN:-100}
NEVENTS=${NEVENTS:-10000}
for s in ${*:-signal qcd variants}; do
    m=$DIR/manifest_$s.txt
    [[ -s $m ]] || { echo "manifest $m empty/missing; run ./slurm/manifest.sh"; continue; }
    n=$(wc -l < "$m")
    echo -n "$s: $n tasks -> "
    sbatch --job-name="gen-$s" --array="1-${n}%${MAXRUN}" \
           --export=ALL,MANIFEST="$m",NEVENTS="$NEVENTS" "$DIR/generate.sbatch"
done
