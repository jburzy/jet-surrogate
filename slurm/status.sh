#!/bin/bash
# Production progress: files on disk vs manifests, queue state, failures in logs.
REPO=/home/jburzyns/jet-surrogate
echo "ROOT files : $(ls $REPO/data/delphes/*.root 2>/dev/null | wc -l)"
echo "skims      : $(ls $REPO/data/skim/*.h5 2>/dev/null | wc -l)"
echo "expected   : $(cat $REPO/slurm/manifest_*.txt | wc -l)"
echo "disk       : $(du -sh /ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/data 2>/dev/null | cut -f1)"
squeue -u "$USER" -h -o '%j %t' | sort | uniq -c
grep -l -i 'error\|traceback\|killed' /ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/gen-*.err 2>/dev/null | head
