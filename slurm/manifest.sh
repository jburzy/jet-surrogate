#!/bin/bash
# Build the per-sample job manifests: one line per 10k-event file,
#   "<sample> <mpid|-> <ctau|-> <seed>"
# Usage: ./slurm/manifest.sh [--todo]   (--todo: only files whose ROOT output is missing)
set -euo pipefail
REPO=/home/jburzyns/jet-surrogate
DATA=$REPO/data/delphes
DIR=$REPO/slurm
TODO_ONLY=${1:-}

CTAUS="0.01 0.05 0.1 0.5 1 5"
SIGNAL_SEEDS=${SIGNAL_SEEDS:-54}     # nominal m_pid = 5 GeV: 1-24 tagger train, 25-27 val, 28-34 test, 35-54 surrogate train
TEST_MPIDS="10 2"                    # surrogate test points (rescaled dark sector)
TEST_SEEDS=${TEST_SEEDS:-5}          # 5 x 10k per (mass, ctau)
QCD_SEEDS=${QCD_SEEDS:-100}          # 1M QCD events

outfile() {  # <sample> <mpid> <ctau> <seed>
    case $1 in
        signal) printf '%s/signal_m%g_ctau%gmm_seed%d.root' "$DATA" "$2" "$3" "$4" ;;
        qcd)    printf '%s/qcd_seed%d.root' "$DATA" "$4" ;;
    esac
}
emit() {
    if [[ $TODO_ONLY == --todo ]] && [[ -s $(outfile "$@") ]]; then return; fi
    echo "$1 $2 $3 $4"
}

{ for s in $(seq 1 "$SIGNAL_SEEDS"); do for c in $CTAUS; do emit signal 5 "$c" "$s"; done; done
  for m in $TEST_MPIDS; do for s in $(seq 1 "$TEST_SEEDS"); do for c in $CTAUS; do emit signal "$m" "$c" "$s"; done; done; done
} > "$DIR/manifest_signal.txt"
for s in $(seq 1 "$QCD_SEEDS"); do emit qcd - - "$s"; done > "$DIR/manifest_qcd.txt"

# model variants at the nominal mass, evaluation only: Lambda scan at fixed m_pi (m_pi/Lambda = 0.2 .. 1.4)
# and a dark-flavour scan, both at ctau = 0.1 mm
VARIANT_CTAU=${VARIANT_CTAU:-0.1}
VARIANT_LAMBDAS="25 14.3 7.1 5 3.6"      # m_pi / Lambda = 0.2 0.35 0.7 1.0 1.4 (0.5 is the nominal)
VARIANT_NFLAVS="2 3"
VARIANT_SEEDS=${VARIANT_SEEDS:-5}
voutfile() { printf '%s/signal_m5_ctau%gmm_%s_seed%d.root' "$DATA" "$VARIANT_CTAU" "$1" "$2"; }
vemit() {   # vemit <suffix> <seed> <extra args>
    if [[ $TODO_ONLY == --todo ]] && [[ -s $(voutfile "$1" "$2") ]]; then return; fi
    echo "signal 5 $VARIANT_CTAU $2 $3"
}
{ for l in $VARIANT_LAMBDAS; do for s in $(seq 1 "$VARIANT_SEEDS"); do vemit "lam$l" "$s" "--lambda $l"; done; done
  for n in $VARIANT_NFLAVS; do for s in $(seq 1 "$VARIANT_SEEDS"); do vemit "nf$n" "$s" "--nflav $n"; done; done
} > "$DIR/manifest_variants.txt"
# Lambda points included in surrogate training (data.TRAIN_LAMBDAS), all six
# lifetimes, dedicated seeds 35-44 (disjoint from the 1-5 evaluation seeds
# and every tagger seed)
TRAIN_LAMBDAS="25 5"
TRAIN_VARIANT_SEEDS=${TRAIN_VARIANT_SEEDS:-"35 44"}
tvoutfile() { printf '%s/signal_m5_ctau%gmm_lam%s_seed%d.root' "$DATA" "$1" "$2" "$3"; }
{ for l in $TRAIN_LAMBDAS; do for c in $CTAUS; do for s in $(seq ${TRAIN_VARIANT_SEEDS// / }); do
    if [[ $TODO_ONLY == --todo ]] && [[ -s $(tvoutfile "$c" "$l" "$s") ]]; then continue; fi
    echo "signal 5 $c $s --lambda $l"
  done; done; done
} > "$DIR/manifest_variants_train.txt"

# Z'-mass scan at the nominal dark sector, all lifetimes: closure test of the
# learned pT dependence (evaluation seeds 1-5) plus dedicated training seeds
# 35-44 for the swap experiments
MZP_POINTS="2000 2500 3000"
mzoutfile() { printf '%s/signal_m5_ctau%gmm_mzp%s_seed%d.root' "$DATA" "$1" "$2" "$3"; }
{ for z in $MZP_POINTS; do for c in $CTAUS; do for s in $(seq 1 5) $(seq 35 44); do
    if [[ $TODO_ONLY == --todo ]] && [[ -s $(mzoutfile "$c" "$z" "$s") ]]; then continue; fi
    echo "signal 5 $c $s --mzp $z"
  done; done; done
} > "$DIR/manifest_mzp.txt"
for f in "$DIR"/manifest_*.txt; do printf '%-50s %5d jobs\n' "$f" "$(wc -l < "$f")"; done
