#!/bin/bash
# Smoke test of the Lightning training path on a few files, then a
# checkpoint-consistency check (.ckpt vs .pt vs ONNX logits).
#   SKIM=data/skim sbatch slurm/tools/smoke_lightning.sbatch
#SBATCH --job-name=js-smoke-pl
#SBATCH --partition=ouheptmp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/%x_%j.out
#SBATCH --error=/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/logs/%x_%j.err
set -euo pipefail
cd /home/jburzyns/jet-surrogate
export WORKDIR=/home/jburzyns/jet-surrogate
export PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache CONDA_OVERRIDE_CUDA=12.4 PYTHONUNBUFFERED=1
export JS_SCORES_DIR=data/scores_smoke
SKIM=${SKIM:-data/skim}
run() { echo "== $1 ($(date +%T))"; pixi run --manifest-path /home/jburzyns/jet-surrogate/pixi.toml -e gpu jet-surrogate "$@"; }
rm -rf models/smoke_pl logs/smoke-tagger_* logs/smoke-surrogate_* data/scores_smoke
run train-tagger --out models/smoke_pl/tagger --epochs 2 --max-files 3 --name smoke-tagger --data.data_dir=$SKIM --log_suffix=smoke
run apply-tagger --model models/smoke_pl/tagger/tagger.pt --out data/scores_smoke --max-files 3 --force
run train-surrogate --out models/smoke_pl/surrogate --epochs 2 --max-files 3 --name smoke-surrogate --data.data_dir=$SKIM --data.qcd_jets_per_file=2000 --log_suffix=smoke
echo "== consistency ($(date +%T))"
pixi run --manifest-path /home/jburzyns/jet-surrogate/pixi.toml -e gpu python slurm/tools/check_checkpoints.py models/smoke_pl/tagger $SKIM
echo SMOKE-PL-OK
