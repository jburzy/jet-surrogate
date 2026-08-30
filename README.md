# jet-surrogate

A **truth-level surrogate** for a detector-level emerging-jet tagger.

The question: given a new dark-shower model (different dark-pion mass,
lifetime, ...), can we predict the signal-region efficiency of a full
detector-level analysis from generator truth alone, without running the
detector simulation and the tagger? The chain built here is

```
Pythia8 (Hidden Valley Z' -> dark showers; QCD dijets)
   -> Delphes, ATLAS card with smeared track impact parameters
   -> anti-kt R=0.4 jets reclustered into R=1.0 jets  (reco and truth)
   -> transformer tagger on the Delphes tracks in each large-R jet
   -> signal region: >= 2 large-R jets, pT > 200 GeV, tagger score above
      the working point at 1000x QCD jet rejection
   -> transformer surrogate on the generator particles in each truth
      large-R jet, trained to predict the tagger decision
   -> closure: predicted vs actual SR efficiency on an unseen mass point
```

## Setup

Everything (Pythia 8, Delphes + ROOT, fastjet, torch, onnx) is one
reproducible [pixi](https://pixi.sh) environment:

```bash
git clone <this repo> && cd jet-surrogate
pixi install                                    # default env (CPU torch): generation, skim, visualize
CONDA_OVERRIDE_CUDA=12.4 pixi install -e gpu    # CUDA torch (linux-64) for training; the override is
                                                # needed on login nodes without an NVIDIA driver
pixi run test                                   # 9 unit tests
```

`data/` is a symlink to bulk storage. On OSCER:

```bash
mkdir -p /ourdisk/hpc/ouhep/$USER/dont_archive/jet-surrogate/{data,logs}
ln -sfn /ourdisk/hpc/ouhep/$USER/dont_archive/jet-surrogate/data data
```

The `slurm/` scripts hard-code the author's repo and ourdisk paths, so edit
`REPO`/`WORKDIR` and the `#SBATCH --output` lines once on a new account.

## Running the full chain

```bash
# 1. generate + skim on the CPU partition (one array task per 10k-event file;
#    484 tasks: 100 QCD, 54 seeds x 6 ctau nominal signal, 5 seeds x 6 ctau x 2 test masses)
./slurm/manifest.sh && ./slurm/submit.sh
./slurm/status.sh                          # until "ROOT files" == "skims" == expected
./slurm/manifest.sh --todo && ./slurm/submit.sh   # refill anything missing (idempotent)

# 2. tagger -> apply -> surrogate -> evaluate on the GPU node, as a dependency chain
./slurm/ml_chain.sh --epochs 30            # add --after <genjob> to queue behind generation
squeue -u $USER                            # ~1 h tagger, ~10 min apply, ~2-3 h surrogate, ~10 min evaluate

# 3. figures and the paper, on any machine with the default env
pixi run jet-surrogate visualize           # results/figures/*.{png,pdf}
cat results/summary.md
pixi run paper                             # paper/main.pdf
```

To try the whole chain on a few files first: `SKIM=data/skim sbatch slurm/tools/smoke_lightning.sbatch`
(~10 min on the GPU node, writes to `models/smoke_pl`, `data/scores_smoke`, `results_smoke`, `logs/smoke-*`).

## Samples

Signal: Pythia8 Hidden Valley, s-channel Z'(1.5 TeV) -> q_v q_v, SU(3)
dark sector, m(pi_d) = 5 GeV nominal, lifetime grid **cτ = 0.01, 0.05, 0.1, 0.5, 1, 5 mm**
(`cards/pythia/signal_zprime_hv.cmnd`, identical benchmark to
displaced-observables). `--mpid` rescales the whole dark sector coherently
(Λ, m_qv, m_πd, m_ρd, pTminFSR ∝ m_πd); the surrogate is evaluated on
m(π_d) = 10 GeV and 2 GeV. Background: `HardQCD:all`, pTHat > 450 GeV.

```bash
pixi run jet-surrogate generate --sample signal --ctau 0.1 --nevents 10000 --seed 1
pixi run jet-surrogate generate --sample signal --grid --mpid 10 --nevents 10000
pixi run jet-surrogate generate --sample qcd --nevents 10000 --seed 1
pixi run jet-surrogate skim data/delphes/qcd_seed1.root       # -> data/skim/qcd_seed1.h5
```

Generation runs `DelphesPythia8` with
`cards/delphes/delphes_card_ATLAS_tracks.tcl`: the stock Delphes ATLAS card
plus a `TrackSmearing` module carrying the ATLAS d0/z0 resolution tables, so
`Track.D0/DZ` are smeared and `ErrorD0/ErrorDZ` are filled; jets are R=0.4
particle flow; the full generator record is written for the surrogate.
About 0.02 s/event; a 10k-event file is ~0.9 GB of ROOT and ~90 MB skimmed.

On the cluster the same two commands run as SLURM array tasks through
`slurm/manifest.sh` and `slurm/submit.sh` (see *Running the full chain*).
Cluster specifics that cost time to rediscover are in `CLAUDE.md`.

## Jets and inputs

Both sides use the same recipe: anti-kt R=0.4 jets (pT > 20 GeV,
|η| < 2.5) reclustered into anti-kt R=1.0 jets (pT > 200 GeV, |η| < 2.0);
truth large-R jets are kept down to 150 GeV so the surrogate sees the
turn-on. Objects belong to a large-R jet through its constituent small-R
jets (ΔR < 0.4 to the nearest small-R jet axis). Reco and truth large-R
jets are matched one-to-one within ΔR < 0.5.

- **Tagger inputs** (`features.TRACK_FLOATS`): per Delphes track with
  pT > 0.5 GeV, |η| < 2.5, up to 100 per jet: pT, pT/pT_jet, Δη, Δφ, ΔR,
  jet-signed d0, z0, σ(d0), σ(z0), charge, plus e/μ/hadron type.
- **Surrogate inputs** (`features.PART_FLOATS/PART_CATS`): per generator
  particle (stable, or decayed SM/dark hadron) with pT > 0.5 GeV, up to 150
  per jet: kinematics relative to the jet, charge, status, production and
  decay vertex (Lxy, z, flight length and direction), number of children
  and the first two children's kinematics, with embedded PDG ids of the
  particle and its children.

## Training and evaluation (GPU, via SLURM)

```bash
./slurm/ml_chain.sh [--after <genjob>] [--epochs 30]   # the four stages below as a dependency chain
sbatch slurm/gpu.sbatch train-tagger      # models/tagger/  (checkpoint, ONNX, working_point.yaml, roc.json)
sbatch slurm/gpu.sbatch apply-tagger      # data/scores/*.h5: logits, pass flags, surrogate labels
sbatch slurm/gpu.sbatch train-surrogate   # models/surrogate/
sbatch slurm/gpu.sbatch evaluate          # results/summary.{json,md}
pixi run jet-surrogate visualize          # results/figures/*.png|pdf, from stored outputs only
```

One executable drives everything: `jet-surrogate <command>` with the
commands `generate, skim, train-tagger, apply-tagger, train-surrogate,
evaluate, visualize` (`jet-surrogate <command> --help`). Compute steps only
write data (HDF5, checkpoints, JSON); `visualize` is the only step that
plots and it reads nothing but those files, so figures can be remade in
seconds on a laptop.

Splits are by generator seed (file), never by event: QCD seeds 1-60 /
61-70 / 71-100 (train / val / test) and nominal signal seeds 1-24 tagger
train, 25-27 tagger val, 28-34 test, 35-54 surrogate train, so the
surrogate and the tagger never share an event; the alternative mass points
are evaluation-only. Lifetimes: 0.01, 0.05, 0.1, 0.5, 1, 5 mm. The tagger's working point is
the logit threshold giving a QCD jet efficiency of 10⁻³ on the test seeds.
The surrogate target for a truth jet is "its matched reco jet exists and
passes the working point". `evaluate` compares, per (mass, cτ):

- **actual** SR efficiency: events with ≥ 2 passing reco jets (Delphes tracks + tagger);
- **predicted** SR efficiency: mean over events of P(≥ 2 truth jets pass)
  from the surrogate probabilities (Poisson binomial), and the hard 0.5
  threshold variant;
- per-jet closure vs truth jet pT and a calibration curve.

Both networks are `models.ParticleTransformer` (per-object MLP embedding,
4 pre-LN encoder layers, class token, MLP head) and are also exported to
ONNX with their preprocessing spec next to them.

Training runs on PyTorch Lightning (`src/jet_surrogate/lightning/`,
modeled on ej-vae): YAML configs in `src/jet_surrogate/configs/`, a
`LightningDataModule`, a `LightningModule` wrapping the transformer, a
checkpoint per epoch (`epoch=NNN-val_loss=X.ckpt`), early stopping, and
Comet logging (project `jet-surrogate`, offline without `COMET_API_KEY`).
Every run is self-contained in `logs/<name>_<timestamp>/{config.yaml,
norm.yaml, history.json, ckpts/}`; the best epoch is copied to `--out` as
`best.ckpt` next to the inference `.pt` and ONNX. Any `--section.key=value`
argument overrides the config:

```bash
sbatch slurm/gpu.sbatch train-tagger --name tagger --trainer.max_epochs=30 '--data.ctaus=[0.01,0.1,1]'
sbatch slurm/gpu.sbatch train-surrogate --data.qcd_jets_per_file=4000     # mix QCD back in
```

The surrogate trains on signal only (it is never applied to background)
and on seeds the tagger never saw.

## Reinterpreting a new model (no detector simulation)

```bash
jet-surrogate generate --sample signal --ctau 0.1 --mpid 10 --nevents 10000 --format hepmc   # or any generator's HepMC2/3
jet-surrogate predict --hepmc data/hepmc/signal_m10_ctau0.1mm_seed1.hepmc --model models/surrogate/surrogate.pt
```

`predict` clusters truth jets from the generator record, applies the
surrogate, and writes `results/predict/<stem>_hepmc.json` with the
signal-region efficiency (Poisson-binomial over the per-jet probabilities,
with its uncertainty) and an HDF5 with the per-event probabilities.
`export-hepmc` dumps the generator record of a Delphes file to HepMC3 for
cross-checks (the HepMC and Delphes paths agree to 2e-7 per jet).

## The analysis library and the web service

The service hosts a library of preserved analyses, one directory each under
`analyses/`, added by pull request:

```
analyses/<analysis-id>/
  analysis.yaml       title, experiment, status, signal-region definition, inputs, predictor, limits, figures
  surrogate.pt        the surrogate model (+ .onnx and preprocessor JSON)
  README.md           model card shown on the analysis page
  figures/*.png       validation figures shown on the analysis page
```

`tests/test_registry.py` (run by the `validate-analyses` workflow on every
PR) checks each record. An analysis brings its own inference code as
`predictor.py` next to the record (a registered `Predictor` subclass named
by `predictor.type`), may declare large model files under `assets` (fetched
once from their URL into `JS_ASSET_DIR`) and per-job choices under
`options` (a selector on the submit page). `analyses/_template/` is the
starting point. Entries: `emerging-jets-delphes` (the surrogate built here)
and `atlas-exot-2022-04-calratio` (the ATLAS CalRatio search through the
collaboration's published reinterpretation BDTs, Zenodo 12957031). Predictor types are registered in
`service/registry.py`; `jet_surrogate` (truth jets, per-jet probability,
Poisson-binomial event probability) is the first, and the example analysis
`emerging-jets-delphes` is the surrogate built in this repository.

```bash
pixi install -e infer
pixi run -e infer serve                 # http://localhost:8080: Carbon front end + JSON API (/api, /docs)
pixi run -e infer worker                # second shell: runs queued jobs
pixi run -e infer test-service          # end-to-end test on a tiny Pythia sample
jet-surrogate predict --hepmc events.hepmc --analysis emerging-jets-delphes   # the same, from the command line
```

Jobs are queued in `JS_SERVICE_DIR` (SQLite + one directory per job) and
executed by any number of workers; uploads are deleted after processing and
results expire (`JS_JOB_TTL_HOURS`). The front end (`service/static/`, IBM
Carbon, no build step) has pages for the library, each analysis, submission
with upload progress, job status and results, contribution instructions and
the API. `Dockerfile` packages everything; `deploy/paas/` holds the
OpenShift manifests and `deploy/README.md` the CERN PaaS steps. GitHub
issues opened with the *Reinterpretation request* template are processed by
the `reinterpret` workflow with the same container (a zero-infrastructure
fallback).

## Layout

```
pixi.toml                     environment + tasks
cards/pythia/                 Pythia cards (HV signal, QCD dijet)
cards/delphes/                Delphes ATLAS card with track smearing
src/jet_surrogate/
  cli.py                      the `jet-surrogate` executable (subcommand dispatcher)
  commands/                   one module per subcommand: add_arguments(), run()
  generate.py                 DelphesPythia8 driver (card assembly, mass/lifetime overrides)
  delphes_io.py               uproot reader
  jets.py                     small-R -> large-R reclustering, association, matching
  features.py                 track / particle feature tables, padding, transforms
  skim.py                     ROOT -> HDF5 per-jet tables (skim_truth: the generator-only half)
  hepmc_io.py                 HepMC2/3 reader, HepMC3 writer (predict, export-hepmc, generate --format hepmc)
  service/                    web service: registry of analyses, FastAPI app, SQLite job store, worker, static front end
  data.py                     skim discovery, seed splits, in-memory tables
  models.py, training.py      transformer, preprocessing, inference checkpoints, ONNX export
  lightning/                  Lightning data module, module, callbacks, CLI (training)
  configs/                    tagger.yaml, surrogate.yaml
  metrics.py                  working points, Poisson-binomial SR efficiency
slurm/                        generation arrays, GPU wrapper, ML dependency chain, re-skim, status, tools/
analyses/                     the analysis library (one directory per preserved analysis)
paper/                        JHEP-style draft (pixi run paper)
deploy/paas/                  OpenShift manifests for CERN PaaS (deploy/README.md)
Dockerfile, .github/          inference image, issue-driven reinterpretation workflow
tests/                        unit tests (pixi run test)
```
