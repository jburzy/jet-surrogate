# CLAUDE.md — project handoff

Working notes for continuing jet-surrogate in a fresh session. Read the
README first for the physics and the commands; this file carries the
cluster specifics, the design decisions and their reasons, and the state of
play.

## State of play (2026-08-28)

- Repo refactored from the abandoned HEPMC/ONNX prototype to a full chain
  (generation -> Delphes -> skim -> tagger -> surrogate -> closure). The old
  `reader/reconstruction/matching/inference/pipeline/cli` modules are gone;
  the 32-feature truth set they described lives on, reorganized, in
  `features.PART_FLOATS/PART_CATS`.
- Environment: pixi, `default` (CPU torch) and `gpu` (CUDA torch, linux-64
  only). The login node has no NVIDIA driver, so the gpu env only solves
  with `CONDA_OVERRIDE_CUDA=12.4` exported (also set in the sbatch files).
- Production on `sooner_test`, 10k events per task, each task generates and
  skims: 100 QCD (1M events) + nominal signal 3 ctau x 34 seeds (~1M events)
  + alternative-mass signal (m_pid = 10 and 2 GeV, 3 ctau x 5 seeds).
  Author's target: O(1M) background and O(1M) signal events.
- **ML runs only on the GPU node via SLURM** (`slurm/gpu.sbatch`,
  partition `ouheptmp`, 4x L40S, no time limit). Author's instruction: never
  train on CPU.

## Cluster facts

- `sooner_test` (underscore) for generation, account `general`, 1500 job cap,
  `DefMemPerCPU` 1 GB so `--mem` is always explicit. `ouheptmp` for GPU work;
  `--gres=gpu:1` works with or without `--container=el9hw` (tested).
- **All data lives on ourdisk** (author's rule, as for displaced-observables):
  `data/` -> `/ourdisk/hpc/ouhep/jburzyns/dont_archive/jet-surrogate/data`,
  logs in `.../jet-surrogate/logs`. The first production was written to
  `/scratch/jburzyns/jet-surrogate` and rsynced over; pixi cache stays at
  `PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache`.
- Compute nodes cannot see the login node's `/tmp`; helper scripts for
  `srun --overlap` inspection live in `slurm/tools/`.
- NFS attribute caching: a compute node can run a *stale* copy of a source
  file for ~30-60 s after an edit on the login node. Wait before resubmitting.
- Measured: DelphesPythia8 ~0.02 s/event for both samples (200 s per 10k
  file), skim ~200 s per 10k file, 0.9 GB ROOT + 90 MB HDF5 per file, 6 GB
  memory is plenty. The full 190-task production is ~15 min wall at 100
  concurrent tasks.
- `./slurm/manifest.sh --todo && ./slurm/submit.sh` refills missing files;
  `pixi run jet-surrogate skim --all` skims any ROOT file without a skim.

## Plotting

`plotting.py`: `mplhep` ATLAS style (`plt.style.use(hep.style.ATLAS)`),
Okabe-Ito categorical palette in fixed order, `decorate(ax)` annotation
(Pythia + Delphes, sqrt(s), process, jet algorithm; no experiment label),
PNG + PDF side by side. Only `commands/visualize.py` imports it.

## Design decisions

- **One executable.** `jet-surrogate <command>` (`cli.py` + `commands/`)
  drives every step; pixi tasks and sbatch files are aliases of it. No
  standalone scripts. Compute commands write HDF5/JSON/checkpoints only;
  `visualize` is the sole matplotlib consumer and reads only those files
  (author's rule: plotting fully decoupled from heavy compute).

- **DelphesPythia8, not HepMC.** One executable produces the generator
  record, the smeared tracks and the PF jets in one ROOT file; no HepMC
  round trip and no Python event loop (the old generator was ~10x slower).
- **Track smearing.** The stock ATLAS card does not smear d0/z0. The card in
  `cards/delphes/` inserts a `TrackSmearing` module after `TrackMerger` with
  the resolutions from Delphes' own `trackResolutionATLAS.tcl`; ECal and tau
  tagging read the smeared tracks. Delphes tracking efficiency has no
  production-radius dependence and there are no fakes; fine for ctau <= 1 mm.
- **Particle status codes in Delphes.** `Particle.Status` is HepMC-like for
  SM particles (1 stable, 2 decayed hadron/tau) but raw Pythia codes for dark
  hadrons (83/84 fragmentation, 91 decay products). `features.particle_columns`
  therefore keeps status 1, status 2, and dark-sector ids with status 81-99.
  Decay vertices come from the first daughter's production vertex.
- **Jet-signed d0** uses only the smeared D0 and phi:
  sign(d0 sin(phi_trk - phi_jet)), since Delphes' PCA vector is
  d0 (sin phi, -cos phi).
- **Association through subjets** (dR < 0.4 to the nearest constituent
  small-R jet) rather than a fixed cone around the large-R axis; identical on
  the reco and truth side, so the surrogate sees the same "jet" the tagger
  sees.
- **Truth jets down to 150 GeV** so the surrogate can model the reco 200 GeV
  turn-on; reco jets without a truth match above 150 GeV are ignored
  (negligible).
- **Splits by seed, never by event** (`data.py`: QCD 1-60/61-70/71+; nominal
  signal 1-24 tagger train, 25-27 tagger val, 28-34 test, **35-54 surrogate
  train**). Author's rule (2026-08-29): the surrogate and the tagger train on
  disjoint events, so the surrogate labels are the tagger's out-of-sample
  decisions. Before this the surrogate trained on seeds 1-27 (the tagger's
  own training jets), a mild bias in the direction of the tagger's train-set
  optimism. Alternative-mass points are never trained on.
- **Lifetime grid** extended (author, 2026-08-29) to
  {0.01, 0.05, 0.1, 0.5, 1, 5} mm for all masses. At 5 mm the dark pions fly
  ~200 mm, beyond the pixel layers, and Delphes has no radius-dependent
  tracking inefficiency, so that point is optimistic vs a real detector.
- **Coherent dark-sector rescaling** for "different dark-pion mass": all of
  Lambda, m_qv, m_pid, m_rhod, pTminFSR scale with m_pid / 5 GeV, so the
  shower stays self-consistent instead of changing only one mass.
- **Surrogate trains on signal only** (author, 2026-08-29): it is never
  applied to background, and QCD jets at label ~0 would only distort the
  calibration on QCD-like signal jets. `data.qcd_jets_per_file: 0` is the
  default; the production surrogate used 4000 per file (280k QCD jets).
- **Poisson-binomial SR prediction.** The surrogate's per-jet probabilities
  are combined into P(>= 2 pass) per event; the hard-threshold (0.5) variant
  is reported alongside and is expected to be worse.

- **Model-variant scans** (author, 2026-08-29), evaluation only, at
  m_pid = 5 GeV and ctau = 0.1 mm, 5 seeds each: Lambda scan
  m_pi/Lambda in {0.2, 0.35, (0.5 nominal), 0.7, 1.0, 1.4} via
  `generate --lambda` (m_qv = Lambda, pTminFSR = 1.1 Lambda,
  m_rho = max(2 Lambda, 2.2 m_pi) so the forced rho -> pi pi stays open), and
  a dark-flavour scan nFlav in {(1), 2, 3} via `--nflav` (off-diagonal mesons
  4900211/4900213 copy the diagonal masses, lifetime and decays; Z' decays
  to all flavours). Stems carry `_lam<L>` / `_nf<N>`; `SkimFile.variant`
  keeps them out of every training selection. Plots:
  `sr_efficiency_scan_{lambda,nflav}`.

- **HepMC entry point** (`predict`, `export-hepmc`, `generate --format hepmc`,
  `hepmc_io.py`, 2026-08-29): reads HepMC2/3 via the pyHepMC3 bindings into
  the same particle record as the Delphes `Particle` branch (particles
  renumbered so siblings are contiguous), reuses `skim.skim_truth`, and
  returns per-event P(>= 2 tagged) + the sample SR efficiency. Validated on
  identical events exported from Delphes: same jets, same particle sets,
  per-jet probabilities equal to 2e-7. Reading is ~0.15 s/event (Python
  loop over pyHepMC3 objects); the writer builds one vertex per decaying
  particle and adds vertex-less particles with `evt.add_particle` (a root
  vertex without incoming particles breaks the ASCII reader).
- **Delphes `Particle.Charge` is -999 for dark-sector ids** (unknown to
  Delphes' PDG table). Until 2026-08-29 that value fed the surrogate's
  `charge` feature (a de-facto dark-hadron flag). Truth charges now come
  from the PDG id via Pythia's particle table on both paths; all skims were
  regenerated (`slurm/reskim.sh`, array 33400490) and the surrogate
  retrained (33400491 -> evaluate 33400492). Results with the old feature
  are kept in `results/summary_charge999.*`.

## Gotchas hit

- **`sbatch --export=ALL` leaks the submitting shell's `WORKDIR`**: the
  cluster's container wrapper does `cd $WORKDIR` before every `srun`, so with
  `WORKDIR=/home/jburzyns` in the environment every `pixi run` failed with
  "could not find pixi.toml" (jobs 33376719/20), and even with
  `srun --chdir` + `--manifest-path` the CLI then ran from `$WORKDIR` and saw
  no `data/skim` (job 33377217). Every sbatch script now exports
  `WORKDIR=<repo>` explicitly. The wrapper also `eval`s the srun command, so
  never rely on shell quoting inside `srun bash -c '...'`.

- **torch DataLoader workers deadlock** on the GPU node: the forked workers
  re-spawned at epoch 1 hung in a futex forever (job 33367802). Training now
  slices in-memory tensors directly (`PaddedDataset.batches()`, and the
  Lightning loaders use `batch_size=None` + `BatchSampler` with
  `num_workers: 0`).

- pixi `gpu` env: `__cuda` virtual package missing on the login node ->
  `CONDA_OVERRIDE_CUDA=12.4`.
- `generate_sample` runs DelphesPythia8 with `cwd=out_dir`; all paths are
  resolved to absolute first (a relative `data/...` silently broke).
- awkward: `ak.cartesian(..., nested=True)` on events with zero small-R
  jets; `associate()` is written in flat numpy with explicit offsets for
  that reason. In tests, build jagged records with `ak.zip`, not a list of
  dicts (the empty-list field infers `unknown` type and breaks cartesian).
- Reclustering test: two 150 GeV subjets 0.6 apart give 2*150*cos(0.3) =
  286.6 GeV, not 300.

## Results so far

Preliminary tagger (job 33367779, old splits: 12 train seeds, 600k signal
events, 30 epochs, val acc 0.940, AUC 0.973 on test seeds). Jet efficiency
at the 1/1000 QCD working point (600 / 599k QCD test jets pass):

| ctau | 0.01 mm | 0.1 mm | 1 mm |
|---|---|---|---|
| signal jet eff | 0.251 | 0.796 | 0.926 |

So the 0.01 mm point is the hard one (tracks are effectively prompt at
ATLAS d0 resolution, ~10-35 um); the tagger there relies on substructure.

Production tagger (job 33386471, final splits, 2.18M training jets, early
stop after epoch 26, best epoch 20, val loss 0.177, val acc 0.934; AUC 0.975
on test seeds; working point logit 4.315, 600 / 599,183 QCD test jets):

| ctau | 0.01 mm | 0.1 mm | 1 mm |
|---|---|---|---|
| signal jet eff at 1/1000 | 0.295 | 0.802 | 0.921 |

Production chain: js-tagger 33386471 -> js-apply 33386472 -> js-surrogate
33386473 -> js-evaluate 33386474 (2026-08-29). Surrogate: early stop after
epoch 25, best val loss 0.376, val acc 0.843, jet AUC 0.906 (0.853 on
signal jets), mean prob vs label on signal 0.608 vs 0.622, on QCD 0.039 vs
4e-4 (over-predicts QCD: consequence of subsampling QCD to 280k jets).

**Closure, SR efficiency actual -> surrogate (ratio), `results/summary.md`:**

| m_pid | ctau 0.01 mm | 0.1 mm | 1 mm |
|---|---|---|---|
| 5 GeV (test seeds) | 0.066 -> 0.057 (0.87) | 0.418 -> 0.384 (0.92) | 0.538 -> 0.510 (0.95) |
| 10 GeV (unseen) | 0.039 -> 0.044 (1.12) | 0.424 -> 0.318 (0.75) | 0.548 -> 0.490 (0.90) |
| 2 GeV (unseen) | 0.058 -> 0.124 (2.12) | 0.333 -> 0.415 (1.25) | 0.496 -> 0.510 (1.03) |

Reading: in-sample closure 5-13% low (about half of it is passing reco jets
whose truth partner is below the 150 GeV truth threshold, see the
`sr_from_labels` reference); mass extrapolation is the weak point, with
opposite signs for 2 and 10 GeV and the largest error at ctau = 0.1 mm
(calibration plot shows the off-diagonal bands). Hard 0.5 thresholding
fails completely at ctau = 0.01 mm (per-jet probabilities ~0.2), so the
Poisson-binomial combination is essential. Next physics steps: train the
surrogate on two masses and test on the third, lower the truth-jet
threshold, weight rather than subsample QCD. Preview chain with the
preliminary tagger (`results_prelim/`) gave the same picture.

## Results, corrected-charge chain (2026-08-29 evening, `results/summary.md`)

Tagger 33396091 (6 lifetimes): AUC 0.979, WP logit 4.81, jet eff at 1/1000
0.250 / 0.708 / 0.815 / 0.909 / 0.928 / 0.947 for ctau 0.01 / 0.05 / 0.1 /
0.5 / 1 / 5 mm. Surrogate 33400491 (signal only, seeds 35-54, PDG charges,
early-stopped at epoch 2 with val 0.3907, AUC 0.868; patience raised to 10
afterwards). SR efficiency predicted / actual: m5 test seeds 1.17, 0.98,
0.94, 0.97, 0.96, 0.96; m10 1.32, 0.54, 0.70, 0.88, 0.91, 0.94; m2 4.1,
1.60, 1.34, 1.09, 1.04, 1.04; Lambda scan (0.1 mm) 1.61, 0.99, (0.94),
1.15, 1.69, 1.83 for m_pi/Lambda 0.2 .. 1.4; nFlav 2/3: 0.89 / 0.87. The
prediction is flat in Lambda and nFlav (0.38-0.41) while the tagger varies
0.22-0.44: the surrogate learned displacement, not shower structure.
This model is the analysis-library entry `emerging-jets-delphes` v0.3.

## Results, Lambda-augmented surrogate (2026-08-30, job 33410294 -> evaluate 33411517)

Surrogate v0.4 candidate: trained on nominal seeds 35-54 (6 lifetimes) plus
lam25/lam5 seeds 35-44 (6 lifetimes), 3.54M truth jets, signal only. Ran the
full 30 epochs (best epoch 22, val loss 0.4238, AUC 0.867, calibrated:
mean prob 0.6315 vs label 0.6311). Closure (pred/actual):

- Lambda scan at 0.1 mm: 0.91 (0.2, trained), 0.81 (0.35 held out),
  0.90 (0.5), 0.64 (0.7 held out), 0.88 (1.0, trained), 0.96 (1.4 held out).
  Previously flat prediction (ratios 1.6-1.8 at the ends); now the
  prediction tracks the Lambda dependence, with undershoot when
  interpolating between trained points (worst 0.64 at 0.7).
- Z' scan (2/2.5/3 TeV, never trained): ratios 0.88-0.98 for every
  lifetime >= 0.05 mm at all three masses, same level as in-sample. The
  pT dependence is learned. ctau = 0.01 mm degrades with mass
  (1.2 -> 1.9), the known hard regime.
- m2 improved a lot (1.97, 1.12, 1.08, 0.99, 0.99, 1.01 vs 4.1, 1.6,
  1.34, 1.09, 1.04, 1.04 before); **m10 got worse** (0.59, 0.40, 0.53,
  0.72, 0.77, 0.86 vs 1.32, 0.54, 0.70, 0.88, 0.91, 0.94): the
  fewer/harder-dark-hadrons -> lower-efficiency lesson from the Lambda
  scan at m5 misleads at m10, where harder hadrons come with high
  efficiency. Multi-mass training (TRAIN_MZPS-style swap experiments,
  or masses in training) is the indicated next step.
- nFlav 2/3: 0.88 / 0.87 (was 0.89 / 0.87).

## Lightning validation (2026-08-29, job 33396090, `models/validation/tagger`)

Like-for-like with the plain-torch production tagger (same 2.18M jets, 3
lifetimes, seeds 1-24): best val loss 0.1776 vs 0.1770 (both epoch 20),
AUC 0.97451 vs 0.97460, WP threshold 4.419 vs 4.315, jet efficiencies at
1/1000 0.280/0.806/0.926 vs 0.295/0.802/0.921. Loss curves tracked to the
third digit epoch by epoch. The 0.01 mm difference is threshold scatter on
the steepest part of the ROC (run-to-run, not systematic). Checkpoint
consistency (`slurm/tools/check_checkpoints.py`): .ckpt vs .pt logits
bit-exact, ONNX to 5e-6. Refactor considered validated.

## Paper

`paper/main.tex` (JHEP style, `jheppub.sty` vendored, same conventions as
displaced-observables: no semicolons, no em-dashes, `\obshead`,
subfigure layouts from single-panel PDFs, `\stub{}`/`\todo{}` markers).
Structure: Introduction (reuses the PRISM proposal language),
Simulation and event reconstruction, The detector-level tagger, The
truth-level surrogate, Results (closure, unseen masses, what it learns),
Conclusions, appendices with the Pythia and Delphes settings.
`graphicspath` falls back to `results_smoke/figures` until `visualize`
has produced `results/figures`. Build: `pixi run paper` (tectonic; until
the lock is re-solved use displaced-observables' tectonic binary). Zero
overfull boxes at the time of writing. Every number in the results is a
`\todo{}` waiting for the chain.

## Lightning refactor (done 2026-08-29; smoke-tested, validation runs submitted)

Mirror `~/ATLAS/EmergingJets/ej-vae` (surveyed; conventions below), keep the
`jet-surrogate` CLI and every file format unchanged, then validate against
the plain-torch results.

ej-vae conventions to copy:
- `LightningCLI` subclass (jsonargparse): `-n/--name`, `--log_suffix`, links
  `name` -> `trainer.logger.init_args.experiment_name`; per-run dir
  `logs/<name>_<YYYYmmdd-THHMMSS>/{config.yaml, norm.yaml, ckpts/}`;
  `SaveConfigCallback` copies the normalization YAML into the run dir.
- YAML configs in `src/jet_surrogate/configs/{tagger,surrogate}.yaml`:
  `model:` (class_path/init_args), top-level `optimizer:`/`lr_scheduler:`
  (LightningCLI automatic keys, AdamW + cosine), `data:` (files/splits,
  variables, batch_size, num_workers), `trainer:` (max_epochs, accelerator,
  devices, `CometLogger` with `project: jet-surrogate`, callbacks).
- `LightningDataModule` with lazy per-worker HDF5 open in the Dataset
  (`_setup()` inside `__getitem__`), normalization done *inside the model*
  by an `InputNorm` module holding mean/std buffers from a YAML
  `{tracks|particles: {var: {mean, std}}}`.
- `LightningModule` wrapper: `forward` normalizes then calls the net, shared
  `_step(batch, prefix)`, metrics `train/loss`, `val/loss`, `val/acc`
  (`sync_dist=True`), NaN guards raise.
- Callbacks: `Checkpoint(ModelCheckpoint)` with `save_top_k=-1` and
  filename `epoch={epoch:03d}-val_loss={val/loss:.5f}`, dirpath
  `<log_dir>/ckpts`; `LearningRateMonitor`; add `EarlyStopping` (patience
  6, ej-vae has none). Best epoch = regex minimum of `val_loss=` in the
  filenames (`get_best_epoch`).
- `main.py` imports `comet_ml` before torch, sets `COMET_LOG_ENV_CONDA=false`;
  without `COMET_API_KEY` force `online=False` and `COMET_OFFLINE_DIRECTORY`
  = run dir. `COMET_API_KEY` is exported in the author's `.bashrc`.
- Submission: resources forwarded into the command so they cannot disagree
  (`--trainer.devices=$gpus --data.num_workers=$cpus`); keep our
  `slurm/gpu.sbatch` but pass these through.

Implemented as `lightning/{data,module,callbacks,cli}.py` + `configs/*.yaml`;
deviations from ej-vae: normalization stays in numpy (`Preprocessor`, saved
as `norm.yaml` + checkpoint hparams) rather than an in-model InputNorm, and
the warm-up + cosine schedule lives in `configure_optimizers` (needs the
step count) instead of YAML `optimizer:`/`lr_scheduler:` keys. Gotchas:
Lightning 2.6 `CometLogger` accepts no `experiment_name`/`save_dir`
(set via `experiment.set_name()` after instantiation), and `trainer.log_dir`
follows the logger (`./.cometml-runs`), so callbacks use
`trainer.default_root_dir` (the run dir). Original mapping:
`training.py` -> `lightning/{datamodule,module,callbacks,cli}.py`; `train-tagger`/`train-surrogate` become
`jet-surrogate fit --config configs/tagger.yaml [overrides]` plus a
post-fit step that writes `working_point.yaml`, `roc.json` and the ONNX
(tagger) or `summary.json` (surrogate) from the best checkpoint;
`apply-tagger`/`evaluate` load `.ckpt` via `load_from_checkpoint`.
Environment: add `lightning`, `jsonargparse[signatures]`, `comet_ml`
(and `tectonic`) to `pixi.toml` and re-solve once no GPU job reads the env.

Validation plan (author: "carefully validate that the updated structure
gives consistent results"):
1. Tagger retrained with the Lightning stack on the same seeds: AUC, the
   three working-point efficiencies (0.295 / 0.802 / 0.921) and best
   val loss (0.177) within run-to-run scatter; bit-exact logits between
   the `.ckpt` reloaded model and the in-memory model on one test file;
   ONNX output equal to torch output.
2. Surrogate: closure table equal to the production one within errors.
3. `apply-tagger` / `evaluate` / `visualize` produce identical file layouts.

Preview chain with the preliminary tagger, isolated outputs:
`models/prelim/tagger` (copy), `data/scores_prelim`,
`models/prelim/surrogate`, `results_prelim` (jobs 33387295-97).

## PRISM (separate repository)

The web service, the analysis library (including the ATLAS CalRatio entry
built on Zenodo 12957031), the CERN PaaS deployment and the GitHub
reinterpretation workflow moved on 2026-08-30 to
https://github.com/burzynski-lab/prism (site https://prism.web.cern.ch).
This repository is the training side only: nothing here is imported by
PRISM. The emerging-jets entry there carries its own copy of the inference
code (`analyses/emerging-jets-delphes/surrogate/`), verified to reproduce
this chain's per-jet probabilities exactly; particle charges there come
from the PDG numbering scheme (checked against Pythia's table). Releasing
a new surrogate means a pull request to PRISM with the new `surrogate.pt`,
figures, validation table and version. Leftovers on OSCER from the
CalRatio work: `data/external/calratio`, `data/test/calratio`.

## Next steps

1. Corrected-charge surrogate (33400491) -> evaluate (33400492): refresh
   `results/`, `visualize`, `models/release/`, paper tables; compare with
   `results/summary_charge999.md`.
2. Surrogate generalization: the prediction is flat in Lambda and nFlav
   while the tagger response varies by 2x; train on several masses and
   couplings, hold others out (generation is cheap: `generate --mpid`,
   `--lambda`, `--nflav`, then extend `SkimFile.split` for the new training
   points).
3. Truth-jet threshold below 150 GeV (the labels-only SR reference sits
   3-5% below the actual SR efficiency).
