"""Skim discovery, sample splits and in-memory jet tables for training.

Splits are by *file* (generator seed), never by event, so no event leaks
between train / val / test:

    qcd                 seeds 1-60 train, 61-70 val, 71+ test
    signal m_pid = 5    seeds 1-24 train, 25-27 val, 28-34 test, 35+ surrogate
    signal other m_pid  all test (surrogate evaluation points)
    Lambda variants     m_pi/Lambda in TRAIN_LAMBDA_RATIOS and seed >= 35:
                        surrogate; every other variant file: test

The tagger trains on train seeds, early-stops on val seeds and fixes its
working point on test seeds. The surrogate trains on the ``surrogate`` seeds
only, which the tagger has never seen, so its labels are the tagger's
out-of-sample decisions (author's rule: disjoint detector-level and
surrogate training sets). Closure is evaluated on the test seeds, unseen by
both networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import os

import h5py
import numpy as np

from .generate import NOMINAL_MPID
from .skim import parse_stem

SCORES_DIR = os.environ.get("JS_SCORES_DIR", "data/scores")   # override for smoke tests

# Lambda-scan points included in surrogate training (step 1 of the
# generalization plan): train on m_pi/Lambda = 0.2, 0.5 (nominal), 1.0 and
# hold out 0.35, 0.7, 1.4. Only dedicated seeds (>= 35, disjoint from the
# 1-5 evaluation seeds and every tagger seed) are used.
TRAIN_LAMBDAS = (25.0, 5.0)

# Z'-mass points included in surrogate training (empty: the mzp variants are
# evaluation-only closure tests of the learned pT dependence). Swap
# experiments fill this, e.g. TRAIN_MZPS = (2000.0, 3000.0), and retrain.
TRAIN_MZPS: tuple = ()


@dataclass(frozen=True)
class SkimFile:
    path: Path
    sample: str
    tag: str
    ctau: float
    mpid: float
    seed: int
    lam: float = -1.0        # -1: nominal Lambda scaling
    nflav: int = -1          # -1: nominal (one dark flavour)
    mzp: float = -1.0        # -1: nominal Z' mass (1.5 TeV)
    mu: float = -1.0         # -1: no pileup overlay

    @property
    def variant(self) -> bool:
        """Model variant at the nominal dark sector (Lambda, nFlav or Z'-mass scan)."""
        return self.lam > 0 or self.nflav > 0 or self.mzp > 0

    @property
    def split(self) -> str:
        if self.sample == "qcd":
            return "train" if self.seed <= 60 else "val" if self.seed <= 70 else "test"
        if self.variant:
            # dedicated high seeds of the training variant points feed the
            # surrogate; the low evaluation seeds (and every held-out
            # variant) stay test
            trained = self.lam in TRAIN_LAMBDAS or self.mzp in TRAIN_MZPS
            return "surrogate" if trained and self.seed >= 35 else "test"
        if self.mpid == NOMINAL_MPID:
            return ("train" if self.seed <= 24 else "val" if self.seed <= 27
                    else "test" if self.seed <= 34 else "surrogate")
        return "test"

    @property
    def label(self) -> int:
        return 0 if self.sample == "qcd" else 1

    @property
    def scores_path(self) -> Path:
        return Path(SCORES_DIR) / f"{self.path.stem}.h5"


def skim_files(data_dir: str | Path = "data/skim", *, samples=None, splits=None,
               mpid=None, ctaus=None, mu: float | None = None,
               require_scores: bool = False) -> list[SkimFile]:
    """Discover skims, optionally filtered by sample / split / signal mass.
    ``require_scores`` drops (and reports) skims that apply-tagger has not
    scored yet, so a partially scored production never raises deep inside."""
    out, missing = [], []
    for p in sorted(Path(data_dir).glob("*.h5")):
        m = parse_stem(p.stem)
        f = SkimFile(p, m["sample"], m["tag"], m["ctau"], m["mpid"], m["seed"], m["lam"], m["nflav"], m["mzp"], m["mu"])
        if f.mu != (-1.0 if mu is None else mu):
            continue                       # pileup and no-pileup chains never mix
        if samples and f.sample not in samples:
            continue
        if splits and f.split not in splits:
            continue
        if mpid is not None and f.sample == "signal" and (
                f.mpid != mpid or (f.variant and f.split != "surrogate")):
            continue                       # variants only train through their dedicated surrogate seeds
        if ctaus is not None and f.sample == "signal" and not any(abs(f.ctau - c) < 1e-9 for c in ctaus):
            continue
        if require_scores and not f.scores_path.exists():
            missing.append(f.path.name)
            continue
        out.append(f)
    if missing:
        print(f"[skim_files] skipping {len(missing)} skims without scores in {SCORES_DIR} "
              f"(run apply-tagger): {missing[:3]}{' ...' if len(missing) > 3 else ''}")
    return out


@dataclass
class JetTable:
    """Concatenated jets from several skim files, with provenance columns."""
    jets: np.ndarray        # structured: pt eta phi mass nsub event match n_assoc
    objs: np.ndarray        # padded (n_jets, slots) tracks or particles
    file_idx: np.ndarray    # index into ``files``
    label: np.ndarray       # 1 signal / 0 qcd
    ctau: np.ndarray
    mpid: np.ndarray
    files: list[SkimFile]
    n_events: np.ndarray    # per file

    def __len__(self):
        return len(self.jets)

    @property
    def event_offsets(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.n_events)])

    @property
    def global_event(self) -> np.ndarray:
        """Event index unique across files."""
        return self.event_offsets[self.file_idx] + self.jets["event"]

    @property
    def total_events(self) -> int:
        return int(self.n_events.sum())

    def subset(self, mask):
        return JetTable(self.jets[mask], self.objs[mask], self.file_idx[mask], self.label[mask],
                        self.ctau[mask], self.mpid[mask], self.files, self.n_events)


def load_table(files: list[SkimFile], side: str, *, max_jets_per_file: int | None = None,
               seed: int = 0, with_scores: tuple[str, ...] = (), with_objs: bool = True, log=None) -> tuple[JetTable, dict]:
    """Load ``side`` in {"reco", "truth"} jets and their padded objects.

    ``with_scores`` names datasets to read from the matching scores file
    (e.g. "reco_logit", "truth_label"); they are returned in a dict aligned
    with the table. ``max_jets_per_file`` randomly subsamples large files
    (QCD) to bound memory.
    """
    jk, ok = ("reco_jets", "reco_tracks") if side == "reco" else ("truth_jets", "truth_parts")
    if not files:
        raise SystemExit(f"load_table: no skim files given (side={side}); check --data and the seed splits")
    rng = np.random.default_rng(seed)
    jets, objs, fidx, lab, ct, mp, nev = [], [], [], [], [], [], []
    extra = {k: [] for k in with_scores}
    for i, f in enumerate(files):
        with h5py.File(f.path, "r") as h:
            j = h[jk][...]
            n = len(j)
            sel = np.arange(n)
            if max_jets_per_file is not None and n > max_jets_per_file:
                sel = np.sort(rng.choice(n, max_jets_per_file, replace=False))
            if with_objs:
                o = h[ok][sel] if len(sel) < n else h[ok][...]
            else:
                o = np.zeros(len(sel), np.int8)
            nev.append(int(h.attrs["n_events"]))
        if with_scores:
            with h5py.File(f.scores_path, "r") as h:
                for k in with_scores:
                    extra[k].append(h[k][...][sel])
        jets.append(j[sel]); objs.append(o)
        fidx.append(np.full(len(sel), i, np.int32)); lab.append(np.full(len(sel), f.label, np.int8))
        ct.append(np.full(len(sel), f.ctau, np.float32)); mp.append(np.full(len(sel), f.mpid, np.float32))
        if log:
            log(f"  loaded {f.path.name}: {len(sel)}/{n} {side} jets")
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    table = JetTable(cat(jets), cat(objs), cat(fidx), cat(lab), cat(ct), cat(mp), files, np.array(nev))
    return table, {k: cat(v) for k, v in extra.items()}


def load_transformed(files: list[SkimFile], side: str, pre, *, max_jets_per_file: int | None = None,
                     seed: int = 0, with_scores: tuple[str, ...] = (), log=None):
    """Memory-lean loader: read one file at a time, apply ``pre.transform``
    immediately and keep only the model-space arrays (x float32, cats int32,
    mask bool), never the raw structured table for the whole sample.

    Returns (x, cats, mask, meta) with meta = JetTable-like provenance
    (jets, file_idx, label, ctau, mpid, files, n_events) and the requested
    score datasets.
    """
    xs, cs, ms = [], [], []
    jets, fidx, lab, ct, mp, nev = [], [], [], [], [], []
    extra = {k: [] for k in with_scores}
    rng = np.random.default_rng(seed)
    jk, ok = ("reco_jets", "reco_tracks") if side == "reco" else ("truth_jets", "truth_parts")
    for i, f in enumerate(files):
        with h5py.File(f.path, "r") as h:
            j = h[jk][...]
            n = len(j)
            sel = np.arange(n)
            if max_jets_per_file is not None and n > max_jets_per_file:
                sel = np.sort(rng.choice(n, max_jets_per_file, replace=False))
            o = h[ok][sel] if len(sel) < n else h[ok][...]
            nev.append(int(h.attrs["n_events"]))
        x, c, m = pre.transform(o)
        del o
        xs.append(x); cs.append(c.astype(np.int32)); ms.append(m)
        if with_scores:
            with h5py.File(f.scores_path, "r") as h:
                for k in with_scores:
                    extra[k].append(h[k][...][sel])
        jets.append(j[sel]); fidx.append(np.full(len(sel), i, np.int32))
        lab.append(np.full(len(sel), f.label, np.int8))
        ct.append(np.full(len(sel), f.ctau, np.float32)); mp.append(np.full(len(sel), f.mpid, np.float32))
        if log:
            log(f"  loaded {f.path.name}: {len(sel)}/{n} {side} jets")
    cat = lambda a: np.concatenate(a) if a else np.zeros(0)
    meta = JetTable(cat(jets), np.zeros(0, np.int8), cat(fidx), cat(lab), cat(ct), cat(mp), files, np.array(nev))
    return cat(xs), cat(cs), cat(ms), meta, {k: cat(v) for k, v in extra.items()}
