"""Score every reco jet with the trained tagger and derive surrogate labels.

    sbatch slurm/gpu.sbatch apply-tagger

For each data/skim/<stem>.h5 writes data/scores/<stem>.h5 with
    reco_logit  [n_reco]   tagger logit
    reco_pass   [n_reco]   logit > working-point threshold
    truth_label [n_truth]  matched reco jet exists and passes (the surrogate target)
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..data import SCORES_DIR, skim_files
from ..training import load_checkpoint, pick_device, score_padded


def add_arguments(ap) -> None:
    ap.add_argument("--data", default="data/skim")
    ap.add_argument("--model", default="models/tagger/tagger.pt")
    ap.add_argument("--out", default=SCORES_DIR)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="rescore files that already have scores")
    ap.add_argument("--max-files", type=int, default=None,
                    help="debug: keep the first N skims of every (sample point, split)")


def run(args) -> None:
    device = pick_device(args.device)
    model, pre, extra = load_checkpoint(args.model, device)
    model.to(device)
    thr = float(extra["working_point"]["threshold_logit"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = skim_files(args.data)
    if args.max_files:
        seen: dict = {}
        files = [f for f in files if seen.setdefault((f.tag, f.split), [0])[0] < args.max_files
                 and not seen[(f.tag, f.split)].__setitem__(0, seen[(f.tag, f.split)][0] + 1)]
    for f in files:
        dst = out / f"{f.path.stem}.h5"
        if dst.exists() and not args.force:
            continue
        with h5py.File(f.path, "r") as h:
            tracks = h["reco_tracks"][...]
            match = h["truth_jets"]["match"]
        logit = score_padded(model, pre, tracks, device) if len(tracks) else np.zeros(0, np.float32)
        passed = logit > thr
        label = np.where(match >= 0, passed[np.maximum(match, 0)], False)
        with h5py.File(dst, "w") as h:
            h.create_dataset("reco_logit", data=logit.astype(np.float32))
            h.create_dataset("reco_pass", data=passed)
            h.create_dataset("truth_label", data=label)
            h.attrs["threshold_logit"] = thr
        print(f"{f.path.name}: {len(logit)} reco jets, pass {passed.mean() if len(passed) else 0:.4f}; "
              f"{len(label)} truth jets, label {label.mean() if len(label) else 0:.4f}", flush=True)
