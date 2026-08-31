"""Closure test of the truth-level surrogate against the detector-level tagger.

    sbatch slurm/gpu.sbatch evaluate

For every evaluation sample (nominal m_pid = 5 GeV test seeds, and the
alternative-mass points, per lifetime):
  * actual    SR efficiency: fraction of events with >= 2 reco large-R jets
              (pT > 200 GeV) passing the tagger working point (Delphes tracks)
  * predicted SR efficiency: mean over events of P(>= 2 truth jets pass)
              from the surrogate's per-jet probabilities (Poisson binomial),
              plus the hard-threshold variant (prob > 0.5)
  * jet-level closure: efficiency vs truth jet pT, calibration curve, AUC
Writes results/summary.json (+ summary.md) and stores truth_prob in
data/scores/<stem>.h5. Figures come from ``jet-surrogate visualize``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from ..data import load_table, skim_files
from ..metrics import efficiency, predicted_sr_efficiency, sr_efficiency
from ..training import load_checkpoint, pick_device, score_padded

PT_BINS = np.array([150, 200, 250, 300, 400, 500, 600, 800, 1000, 1500])


def add_arguments(ap) -> None:
    ap.add_argument("--data", default="data/skim")
    ap.add_argument("--model", default="models/surrogate/surrogate.pt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-files", type=int, default=None, help="debug: cap files per sample point")


def evaluate_group(files, model, pre, device, log=print) -> dict:
    from sklearn.metrics import roc_auc_score  # lazy: keeps the inference env free of sklearn
    truth, ex = load_table(files, "truth", with_scores=("truth_label",))
    reco, rx = load_table(files, "reco", with_scores=("reco_pass",), with_objs=False)
    label = ex["truth_label"].astype(bool)
    logit = score_padded(model, pre, truth.objs, device)
    prob = 1 / (1 + np.exp(-logit))
    for i, f in enumerate(files):                   # keep the probabilities next to the tagger scores
        with h5py.File(f.scores_path, "a") as h:
            if "truth_prob" in h:
                del h["truth_prob"]
            h.create_dataset("truth_prob", data=prob[truth.file_idx == i].astype(np.float32))
    n_ev = truth.total_events
    act, act_err = sr_efficiency(reco.global_event, rx["reco_pass"].astype(bool), n_ev)
    pred, pred_err = predicted_sr_efficiency(truth.global_event, prob, n_ev)
    pred_thr, pred_thr_err = sr_efficiency(truth.global_event, prob > 0.5, n_ev)
    lab_sr, lab_sr_err = sr_efficiency(truth.global_event, label, n_ev)   # perfect-surrogate reference
    pt = truth.jets["pt"]
    idx = np.clip(np.digitize(pt, PT_BINS) - 1, 0, len(PT_BINS) - 2)
    vs_pt = []
    for b in range(len(PT_BINS) - 1):
        sel = idx == b
        if sel.sum() >= 20:
            e, err = efficiency(label[sel])
            vs_pt.append({"lo": float(PT_BINS[b]), "hi": float(PT_BINS[b + 1]), "n": int(sel.sum()),
                          "actual": e, "actual_err": err, "pred": float(prob[sel].mean())})
    edges = np.linspace(0, 1, 11)
    cal = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (prob >= lo) & (prob < hi + (hi == 1.0))
        if sel.sum() >= 20:
            e, err = efficiency(label[sel])
            cal.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()), "actual": e,
                        "actual_err": err, "pred": float(prob[sel].mean())})
    auc = float(roc_auc_score(label, logit)) if 0 < label.sum() < len(label) else None
    hedges = np.linspace(0, 1, 41)
    prob_hist = {"edges": hedges.tolist(),
                 "tagged": np.histogram(prob[label], hedges)[0].tolist(),
                 "untagged": np.histogram(prob[~label], hedges)[0].tolist()}
    res = {"n_events": int(n_ev), "n_truth_jets": int(len(truth)), "n_reco_jets": int(len(reco)),
           "sr_actual": act, "sr_actual_err": act_err, "sr_pred": pred, "sr_pred_err": pred_err,
           "sr_pred_thr": pred_thr, "sr_pred_thr_err": pred_thr_err,
           "sr_from_labels": lab_sr, "sr_from_labels_err": lab_sr_err,
           "jet_eff_actual": float(label.mean()), "jet_eff_pred": float(prob.mean()),
           "jet_auc": auc, "vs_pt": vs_pt, "calibration": cal, "prob_hist": prob_hist}
    log(f"  events {n_ev}: SR actual {act:.4f}+-{act_err:.4f}  predicted {pred:.4f}+-{pred_err:.4f}"
        f"  (thr 0.5: {pred_thr:.4f})  labels-only {lab_sr:.4f}  jet eff {label.mean():.4f} vs {prob.mean():.4f}")
    return res


def write_markdown(results: dict, path: Path) -> str:
    lines = ["# Surrogate closure", "",
             "| sample | events | SR eff actual | SR eff predicted | pred (thr 0.5) | jet eff actual | jet eff pred | jet AUC |",
             "|---|---|---|---|---|---|---|---|"]
    for tag, r in results.items():
        auc = "-" if r["jet_auc"] is None else f"{r['jet_auc']:.4f}"
        lines.append(f"| {tag} | {r['n_events']} | {r['sr_actual']:.4f} ± {r['sr_actual_err']:.4f} | "
                     f"{r['sr_pred']:.4f} ± {r['sr_pred_err']:.4f} | {r['sr_pred_thr']:.4f} | "
                     f"{r['jet_eff_actual']:.4f} | {r['jet_eff_pred']:.4f} | {auc} |")
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def run(args) -> None:
    from sklearn.metrics import roc_auc_score  # lazy: keeps the inference env free of sklearn
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    model, pre, extra = load_checkpoint(args.model, device)
    model.to(device)
    groups: dict[str, list] = defaultdict(list)
    for f in skim_files(args.data, splits=("test",), require_scores=True):
        groups[f.tag].append(f)
    results = {}
    for tag in sorted(groups):
        fs = groups[tag][: args.max_files] if args.max_files else groups[tag]
        print(f"{tag}: {len(fs)} files")
        r = evaluate_group(fs, model, pre, device)
        r.update({"tag": tag, "sample": fs[0].sample, "ctau": fs[0].ctau, "mpid": fs[0].mpid,
                  "lam": fs[0].lam, "nflav": fs[0].nflav, "mzp": fs[0].mzp, "mu": fs[0].mu,
                  "n_files": len(fs)})
        results[tag] = r
    (out / "summary.json").write_text(json.dumps(results, indent=1))
    print(write_markdown(results, out / "summary.md"))
    print(f"wrote {out}")
