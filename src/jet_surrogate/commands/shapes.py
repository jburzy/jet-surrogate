"""Distribution study of the model variants (Lambda and nFlav scans).

    jet-surrogate shapes [--out results]

Why does the detector-level tagger efficiency change with m_pi/Lambda while
the surrogate prediction stays flat? This histograms, for the nominal test
seeds and every variant, the tagger-side observables (tracks in reco jets,
tagger logit) and the surrogate-side observables, across the Lambda and
nFlav scans at the reference lifetime and across the lifetime grid of the
nominal model, (truth particles, dark
hadrons in truth jets), plus per-jet efficiencies versus a few of them.
Writes results/shapes.json; figures come from ``visualize --only shapes``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..data import load_table, skim_files
from ..generate import NOMINAL_MPID
from ..metrics import efficiency

PT_MIN = 200.0
DARK_PID = 4900000


def add_arguments(ap) -> None:
    ap.add_argument("--data", default="data/skim")
    ap.add_argument("--out", default="results")


def _hist(x, edges, w=None):
    c, _ = np.histogram(np.asarray(x, float), edges, weights=w)
    n = c.sum()
    return {"edges": np.asarray(edges, float).round(6).tolist(),
            "density": (c / max(n, 1) / np.diff(edges)).tolist(), "n": int(n)}


def _eff_vs(x, passed, edges):
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= 50:
            e, err = efficiency(passed[sel])
            rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(sel.sum()),
                         "eff": e, "err": err})
    return rows


def group_shapes(files) -> dict:
    reco, rx = load_table(files, "reco", with_scores=("reco_logit", "reco_pass"))
    truth, tx = load_table(files, "truth", with_scores=("truth_label",))
    rsel = reco.jets["pt"] > PT_MIN
    tsel = truth.jets["pt"] > PT_MIN
    reco, truth = reco.subset(rsel), truth.subset(tsel)
    logit, rpass = rx["reco_logit"][rsel], rx["reco_pass"][rsel].astype(bool)
    label = tx["truth_label"][tsel].astype(bool)

    trk, part = reco.objs, truth.objs
    tv, pv = trk["valid"], part["valid"]
    # ---- tagger side (per reco jet / per track)
    d0sig = np.abs(trk["d0"]) / np.maximum(trk["sigma_d0"], 1e-6)
    disp = tv & (d0sig > 3.0)
    n_trk = tv.sum(1); n_disp = disp.sum(1)
    # ---- surrogate side (per truth jet / per particle)
    dark = pv & (np.abs(part["pdgid"]) >= DARK_PID)
    decayed_sm = pv & (part["has_decay"] > 0.5) & (np.abs(part["pdgid"]) < DARK_PID)
    charged = pv & (np.abs(part["charge"]) > 0.1) & (part["has_decay"] < 0.5)
    n_dark = dark.sum(1)

    hists = {
        "trk_n": _hist(n_trk, np.arange(-0.5, 90.5, 2)),
        "trk_n_displaced": _hist(n_disp, np.arange(-0.5, 60.5, 2)),
        "trk_frac_displaced": _hist(n_disp / np.maximum(n_trk, 1), np.linspace(0, 1, 26)),
        "trk_d0sig": _hist(d0sig[tv], np.geomspace(0.03, 3000, 51)),
        "trk_absd0": _hist(np.abs(trk["d0"][tv]), np.geomspace(1e-3, 30, 51)),
        "trk_pt": _hist(trk["pt"][tv], np.geomspace(0.5, 300, 51)),
        "reco_logit": _hist(logit, np.linspace(-12, 14, 53)),
        "reco_jet_mass": _hist(reco.jets["mass"], np.linspace(0, 500, 51)),
        "part_n": _hist(pv.sum(1), np.arange(-0.5, 150.5, 4)),
        "part_n_charged": _hist(charged.sum(1), np.arange(-0.5, 80.5, 2)),
        "dark_n": _hist(n_dark, np.arange(-0.5, 40.5, 1)),
        "dark_pt": _hist(part["pt"][dark], np.geomspace(1, 700, 51)),
        "dark_ptfrac": _hist((np.where(dark, part["ptrel"], 0.0)).sum(1), np.linspace(0, 1.2, 31)),
        "dark_decay_len": _hist(part["decay_len"][dark], np.geomspace(1e-3, 100, 51)),
        "dark_nchild": _hist(part["n_children"][dark], np.arange(-0.5, 12.5, 1)),
        "sm_decayed_n": _hist(decayed_sm.sum(1), np.arange(-0.5, 60.5, 2)),
        "truth_jet_mass": _hist(truth.jets["mass"], np.linspace(0, 500, 51)),
    }
    effs = {
        "reco_eff_vs_n_displaced": _eff_vs(n_disp, rpass, np.arange(-0.5, 40.5, 2)),
        "reco_eff_vs_frac_displaced": _eff_vs(n_disp / np.maximum(n_trk, 1), rpass, np.linspace(0, 1, 21)),
        "truth_eff_vs_dark_n": _eff_vs(n_dark, label, np.arange(-0.5, 30.5, 2)),
        "truth_eff_vs_part_n": _eff_vs(pv.sum(1), label, np.arange(-0.5, 150.5, 10)),
    }
    return {"n_reco_jets": int(len(reco)), "n_truth_jets": int(len(truth)),
            "jet_eff_actual": float(rpass.mean()), "jet_eff_label": float(label.mean()),
            "hists": hists, "effs": effs,
            "means": {"n_trk": float(n_trk.mean()), "n_disp": float(n_disp.mean()),
                      "n_dark": float(n_dark.mean()), "n_part": float(pv.sum(1).mean()),
                      "dark_pt_mean": float(part["pt"][dark].mean()),
                      "dark_nchild_mean": float(part["n_children"][dark].mean())}}


def run(args) -> None:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list] = defaultdict(list)
    for f in skim_files(args.data, samples=("signal",), require_scores=True):
        if f.mpid != NOMINAL_MPID or f.split != "test":
            continue
        if f.variant and abs(f.ctau - 0.1) < 1e-9:
            groups[f.tag].append(f)          # Lambda / nFlav scan at the reference lifetime
        elif not f.variant:
            groups[f.tag].append(f)          # nominal model, every lifetime
    results = {}
    for tag in sorted(groups):
        print(f"{tag}: {len(groups[tag])} files", flush=True)
        r = group_shapes(groups[tag])
        f0 = groups[tag][0]
        r.update({"tag": tag, "ctau": f0.ctau, "mpid": f0.mpid, "lam": f0.lam, "nflav": f0.nflav, "mzp": f0.mzp})
        results[tag] = r
        print(f"  jets {r['n_reco_jets']}, eff {r['jet_eff_actual']:.3f}, "
              f"<n_trk> {r['means']['n_trk']:.1f}, <n_disp> {r['means']['n_disp']:.1f}, "
              f"<n_dark> {r['means']['n_dark']:.1f}, <dark pT> {r['means']['dark_pt_mean']:.0f}", flush=True)
    (out / "shapes.json").write_text(json.dumps(results))
    print(f"wrote {out}/shapes.json")
