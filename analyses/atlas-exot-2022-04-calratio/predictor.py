"""ATLAS-EXOT-2022-04 (CalRatio displaced jets) reinterpretation BDTs.

The collaboration published one scikit-learn random forest per selection
(Zenodo 12957031). Each takes per-event features of the two long-lived
particles (LLPs) and, for the W/Z selections, the vector boson, and returns
the probability of the event landing in each region of the ABCD plane
(classes 0 = none, 1..4 = A..D). The features follow the published CSV
convention: pT and eta of each LLP, ET = sqrt(pT^2 + m^2), decay position
Lxy and |z| in metres, |pdg id| of the first decay product, and the boson pT
and eta. For a single LLP (ALP), llp2 = llp1 as instructed by the authors.
The applicability guard on the mean decay positions is applied as in the
published ``recast_bdts.py``.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np

from jet_surrogate.service.predictors import Predictor, finish_summary, register

SELECTIONS = ("CR+2J", "WHS_lowET", "WHS_highET", "WALP", "ZHS_lowET", "ZHS_highET")


B_HADRONS = lambda a: ((a // 100) % 10 == 5) | ((a // 1000) % 10 == 5)
C_HADRONS = lambda a: ((a // 100) % 10 == 4) | ((a // 1000) % 10 == 4)


def child_pdgid(part, i, override=None) -> int:
    """|pdg id| of the LLP's decay product as the published CSV records it (5 for
    S -> b bbar, 21 for an ALP -> gg, 15 for taus). The published files were
    made from generator records that keep the parton-level daughters. Pythia
    8.3 with a hadron-like long-lived particle writes the hadrons of the
    decay directly, so when no parton or lepton daughter is present the
    channel is inferred from the hadrons: a b hadron -> 5, a c hadron -> 4,
    otherwise gluons (21). Set predictor.config.child_pdgid to force a value."""
    if override is not None:
        return int(override)
    pid, d1, d2 = part["pid"], part["d1"], part["d2"]
    kids = np.abs(pid[d1[i]:d2[i] + 1]) if d1[i] >= 0 else np.array([], dtype=int)
    partons = kids[(kids <= 25) & (kids > 0)]
    if len(partons):
        return int(partons[0])
    if len(kids) == 0:
        return 0
    if B_HADRONS(kids).any():
        return 5
    if C_HADRONS(kids).any():
        return 4
    return 21


def llp_features(part, llp_ids, boson_ids, child_override=None) -> np.ndarray | None:
    """Feature row for one event from its particle record (numpy columns).
    Returns None when no decaying LLP candidate is found."""
    pid, status = part["pid"], part["status"]
    d1 = part["d1"]
    x, y, z = part["x"], part["y"], part["z"]
    cand = np.flatnonzero(np.isin(np.abs(pid), llp_ids) & (d1 >= 0))
    if len(cand) == 0:                                       # fall back: any decaying particle displaced by > 1 mm
        disp = (d1 >= 0) & (np.hypot(x[np.maximum(d1, 0)], y[np.maximum(d1, 0)]) > 1.0) & (np.abs(pid) > 100)
        cand = np.flatnonzero(disp & ~np.isin(np.abs(pid), (310, 130, 3122, 3112, 3222, 3312, 3322, 3334)))
        if len(cand) == 0:
            return None
    # keep the last copy of each LLP (the one that decays to non-LLP daughters)
    keep = [i for i in cand if abs(pid[d1[i]]) != abs(pid[i])]
    cand = np.array(keep or list(cand))
    if len(cand) > 2:                                        # the two most displaced decays
        lxy = np.hypot(x[d1[cand]], y[d1[cand]])
        cand = cand[np.argsort(-lxy)[:2]]
    cand = np.sort(cand)
    if len(cand) == 1:
        cand = np.array([cand[0], cand[0]])

    def one(i):
        dx, dy, dz = x[d1[i]], y[d1[i]], z[d1[i]]
        pt, eta, m = part["pt"][i], part["eta"][i], part["mass"][i]
        return [np.hypot(dx, dy) / 1000.0, abs(dz) / 1000.0, eta, pt, np.sqrt(pt * pt + m * m),
                child_pdgid(part, i, child_override)]

    row = one(cand[0]) + one(cand[1])
    bos = np.flatnonzero(np.isin(np.abs(pid), boson_ids))
    if len(bos):
        # the last copy of the boson: the one whose daughters are not the same boson
        last = [i for i in bos if d1[i] < 0 or abs(pid[d1[i]]) != abs(pid[i])]
        b = (last or list(bos))[-1]
        row += [part["pt"][b], part["eta"][b]]
    else:
        row += [np.nan, np.nan]
    return np.array(row, dtype=np.float64)


COLUMNS = ["llp1_Lxy", "llp1_Lz", "llp1_eta", "llp1_pT", "llp1_ET", "llp1_child_pdgId",
           "llp2_Lxy", "llp2_Lz", "llp2_eta", "llp2_pT", "llp2_ET", "llp2_child_pdgId", "V_pt", "V_eta"]


@register("calratio_bdt")
class CalRatioBDT(Predictor):
    def __init__(self, analysis, device: str = "cpu"):
        super().__init__(analysis, device)
        cfg = analysis.record["predictor"].get("config", {})
        self.llp_ids = tuple(cfg.get("llp_pdgids", [35]))
        self.boson_ids = tuple(cfg.get("boson_pdgids", [23, 24]))
        self.child_override = cfg.get("child_pdgid")
        self.models_root = analysis.assets_dir
        self._models = {}

    def model(self, selection: str):
        if selection not in self._models:
            d = self.models_root / selection / "models"
            with open(d / f"{selection}_features.txt") as g:
                var = eval(g.read())        # the published feature list, a Python literal
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")           # pickles from scikit-learn 1.4.2
                clf = pickle.load(open(d / f"{selection}_model.pkl", "rb"))
            self._models[selection] = (np.load(d / f"{selection}_scaler_mean.npy"),
                                       np.load(d / f"{selection}_scaler_std.npy"), var, clf)
        return self._models[selection]

    def run(self, hepmc: Path, max_events: int, progress=None, options=None):
        import awkward as ak
        from jet_surrogate.hepmc_io import read_hepmc
        selection = (options or {}).get("selection", "CR+2J")
        if selection not in SELECTIONS:
            raise ValueError(f"unknown selection {selection!r}")
        mean, std, var, clf = self.model(selection)
        rows, n_seen, n_missing = [], 0, 0
        for batch in read_hepmc(hepmc, max_events=max_events):
            for i in range(len(batch)):
                p = batch.part[i]
                cols = {f: ak.to_numpy(p[f]) for f in ("pid", "status", "d1", "d2", "x", "y", "z", "pt", "eta", "mass")}
                row = llp_features(cols, self.llp_ids, self.boson_ids, self.child_override)
                n_seen += 1
                if row is None:
                    n_missing += 1
                else:
                    rows.append(row)
            if progress:
                progress(f"{n_seen} events read, {n_missing} without a long-lived particle")
        if not rows:
            raise ValueError("no event contained a decaying long-lived particle (set predictor.config.llp_pdgids)")
        X = np.array(rows)
        cols = {c: X[:, k] for k, c in enumerate(COLUMNS)}
        # applicability guard from the published recast_bdts.py (decays in the calorimeters)
        def mean_where(v, m):
            return float(np.nanmean(v[m])) if m.any() else np.nan
        guards = [
            mean_where(cols["llp1_Lxy"], cols["llp1_eta"] > 1.5), mean_where(cols["llp2_Lxy"], cols["llp2_eta"] > 1.5),
            mean_where(cols["llp1_Lz"], cols["llp1_eta"] < 1.5), mean_where(cols["llp2_Lz"], cols["llp2_eta"] < 1.5)]
        in_range = (all(0.25 <= g <= 16 for g in guards[:2] if not np.isnan(g))
                    and all(0.75 <= g <= 28 for g in guards[2:] if not np.isnan(g)))
        feats = np.array([cols[v.replace("W_", "V_").replace("Z_", "V_")] for v in var]).T
        if np.isnan(feats).any():
            raise ValueError(f"selection {selection} needs a W or Z boson in the record and none was found")
        proba = clf.predict_proba((feats - mean) / std)      # columns: none, A, B, C, D
        per_event = proba[:, 1] if in_range else np.full(len(proba), -1.0)
        effs = proba[:, 1:5].mean(0) if in_range else np.full(4, -1.0)
        summary = {"n_events": int(n_seen), "sr_efficiency": float(effs[0]),
                   "sr_efficiency_err": float(proba[:, 1].std(ddof=1) / np.sqrt(len(proba))) if in_range and len(proba) > 1 else 0.0,
                   "selection": selection, "n_events_without_llp": int(n_missing), "in_validity_range": bool(in_range)}
        quantities = [{"name": f"region {r} efficiency", "value": float(e), "unit": "",
                       "note": selection} for r, e in zip("BCD", effs[1:])]
        quantities.append({"name": "mean LLP decay position", "value": float(np.nanmean(np.r_[cols['llp1_Lxy'], cols['llp2_Lxy']])),
                           "unit": "m (Lxy)", "note": "BDTs valid for decays in the calorimeters" + ("" if in_range else ": OUT OF RANGE, efficiencies set to -1")})
        summary = finish_summary(self.analysis, summary, np.clip(per_event, 0, 1),
                                 objects={"label": "long-lived particles", "count": 2 * len(rows), "mean_probability": None},
                                 quantities=quantities)
        summary["sr_efficiency"] = float(effs[0])            # keep -1 when out of range
        return summary, per_event, {"region_probabilities": proba.astype(np.float32), "features": feats.astype(np.float32)}
