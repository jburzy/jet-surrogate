"""Delphes ROOT -> per-jet HDF5 skim.

For every event: reco large-R jets (from Delphes R=0.4 PF jets) with their
associated Delphes tracks, truth large-R jets (from stable visible
generator particles) with their associated generator particles, and the
reco<->truth jet matching. One HDF5 per input file:

    reco_jets   [n_reco]            pt eta phi mass nsub event match n_assoc
    reco_tracks [n_reco,  MAX_TRK]  TRACK_FLOATS + TRACK_CATS + valid
    truth_jets  [n_truth]           (same layout; match -> reco row or -1)
    truth_parts [n_truth, MAX_PART] PART_FLOATS + PART_CATS + valid
    attrs: n_events, sample, ctau, mpid, seed, source

Truth large-R jets are kept down to TRUTH_PT_MIN (< the 200 GeV analysis cut)
so the surrogate can learn the reco turn-on around the threshold.
"""

from __future__ import annotations

import re
from pathlib import Path

import awkward as ak
import h5py
import numpy as np

from . import features as F
from .delphes_io import read_delphes
from .jets import (LARGE_PT_MIN, associate, jet_table, match_jets, recluster_large_r,
                   select_small_r, small_r_jets_from_particles)

TRUTH_PT_MIN = 150.0
_TAG_RE = re.compile(r"^(?P<tag>(?:signal_m(?P<mpid>[\d.]+)_ctau(?P<ctau>[\d.]+)mm"
                     r"(?:_lam(?P<lam>[\d.]+))?(?:_nf(?P<nflav>\d+))?(?:_mzp(?P<mzp>[\d.]+))?|qcd)"
                     r"(?:_mu(?P<mu>[\d.]+))?)_seed(?P<seed>\d+)$")


def parse_stem(stem: str) -> dict:
    m = _TAG_RE.match(stem)
    if not m:
        raise ValueError(f"unrecognized file stem {stem!r}")
    d = m.groupdict()
    return {"sample": "qcd" if d["tag"].startswith("qcd") else "signal", "tag": d["tag"],
            "ctau": float(d["ctau"]) if d["ctau"] else -1.0,
            "mpid": float(d["mpid"]) if d["mpid"] else -1.0, "seed": int(d["seed"]),
            "lam": float(d["lam"]) if d["lam"] else -1.0, "nflav": int(d["nflav"]) if d["nflav"] else -1,
            "mzp": float(d["mzp"]) if d["mzp"] else -1.0,
            "mu": float(d["mu"]) if d["mu"] else -1.0}


def skim_truth(part: ak.Array) -> tuple[np.ndarray, np.ndarray]:
    """Truth side only (generator particles -> truth large-R jets and padded
    particle tables), shared by the Delphes skim and the HepMC predict path.
    Returns (truth_jets, truth_parts) with ``match`` = -1."""
    truth = recluster_large_r(small_r_jets_from_particles(part), pt_min=TRUTH_PT_MIN)
    part_jet = associate(part, truth)
    truth_jets = jet_table(truth, np.full(len(truth), -1, np.int32), np.zeros(len(truth), np.int32))
    pcols, part_jet = F.particle_columns(part, part_jet, truth_jets)
    truth_jets["n_assoc"] = np.bincount(part_jet[part_jet >= 0], minlength=len(truth))
    truth_parts = F.pad_groups(part_jet, len(truth), pcols["pt"], pcols, F.MAX_PART,
                               F.PART_FLOATS, F.PART_CATS)
    return truth_jets, truth_parts


def skim_events(ev: ak.Array) -> dict:
    """Build the four tables from one awkward event batch."""
    # ---- reco: Delphes R=0.4 PF jets -> R=1.0
    reco = recluster_large_r(select_small_r(ev.jet))
    # ---- truth: stable visible particles -> R=0.4 -> R=1.0
    truth = recluster_large_r(small_r_jets_from_particles(ev.part), pt_min=TRUTH_PT_MIN)
    r2t, t2r = match_jets(reco, truth)

    # tracks -> reco jets
    trk_jet = associate(ev.trk, reco)
    reco_jets = jet_table(reco, r2t, np.zeros(len(reco), np.int32))
    tcols, trk_jet = F.track_columns(ev.trk, trk_jet, reco_jets)
    reco_jets["n_assoc"] = np.bincount(trk_jet[trk_jet >= 0], minlength=len(reco))
    reco_tracks = F.pad_groups(trk_jet, len(reco), tcols["pt"], tcols, F.MAX_TRK,
                               F.TRACK_FLOATS, F.TRACK_CATS)

    # generator particles -> truth jets
    part_jet = associate(ev.part, truth)
    truth_jets = jet_table(truth, t2r, np.zeros(len(truth), np.int32))
    pcols, part_jet = F.particle_columns(ev.part, part_jet, truth_jets)
    truth_jets["n_assoc"] = np.bincount(part_jet[part_jet >= 0], minlength=len(truth))
    truth_parts = F.pad_groups(part_jet, len(truth), pcols["pt"], pcols, F.MAX_PART,
                               F.PART_FLOATS, F.PART_CATS)
    return {"reco_jets": reco_jets, "reco_tracks": reco_tracks,
            "truth_jets": truth_jets, "truth_parts": truth_parts, "n_events": len(ev)}


def skim_file(root: str | Path, out: str | Path, chunk: int = 2000, max_events: int | None = None) -> Path:
    root, out = Path(root), Path(out)
    meta = parse_stem(root.stem)
    from .delphes_io import n_events as _n
    n_tot = _n(root) if max_events is None else min(max_events, _n(root))
    parts = {k: [] for k in ("reco_jets", "reco_tracks", "truth_jets", "truth_parts")}
    done = n_reco = n_truth = 0
    while done < n_tot:
        stop = min(done + chunk, n_tot)
        ev = read_delphes(root, stop)[done:stop]
        res = skim_events(ev)
        for k in ("reco_jets", "truth_jets"):
            res[k]["event"] += done
        # ``match`` is a row index into the other collection, local to this chunk:
        # offset it by the rows already written, or after concatenation every jet
        # past the first chunk points at an unrelated jet in another event
        m = res["reco_jets"]["match"]
        res["reco_jets"]["match"] = np.where(m >= 0, m + n_truth, -1)
        m = res["truth_jets"]["match"]
        res["truth_jets"]["match"] = np.where(m >= 0, m + n_reco, -1)
        n_reco += len(res["reco_jets"]); n_truth += len(res["truth_jets"])
        for k in parts:
            parts[k].append(res[k])
        done = stop
    tables = {k: np.concatenate(v) for k, v in parts.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5.part")
    with h5py.File(tmp, "w") as h:
        for k, v in tables.items():
            h.create_dataset(k, data=v, compression="gzip", compression_opts=4, chunks=True)
        h.attrs.update({"n_events": n_tot, "source": str(root), **meta})
    tmp.rename(out)
    return out
