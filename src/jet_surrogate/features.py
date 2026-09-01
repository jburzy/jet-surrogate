"""Per-object feature tables and padded (jet, slot) arrays.

Reco side  : Delphes tracks -> TRACK_FLOATS (+ TRACK_CATS)  -> tagger input
Truth side : generator particles (stable, or decayed physical particles:
             SM hadrons, taus, dark hadrons) -> PART_FLOATS
             (+ PART_CATS, embedded PDG ids) -> surrogate input

Raw values are stored on disk; ``model_space()`` applies the fixed
transforms (logs, symlogs) that the networks consume, so training and
inference share one definition.
"""

from __future__ import annotations

from functools import lru_cache

import awkward as ak
import numpy as np

from .jets import delta_phi, delta_r

MAX_TRK = 100
MAX_PART = 150
TRK_PT_MIN = 0.5
TRK_ETA_MAX = 2.5
PART_PT_MIN = 0.5

# ---------------------------------------------------------------- tracks
TRACK_FLOATS = ["pt", "ptrel", "deta", "dphi", "dr", "d0", "z0", "sigma_d0", "sigma_z0", "charge"]
TRACK_CATS = ["type"]           # 0 hadron, 1 electron, 2 muon
TRACK_CAT_SIZES = [3]

# ---------------------------------------------------------------- truth particles
PART_FLOATS = [
    "pt", "ptrel", "energy", "mass", "deta", "dphi", "dr", "charge", "status",
    "prod_lxy", "prod_z", "decay_lxy", "decay_z", "decay_len",
    "decay_dphi", "decay_deta", "has_decay", "n_children",
    "c0_pt", "c0_deta", "c0_dphi", "c0_mass",
    "c1_pt", "c1_deta", "c1_dphi", "c1_mass",
]
PART_CATS = ["pdgid", "c0_pdgid", "c1_pdgid"]   # embedded via a vocabulary

# transform applied before normalization: name -> (kind, scale)
_LOG = ("log", 1.0)
_SYMLOG_UM = ("symlog", 0.01)     # mm quantities: sign(x) log1p(|x| / 10 um)
_SYMLOG_MM = ("symlog", 1.0)
TRANSFORMS = {
    "pt": _LOG, "ptrel": _LOG, "energy": _LOG, "c0_pt": ("log0", 1.0), "c1_pt": ("log0", 1.0),
    "d0": _SYMLOG_UM, "z0": _SYMLOG_MM, "sigma_d0": _LOG, "sigma_z0": _LOG,
    "prod_lxy": _SYMLOG_UM, "prod_z": _SYMLOG_MM, "decay_lxy": _SYMLOG_UM,
    "decay_z": _SYMLOG_MM, "decay_len": _SYMLOG_UM,
}


def model_space(name: str, x: np.ndarray) -> np.ndarray:
    kind, scale = TRANSFORMS.get(name, ("lin", 1.0))
    x = np.asarray(x, dtype=np.float32)
    if kind == "log":
        return np.log(np.maximum(x, 1e-6))
    if kind == "log0":                      # 0 stays 0 (absent child)
        return np.where(x > 0, np.log(np.maximum(x, 1e-6)), 0.0).astype(np.float32)
    if kind == "symlog":
        return (np.sign(x) * np.log1p(np.abs(x) / scale)).astype(np.float32)
    return x


def d0_resolution_mm(pt):
    """ATLAS-like d0 resolution table (trackResolutionATLAS.tcl), fallback when
    Delphes does not fill ErrorD0."""
    edges = [0.0, 1.0, 2.0, 4.0, 6.0, 10.0, 30.0, np.inf]
    vals = [0.08, 0.045, 0.028, 0.020, 0.016, 0.014, 0.0115]
    return np.asarray(vals)[np.clip(np.digitize(pt, edges) - 1, 0, 6)]


def z0_resolution_mm(pt):
    edges = [0.0, 1.0, 2.0, 4.0, 6.0, 15.0, 30.0, np.inf]
    vals = [0.140, 0.096, 0.075, 0.058, 0.055, 0.050, 0.047]
    return np.asarray(vals)[np.clip(np.digitize(pt, edges) - 1, 0, 6)]


@lru_cache(maxsize=None)
def pdg_charge(pid: int) -> float:
    """Electric charge from Pythia's particle table. Delphes writes Charge = -999
    for ids it does not know (every dark-sector state), so truth-particle
    charges are always derived from the PDG id, on the Delphes and the HepMC
    path alike."""
    return float(_pythia_particle_data().charge(int(pid)))


@lru_cache(maxsize=1)
def _pythia_particle_data():
    import pythia8
    return pythia8.Pythia("", False).particleData


def charges_from_pid(pid: np.ndarray) -> np.ndarray:
    ids, inv = np.unique(pid, return_inverse=True)
    return np.array([pdg_charge(int(i)) for i in ids], dtype=np.float32)[inv]


def primary_vertex_z(part) -> np.ndarray:
    """Per-event z of the primary interaction, the median production z of the
    prompt particles (those made within 1 mm of the beamline).

    Every longitudinal quantity is measured from this point, so the features do
    not depend on where along the beamline the event was placed. Without that,
    a model simulated with pile-up and the same model simulated without it
    would not give the same surrogate inputs, and a generator file uploaded to
    the service would be scored out of distribution.
    """
    import awkward as ak
    n = ak.to_numpy(ak.num(part.pt))
    p = ak.flatten(part)
    z = ak.to_numpy(p.z)
    prompt = np.hypot(ak.to_numpy(p.x), ak.to_numpy(p.y)) < 1.0
    ev = np.repeat(np.arange(len(n)), n)
    pv = np.zeros(len(n), dtype=np.float32)
    if prompt.any():
        order = np.lexsort((z[prompt], ev[prompt]))
        e, zz = ev[prompt][order], z[prompt][order]
        _, start, cnt = np.unique(e, return_index=True, return_counts=True)
        pv[np.unique(e)] = zz[start + cnt // 2]
    return pv


def padded_dtype(floats, cats):
    return np.dtype([(n, "f4") for n in floats] + [(n, "i4") for n in cats] + [("valid", "?")])


def pad_groups(jet_id: np.ndarray, n_jets: int, sort_key: np.ndarray, columns: dict,
               max_n: int, floats, cats) -> np.ndarray:
    """Scatter flat per-object columns into a (n_jets, max_n) structured array,
    ordering objects within a jet by descending ``sort_key`` and truncating."""
    out = np.zeros((n_jets, max_n), dtype=padded_dtype(floats, cats))
    sel = jet_id >= 0
    jid, key = jet_id[sel], sort_key[sel]
    order = np.lexsort((-key, jid))
    jid = jid[order]
    starts = np.searchsorted(jid, jid, side="left")
    pos = np.arange(len(jid)) - starts
    keep = pos < max_n
    rows, cols = jid[keep], pos[keep]
    src = np.flatnonzero(sel)[order][keep]
    for name in list(floats) + list(cats):
        out[name][rows, cols] = columns[name][src]
    out["valid"][rows, cols] = True
    return out


# ---------------------------------------------------------------- reco tracks
def track_columns(trk: ak.Array, jet_id: np.ndarray, jets: np.ndarray,
                  pv_z: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    """Flat per-track feature columns relative to the associated large-R jet.
    Tracks failing the selection get jet_id -1. ``pv_z`` (per event) is
    subtracted from the longitudinal impact parameter, so z0 is measured from
    the primary vertex rather than from the origin, as in the experiments."""
    t = ak.flatten(trk)
    pt = ak.to_numpy(t.pt); eta = ak.to_numpy(t.eta); phi = ak.to_numpy(t.phi)
    d0 = ak.to_numpy(t.d0); dz = ak.to_numpy(t.dz)
    ed0 = ak.to_numpy(t.errord0); edz = ak.to_numpy(t.errordz)
    if pv_z is not None:
        n_trk = ak.to_numpy(ak.num(trk.pt))
        dz = dz - pv_z[np.repeat(np.arange(len(n_trk)), n_trk)]
    q = ak.to_numpy(t.charge).astype(np.float32); pid = np.abs(ak.to_numpy(t.pid))
    ok = (pt > TRK_PT_MIN) & (np.abs(eta) < TRK_ETA_MAX) & (jet_id >= 0)
    jid = np.where(ok, jet_id, -1)
    j = jets[np.maximum(jid, 0)]
    jphi, jeta, jpt = j["phi"], j["eta"], j["pt"]
    sig_d0 = np.where(ed0 > 0, ed0, d0_resolution_mm(pt))
    sig_z0 = np.where(edz > 0, edz, z0_resolution_mm(pt))
    # jet-signed d0: sign of (PCA vector . jet transverse direction);
    # Delphes' PCA vector is d0 (sin phi, -cos phi), so the sign is d0 sin(phi - phi_jet)
    sign = np.where(d0 * np.sin(phi - jphi) >= 0, 1.0, -1.0)
    cols = {
        "pt": pt, "ptrel": pt / np.maximum(jpt, 1e-6),
        "deta": eta - jeta, "dphi": delta_phi(phi, jphi), "dr": delta_r(eta, phi, jeta, jphi),
        "d0": np.abs(d0) * sign, "z0": dz, "sigma_d0": sig_d0, "sigma_z0": sig_z0,
        "charge": q, "type": np.where(pid == 11, 1, np.where(pid == 13, 2, 0)),
    }
    return {k: np.asarray(v, dtype=np.float32) if k != "type" else v for k, v in cols.items()}, jid


# ---------------------------------------------------------------- truth particles
def particle_columns(part: ak.Array, jet_id: np.ndarray, jets: np.ndarray,
                     pv_z: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    """Flat per-particle feature columns (status 1 or 2, pT > PART_PT_MIN, not
    a neutrino) relative to the associated truth large-R jet. ``pv_z`` is the
    per-event primary vertex; every z is referred to it, so that production and
    decay vertices, and hence flight lengths, do not depend on where the event
    sits along the beamline."""
    n = ak.to_numpy(ak.num(part.pt))
    off = np.concatenate([[0], np.cumsum(n)])
    p = ak.flatten(part)
    pid = ak.to_numpy(p.pid); apid = np.abs(pid)
    status = ak.to_numpy(p.status)
    pt = ak.to_numpy(p.pt); eta = ak.to_numpy(p.eta); phi = ak.to_numpy(p.phi)
    e = ak.to_numpy(p.e); m = ak.to_numpy(p.mass)
    q = charges_from_pid(pid)               # never Delphes' Charge (-999 for dark-sector ids)
    x = ak.to_numpy(p.x); y = ak.to_numpy(p.y); z = ak.to_numpy(p.z)
    d1 = ak.to_numpy(p.d1); d2 = ak.to_numpy(p.d2)
    ev = np.repeat(np.arange(len(n)), n)
    if pv_z is not None:
        z = z - pv_z[ev]
    # Delphes stores HepMC-style status for SM particles (1 stable, 2 decayed
    # hadron/tau) but keeps raw Pythia codes for dark hadrons: 81-89 from
    # fragmentation, 91-99 from decays. Keep every stable particle and every
    # decayed physical particle, SM or dark.
    dark = apid >= 4900000
    decayed = (status == 2) | (dark & (status >= 81) & (status <= 99))
    ok = ((status == 1) | decayed) & (pt > PART_PT_MIN) & (jet_id >= 0)
    ok &= ~np.isin(apid, (12, 14, 16))
    jid = np.where(ok, jet_id, -1)
    j = jets[np.maximum(jid, 0)]

    # children: D1..D2 are event-local indices into the Particle list
    has = decayed & (d1 >= 0)
    g1 = np.where(has, off[ev] + d1, 0)
    nchild = np.where(has, np.where(d2 >= d1, d2 - d1 + 1, 1), 0)
    has2 = has & (nchild >= 2)
    g2 = np.where(has2, g1 + 1, 0)
    # decay vertex = production vertex of the first child
    dx = np.where(has, x[g1], 0.0); dy = np.where(has, y[g1], 0.0); dz = np.where(has, z[g1], 0.0)
    fx, fy, fz = dx - x, dy - y, dz - z
    flight_t = np.hypot(fx, fy)
    dec_phi = np.where(has & (flight_t > 0), delta_phi(np.arctan2(fy, fx), phi), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dec_eta = np.where(has & (flight_t > 0), np.arcsinh(fz / np.maximum(flight_t, 1e-9)) - eta, 0.0)

    def child(g, ok_c):
        return {
            "pt": np.where(ok_c, pt[g], 0.0),
            "deta": np.where(ok_c, eta[g] - eta, 0.0),
            "dphi": np.where(ok_c, delta_phi(phi[g], phi), 0.0),
            "mass": np.where(ok_c, m[g], 0.0),
            "pdgid": np.where(ok_c, pid[g], 0),
        }
    c0, c1 = child(g1, has), child(g2, has2)
    cols = {
        "pt": pt, "ptrel": pt / np.maximum(j["pt"], 1e-6), "energy": e, "mass": m,
        "deta": eta - j["eta"], "dphi": delta_phi(phi, j["phi"]), "dr": delta_r(eta, phi, j["eta"], j["phi"]),
        "charge": q, "status": status.astype(np.float32),
        "prod_lxy": np.hypot(x, y), "prod_z": z,
        "decay_lxy": np.hypot(dx, dy), "decay_z": dz, "decay_len": np.sqrt(fx**2 + fy**2 + fz**2),
        "decay_dphi": dec_phi, "decay_deta": dec_eta, "has_decay": has.astype(np.float32),
        "n_children": nchild.astype(np.float32),
        "c0_pt": c0["pt"], "c0_deta": c0["deta"], "c0_dphi": c0["dphi"], "c0_mass": c0["mass"],
        "c1_pt": c1["pt"], "c1_deta": c1["deta"], "c1_dphi": c1["dphi"], "c1_mass": c1["mass"],
        "pdgid": pid, "c0_pdgid": c0["pdgid"], "c1_pdgid": c1["pdgid"],
    }
    return {k: (np.asarray(v, dtype=np.float32) if k not in PART_CATS else np.asarray(v, dtype=np.int32))
            for k, v in cols.items()}, jid
