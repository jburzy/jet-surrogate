"""Jet reconstruction shared by the reco and truth sides.

Both sides follow the same recipe (ATLAS reclustered-jet convention):

  anti-kt R=0.4 small-R jets (pT > 20 GeV, |eta| < 2.5)
      -> anti-kt R=1.0 large-R jets reclustered from the small-R jets
         (pT > 200 GeV, |eta| < 2.0)
      -> objects (tracks / truth particles) are associated to a large-R jet
         through its constituent small-R jets: an object belongs to the
         nearest small-R jet if dR < 0.4, and hence to the large-R jet that
         small-R jet was reclustered into.

Reco small-R jets are Delphes R=0.4 particle-flow jets; truth small-R jets
are clustered here from stable visible generator particles (no neutrinos,
no muons: the ATLAS truth-jet convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import awkward as ak
import numpy as np
import vector

vector.register_awkward()

SMALL_R = 0.4
LARGE_R = 1.0
SMALL_PT_MIN = 20.0
SMALL_ETA_MAX = 2.5
LARGE_PT_MIN = 200.0
LARGE_ETA_MAX = 2.0
ASSOC_DR = 0.4
MATCH_DR = 0.5

_INVISIBLE = (12, 14, 16, 13)


def delta_phi(a, b):
    return np.mod(a - b + np.pi, 2 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2)


def p4(pt, eta, phi, mass) -> ak.Array:
    return ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass}, with_name="Momentum4D")


def stable_visible(part: ak.Array) -> ak.Array:
    """Truth-jet inputs: status 1, not neutrino, not muon, pT > 0."""
    pid = np.abs(part.pid)
    keep = (part.status == 1) & (part.pt > 0)
    for p in _INVISIBLE:
        keep = keep & (pid != p)
    return part[keep]


def cluster(inputs: ak.Array, radius: float, pt_min: float):
    """anti-kt on per-event Momentum4D records. Returns (jets, constituent_index)."""
    import fastjet

    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, radius)
    cs = fastjet.ClusterSequence(inputs, jetdef)
    jets = cs.inclusive_jets(min_pt=pt_min)
    idx = cs.constituent_index(min_pt=pt_min)
    order = ak.argsort(jets.pt, ascending=False)
    return jets[order], idx[order]


def small_r_jets_from_particles(part: ak.Array) -> ak.Array:
    """Truth anti-kt R=0.4 jets from stable visible particles, ATLAS selection."""
    vis = stable_visible(part)
    jets, _ = cluster(p4(vis.pt, vis.eta, vis.phi, vis.mass), SMALL_R, SMALL_PT_MIN)
    return select_small_r(ak.zip({"pt": jets.pt, "eta": jets.eta, "phi": jets.phi, "mass": jets.mass}))


def select_small_r(jets: ak.Array) -> ak.Array:
    keep = (jets.pt > SMALL_PT_MIN) & (np.abs(jets.eta) < SMALL_ETA_MAX)
    jets = jets[keep]
    return jets[ak.argsort(jets.pt, ascending=False)]


@dataclass
class LargeRJets:
    """Per-event large-R jets plus bookkeeping to reach their constituents."""
    jets: ak.Array          # event -> jet {pt, eta, phi, mass, nsub}
    cons: ak.Array          # event -> jet -> index into the small-R jet list
    small: ak.Array         # event -> small-R jets (as passed in)

    @property
    def counts(self) -> np.ndarray:
        return ak.to_numpy(ak.num(self.jets.pt))

    @property
    def offsets(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.counts)])

    def __len__(self) -> int:
        return int(self.counts.sum())


def recluster_large_r(small: ak.Array, pt_min: float = LARGE_PT_MIN,
                      eta_max: float = LARGE_ETA_MAX) -> LargeRJets:
    """anti-kt R=1.0 jets whose inputs are the (selected) small-R jets."""
    jets, idx = cluster(p4(small.pt, small.eta, small.phi, small.mass), LARGE_R, pt_min)
    keep = np.abs(jets.eta) < eta_max
    jets, idx = jets[keep], idx[keep]
    rec = ak.zip({"pt": jets.pt, "eta": jets.eta, "phi": jets.phi, "mass": jets.mass,
                  "nsub": ak.num(idx, axis=2)})
    return LargeRJets(rec, idx, small)


def large_of_small(lr: LargeRJets) -> np.ndarray:
    """Flat (over all events) small-R jet list -> local index of the large-R
    jet each small-R jet was reclustered into (-1: none)."""
    n_small = ak.to_numpy(ak.num(lr.small.pt))
    small_off = np.concatenate([[0], np.cumsum(n_small)])
    n_cons = ak.to_numpy(ak.flatten(ak.num(lr.cons, axis=2)))         # per large jet
    event_of_large = np.repeat(np.arange(len(n_small)), lr.counts)
    local_of_large = ak.to_numpy(ak.flatten(ak.local_index(lr.jets.pt)))
    flat_small = ak.to_numpy(ak.flatten(ak.flatten(lr.cons)))
    glob_small = np.repeat(small_off[event_of_large], n_cons) + flat_small
    out = np.full(int(n_small.sum()), -1, dtype=np.int64)
    out[glob_small] = np.repeat(local_of_large, n_cons)
    return out


def associate(objs: ak.Array, lr: LargeRJets, dr_max: float = ASSOC_DR) -> np.ndarray:
    """Global large-R jet id (row in the flattened jet list) for every object,
    or -1. Flat numpy array aligned with ``ak.flatten(objs)``."""
    n_obj = ak.to_numpy(ak.num(objs.pt))
    n_small = ak.to_numpy(ak.num(lr.small.pt))
    small_off = np.concatenate([[0], np.cumsum(n_small)])
    pairs = ak.cartesian({"o": objs, "j": lr.small}, axis=1, nested=True)
    dr = delta_r(pairs.o.eta, pairs.o.phi, pairs.j.eta, pairs.j.phi)
    imin = ak.argmin(dr, axis=2, keepdims=True)
    drmin = ak.to_numpy(ak.fill_none(ak.flatten(ak.flatten(dr[imin], axis=2)), np.inf))
    small_idx = ak.to_numpy(ak.fill_none(ak.flatten(ak.flatten(imin, axis=2)), 0))
    event_of_obj = np.repeat(np.arange(len(n_obj)), n_obj)
    ok = (drmin < dr_max) & (n_small[event_of_obj] > 0)
    los = large_of_small(lr)
    g = np.minimum(small_off[event_of_obj] + small_idx, max(len(los) - 1, 0))
    large_local = np.where(ok, los[g] if len(los) else -1, -1)
    glob = lr.offsets[event_of_obj] + large_local
    return np.where(large_local >= 0, glob, -1)


def match_jets(a: LargeRJets, b: LargeRJets, dr_max: float = MATCH_DR) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one dR matching between two large-R jet collections in the same
    events, greedy in descending pT of ``a``. Returns (a_to_b, b_to_a) as
    global-row indices (-1 when unmatched)."""
    a_off, b_off = a.offsets, b.offsets
    a_pt = ak.to_numpy(ak.flatten(a.jets.pt)); a_eta = ak.to_numpy(ak.flatten(a.jets.eta))
    a_phi = ak.to_numpy(ak.flatten(a.jets.phi))
    b_eta = ak.to_numpy(ak.flatten(b.jets.eta)); b_phi = ak.to_numpy(ak.flatten(b.jets.phi))
    a2b = np.full(len(a_pt), -1, dtype=np.int64)
    b2a = np.full(len(b_eta), -1, dtype=np.int64)
    for ev in range(len(a.counts)):
        ia = np.arange(a_off[ev], a_off[ev + 1])
        ib = np.arange(b_off[ev], b_off[ev + 1])
        if len(ia) == 0 or len(ib) == 0:
            continue
        ia = ia[np.argsort(-a_pt[ia])]
        used = np.zeros(len(ib), dtype=bool)
        for i in ia:
            dr = delta_r(a_eta[i], a_phi[i], b_eta[ib], b_phi[ib])
            dr[used] = np.inf
            k = int(np.argmin(dr))
            if dr[k] < dr_max:
                a2b[i] = ib[k]; b2a[ib[k]] = i; used[k] = True
    return a2b, b2a


def jet_table(lr: LargeRJets, match: np.ndarray, n_assoc: np.ndarray) -> np.ndarray:
    """Flat structured array, one row per large-R jet."""
    dtype = np.dtype([("pt", "f4"), ("eta", "f4"), ("phi", "f4"), ("mass", "f4"),
                      ("nsub", "i4"), ("event", "i4"), ("match", "i4"), ("n_assoc", "i4")])
    out = np.zeros(len(lr), dtype=dtype)
    for f in ("pt", "eta", "phi", "mass", "nsub"):
        out[f] = ak.to_numpy(ak.flatten(lr.jets[f]))
    out["event"] = np.repeat(np.arange(len(lr.counts)), lr.counts)
    out["match"] = match
    out["n_assoc"] = n_assoc
    return out
