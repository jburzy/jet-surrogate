"""FastJet jet reconstruction: anti-kt R=0.4 (ATLAS truth) then recluster to R=1.0."""

import math
from typing import List

import awkward as ak
import fastjet

_INVISIBLE = frozenset({12, 14, 16, 13})  # neutrinos + muons


class _Jet:
    """Minimal jet object exposing pt, eta, phi, mass."""
    __slots__ = ("pt", "eta", "phi", "mass")

    def __init__(self, pt: float, eta: float, phi: float, mass: float) -> None:
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.mass = mass


def _eta(px: float, py: float, pz: float) -> float:
    pt = math.sqrt(px * px + py * py)
    if pt == 0.0:
        return -999.0
    p = math.sqrt(pt * pt + pz * pz)
    ct = max(-1.0 + 1e-12, min(1.0 - 1e-12, pz / p))
    return -0.5 * math.log((1.0 - ct) / (1.0 + ct))


def _mass(px: float, py: float, pz: float, e: float) -> float:
    return math.sqrt(max(0.0, e * e - px * px - py * py - pz * pz))


def _ak_to_jets(ak_jets) -> List[_Jet]:
    jets = []
    for i in range(len(ak_jets)):
        px = float(ak_jets[i].px)
        py = float(ak_jets[i].py)
        pz = float(ak_jets[i].pz)
        e  = float(ak_jets[i].E)
        pt = math.sqrt(px * px + py * py)
        jets.append(_Jet(
            pt=pt,
            eta=_eta(px, py, pz),
            phi=math.atan2(py, px),
            mass=_mass(px, py, pz, e),
        ))
    return jets


def _hepmc_to_ak(particles) -> ak.Array:
    """Stable visible HEPMC particles → awkward array for fastjet."""
    px_l, py_l, pz_l, e_l = [], [], [], []
    for p in particles:
        if p.status != 1:
            continue
        if abs(p.pid) in _INVISIBLE:
            continue
        m = p.momentum
        if math.sqrt(m.px * m.px + m.py * m.py) <= 0.0:
            continue
        px_l.append(m.px)
        py_l.append(m.py)
        pz_l.append(m.pz)
        e_l.append(m.e)
    return ak.Array({"px": px_l, "py": py_l, "pz": pz_l, "E": e_l})


def reconstruct_r04_jets(particles, ptmin: float = 10.0) -> List[_Jet]:
    """Anti-kt R=0.4 truth jets from stable visible particles (ATLAS convention)."""
    arr = _hepmc_to_ak(particles)
    if len(arr) == 0:
        return []
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    jets = fastjet.ClusterSequence(arr, jetdef).inclusive_jets(min_pt=ptmin)
    return _ak_to_jets(jets)


def recluster_r10_jets(small_jets: List[_Jet], ptmin: float = 200.0) -> List[_Jet]:
    """Anti-kt R=1.0 jets formed by reclustering R=0.4 jets as inputs."""
    if not small_jets:
        return []
    arr = ak.Array({
        "px": [j.pt * math.cos(j.phi) for j in small_jets],
        "py": [j.pt * math.sin(j.phi) for j in small_jets],
        "pz": [j.pt * math.sinh(j.eta) for j in small_jets],
        "E":  [math.sqrt(j.pt ** 2 * math.cosh(j.eta) ** 2 + j.mass ** 2)
               for j in small_jets],
    })
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0)
    jets = fastjet.ClusterSequence(arr, jetdef).inclusive_jets(min_pt=ptmin)
    return _ak_to_jets(jets)
