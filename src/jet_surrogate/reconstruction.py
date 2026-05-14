"""FastJet jet reconstruction: anti-kt R=0.4 (ATLAS truth) then recluster to R=1.0."""

import math
from typing import List

import numpy as np
from pyjet import cluster

_DTYPE_PTEPM = np.dtype([("pT", "f8"), ("eta", "f8"), ("phi", "f8"), ("mass", "f8")])

# PDG IDs excluded from ATLAS truth jet inputs: neutrinos and muons
_INVISIBLE = frozenset({12, 14, 16, 13})


def _pt(px: float, py: float) -> float:
    return math.sqrt(px * px + py * py)


def _eta(px: float, py: float, pz: float) -> float:
    pt = _pt(px, py)
    if pt == 0.0:
        return math.copysign(1e9, pz)
    p3 = math.sqrt(pt * pt + pz * pz)
    cos_theta = pz / p3
    cos_theta = max(-1.0 + 1e-12, min(1.0 - 1e-12, cos_theta))
    return -0.5 * math.log((1.0 - cos_theta) / (1.0 + cos_theta))


def _phi(px: float, py: float) -> float:
    return math.atan2(py, px)


def _mass(px: float, py: float, pz: float, e: float) -> float:
    return math.sqrt(max(0.0, e * e - px * px - py * py - pz * pz))


def _particles_to_array(particles) -> np.ndarray:
    """Build pyjet input array from stable, visible HEPMC particles."""
    rows = []
    for p in particles:
        if p.status != 1:
            continue
        if abs(p.pid) in _INVISIBLE:
            continue
        m = p.momentum
        pt = _pt(m.px, m.py)
        if pt <= 0.0:
            continue
        rows.append((
            pt,
            _eta(m.px, m.py, m.pz),
            _phi(m.px, m.py),
            _mass(m.px, m.py, m.pz, m.e),
        ))
    if not rows:
        return np.empty(0, dtype=_DTYPE_PTEPM)
    return np.array(rows, dtype=_DTYPE_PTEPM)


def reconstruct_r04_jets(particles, ptmin: float = 10.0) -> List:
    """Anti-kt R=0.4 truth jets from stable visible particles (ATLAS convention)."""
    arr = _particles_to_array(particles)
    if len(arr) == 0:
        return []
    return cluster(arr, R=0.4, p=-1).inclusive_jets(ptmin=ptmin)


def recluster_r10_jets(small_jets, ptmin: float = 200.0) -> List:
    """Anti-kt R=1.0 jets formed by reclustering R=0.4 jets as inputs."""
    if not small_jets:
        return []
    arr = np.array(
        [(j.pt, j.eta, j.phi, j.mass) for j in small_jets],
        dtype=_DTYPE_PTEPM,
    )
    return cluster(arr, R=1.0, p=-1).inclusive_jets(ptmin=ptmin)
