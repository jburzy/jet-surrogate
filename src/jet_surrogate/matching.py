"""DeltaR-based truth particle to jet association."""

import math
from typing import List


def _delta_phi(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def _delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    deta = eta1 - eta2
    dphi = _delta_phi(phi1, phi2)
    return math.sqrt(deta * deta + dphi * dphi)


def _particle_eta_phi(particle):
    m = particle.momentum
    px, py, pz = m.px, m.py, m.pz
    pt = math.sqrt(px * px + py * py)
    if pt == 0.0:
        return None, None
    p3 = math.sqrt(pt * pt + pz * pz)
    cos_theta = max(-1.0 + 1e-12, min(1.0 - 1e-12, pz / p3))
    eta = -0.5 * math.log((1.0 - cos_theta) / (1.0 + cos_theta))
    phi = math.atan2(py, px)
    return eta, phi


def match_particles_to_jet(particles, jet, dr_cut: float = 1.4) -> List:
    """Return all HEPMC particles within dr_cut of a jet axis."""
    jet_eta = jet.eta
    jet_phi = jet.phi
    matched = []
    for p in particles:
        eta, phi = _particle_eta_phi(p)
        if eta is None:
            continue
        if _delta_r(eta, phi, jet_eta, jet_phi) < dr_cut:
            matched.append(p)
    return matched
