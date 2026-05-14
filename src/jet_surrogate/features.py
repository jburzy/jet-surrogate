"""Per-particle feature extraction from HEPMC GenParticle objects."""

import math
from typing import Dict

try:
    from particle import Particle as _PDGParticle
    _HAS_PARTICLE_LIB = True
except ImportError:
    _HAS_PARTICLE_LIB = False

INT_VARS = ["pdgId", "charge", "child0PdgId", "child1PdgId"]

FLOAT_VARS = [
    "pt", "mass", "energy", "eta", "phi",
    "deta", "dphi", "dr",
    "decayVertexX", "decayVertexY", "decayVertexZ",
    "Lxy", "decayVertexDPhi", "decayVertexDEta",
    "prodVertexX", "prodVertexY", "prodVertexZ", "prodLxy",
    "child0Pt", "child0Eta", "child0Phi", "child0E", "child0M",
    "child1Pt", "child1Eta", "child1Phi", "child1E", "child1M",
]

ALL_VARS = INT_VARS + FLOAT_VARS
N_FEATURES = len(ALL_VARS)  # 32


def _charge(pdgid: int) -> int:
    if _HAS_PARTICLE_LIB:
        try:
            return int(round(float(_PDGParticle.from_pdgid(pdgid).charge)))
        except Exception:
            pass
    return 0


def _pt(px: float, py: float) -> float:
    return math.sqrt(px * px + py * py)


def _eta(px: float, py: float, pz: float) -> float:
    pt = _pt(px, py)
    if pt == 0.0:
        return math.copysign(1e9, pz)
    p3 = math.sqrt(pt * pt + pz * pz)
    ct = max(-1.0 + 1e-12, min(1.0 - 1e-12, pz / p3))
    return -0.5 * math.log((1.0 - ct) / (1.0 + ct))


def _phi(px: float, py: float) -> float:
    return math.atan2(py, px)


def _mass(px: float, py: float, pz: float, e: float) -> float:
    return math.sqrt(max(0.0, e * e - px * px - py * py - pz * pz))


def _delta_phi(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def _kinematics_from_child(child) -> tuple:
    """Return (pt, eta, phi, e, m) for a GenParticle child."""
    cm = child.momentum
    px, py, pz, e = cm.px, cm.py, cm.pz, cm.e
    return (
        _pt(px, py),
        _eta(px, py, pz),
        _phi(px, py),
        e,
        _mass(px, py, pz, e),
    )


def extract_features(particle, jet_eta: float = 0.0, jet_phi: float = 0.0) -> Dict:
    """
    Extract the full INT_VARS + FLOAT_VARS feature set from a HEPMC GenParticle.

    jet_eta / jet_phi are the parent jet axis, used to compute deta, dphi, dr.
    Decay/production vertex quantities are zero for particles without those vertices.
    Only the first two children from the end vertex are captured.
    """
    mom = particle.momentum
    px, py, pz, e = mom.px, mom.py, mom.pz, mom.e

    pt = _pt(px, py)
    eta = _eta(px, py, pz)
    phi = _phi(px, py)
    m = _mass(px, py, pz, e)
    pdg_id = particle.pid

    deta = eta - jet_eta
    dphi = _delta_phi(phi, jet_phi)
    dr = math.sqrt(deta * deta + dphi * dphi)

    # --- production vertex ---
    pv = particle.production_vertex
    if pv is not None:
        pos = pv.position
        prod_x, prod_y, prod_z = pos.x, pos.y, pos.z
        prod_lxy = math.sqrt(prod_x * prod_x + prod_y * prod_y)
    else:
        prod_x = prod_y = prod_z = prod_lxy = 0.0

    # --- decay (end) vertex ---
    ev = particle.end_vertex
    if ev is not None:
        dpos = ev.position
        dec_x, dec_y, dec_z = dpos.x, dpos.y, dpos.z
        lxy = math.sqrt(dec_x * dec_x + dec_y * dec_y)

        # Angular alignment of decay displacement with particle momentum direction
        dec_phi = _phi(dec_x, dec_y)
        decay_dphi = _delta_phi(dec_phi, phi)
        dec_eta = _eta(dec_x, dec_y, dec_z) if lxy > 0.0 else 0.0
        decay_deta = dec_eta - eta

        children = list(ev.particles_out)
    else:
        dec_x = dec_y = dec_z = lxy = decay_dphi = decay_deta = 0.0
        children = []

    # --- children ---
    child0_pdg = child1_pdg = 0
    c0_pt = c0_eta = c0_phi = c0_e = c0_m = 0.0
    c1_pt = c1_eta = c1_phi = c1_e = c1_m = 0.0

    if len(children) >= 1:
        child0_pdg = children[0].pid
        c0_pt, c0_eta, c0_phi, c0_e, c0_m = _kinematics_from_child(children[0])
    if len(children) >= 2:
        child1_pdg = children[1].pid
        c1_pt, c1_eta, c1_phi, c1_e, c1_m = _kinematics_from_child(children[1])

    return {
        "ints": {
            "pdgId": pdg_id,
            "charge": _charge(pdg_id),
            "child0PdgId": child0_pdg,
            "child1PdgId": child1_pdg,
        },
        "floats": {
            "pt": pt, "mass": m, "energy": e, "eta": eta, "phi": phi,
            "deta": deta, "dphi": dphi, "dr": dr,
            "decayVertexX": dec_x, "decayVertexY": dec_y, "decayVertexZ": dec_z,
            "Lxy": lxy,
            "decayVertexDPhi": decay_dphi, "decayVertexDEta": decay_deta,
            "prodVertexX": prod_x, "prodVertexY": prod_y, "prodVertexZ": prod_z,
            "prodLxy": prod_lxy,
            "child0Pt": c0_pt, "child0Eta": c0_eta, "child0Phi": c0_phi,
            "child0E": c0_e, "child0M": c0_m,
            "child1Pt": c1_pt, "child1Eta": c1_eta, "child1Phi": c1_phi,
            "child1E": c1_e, "child1M": c1_m,
        },
    }
