"""Read Delphes ROOT output into awkward arrays (uproot, no ROOT needed)."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import uproot

PARTICLE_FIELDS = ("PID", "Status", "M1", "M2", "D1", "D2", "Charge", "Mass",
                   "E", "PT", "Eta", "Phi", "X", "Y", "Z")
TRACK_FIELDS = ("PID", "Charge", "PT", "Eta", "Phi", "D0", "DZ", "ErrorD0", "ErrorDZ",
                "Xd", "Yd", "Zd", "X", "Y", "Z")
JET_FIELDS = ("PT", "Eta", "Phi", "Mass")
VERTEX_FIELDS = ("X", "Y", "Z", "T")


def _collection(tree, name: str, fields, entry_stop=None) -> ak.Array:
    arrs = tree.arrays([f"{name}.{f}" for f in fields], entry_stop=entry_stop)
    return ak.zip({f.lower(): arrs[f"{name}.{f}"] for f in fields})


def read_delphes(path: str | Path, max_events: int | None = None) -> ak.Array:
    """Return one awkward record per event with fields
    ``part`` (GenParticle), ``trk`` (Track), ``jet`` (R=0.4 PF jets),
    ``genjet`` (R=0.4 truth jets), and ``pv_z`` (the primary vertex written by
    PileUpMerger, zero when the file was produced without pile-up). Field names
    are lower-cased Delphes names."""
    with uproot.open(path) as f:
        tree = f["Delphes"]
        part = _collection(tree, "Particle", PARTICLE_FIELDS, max_events)
        trk = _collection(tree, "Track", TRACK_FIELDS, max_events)
        jet = _collection(tree, "Jet", JET_FIELDS, max_events)
        genjet = _collection(tree, "GenJet", JET_FIELDS, max_events)
        n = len(part)
        if any("Vertex.Z" in k for k in tree.keys()):     # uproot exposes it as "Vertex/Vertex.Z"
            vz = _collection(tree, "Vertex", VERTEX_FIELDS, max_events)["z"]
            pv_z = ak.to_numpy(ak.fill_none(ak.firsts(vz), 0.0)).astype(np.float32)
        else:
            pv_z = np.zeros(n, dtype=np.float32)
    if np.any(pv_z != 0.0):
        part = _repair_pileup_vertex_shift(part, pv_z)
    return ak.zip({"part": part, "trk": trk, "jet": jet, "genjet": genjet, "pv_z": pv_z},
                  depth_limit=1)


def _repair_pileup_vertex_shift(part: ak.Array, pv_z: np.ndarray) -> ak.Array:
    """Put the whole generator record at the primary vertex.

    ``PileUpMerger`` displaces the hard scatter along z with
    ``candidate->Position.SetZ(z - dz0 + dz)``, but it only iterates over its
    input array, the *stable* particles, so decayed hadrons, intermediate
    partons and the dark hadrons keep the unshifted vertex. The record that
    reaches us is then internally inconsistent: a decayed hadron sits at z = 0
    while its own daughters sit at z = dz, which hands every prompt hadron a
    flight length of order dz. Delphes offers no switch for this, so the shift
    is completed here, once, and everything downstream sees a record whose
    coordinates share one origin.
    """
    n = ak.to_numpy(ak.num(part.pt))
    flat = ak.flatten(part)
    z = ak.to_numpy(flat.z).copy()
    moved = ak.to_numpy(flat.status) == 1                    # what PileUpMerger already shifted
    z[~moved] += np.repeat(pv_z, n)[~moved]
    fields = {f: flat[f] for f in flat.fields}
    fields["z"] = z
    return ak.unflatten(ak.zip(fields), n)


def n_events(path: str | Path) -> int:
    with uproot.open(path) as f:
        return f["Delphes"].num_entries
