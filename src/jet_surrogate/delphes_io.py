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


def _collection(tree, name: str, fields, entry_stop=None) -> ak.Array:
    arrs = tree.arrays([f"{name}.{f}" for f in fields], entry_stop=entry_stop)
    return ak.zip({f.lower(): arrs[f"{name}.{f}"] for f in fields})


def read_delphes(path: str | Path, max_events: int | None = None) -> ak.Array:
    """Return one awkward record per event with fields
    ``part`` (GenParticle), ``trk`` (Track), ``jet`` (R=0.4 PF jets),
    ``genjet`` (R=0.4 truth jets). Field names are lower-cased Delphes names."""
    with uproot.open(path) as f:
        tree = f["Delphes"]
        part = _collection(tree, "Particle", PARTICLE_FIELDS, max_events)
        trk = _collection(tree, "Track", TRACK_FIELDS, max_events)
        jet = _collection(tree, "Jet", JET_FIELDS, max_events)
        genjet = _collection(tree, "GenJet", JET_FIELDS, max_events)
    return ak.zip({"part": part, "trk": trk, "jet": jet, "genjet": genjet}, depth_limit=1)


def n_events(path: str | Path) -> int:
    with uproot.open(path) as f:
        return f["Delphes"].num_entries
