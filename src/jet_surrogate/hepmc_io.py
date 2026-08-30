"""HepMC2/3 input and HepMC3 output for the truth-level pipeline.

``read_hepmc`` turns a HepMC file into the same per-event particle record
that ``delphes_io.read_delphes`` builds from the Delphes ``Particle`` branch
(pid, status, m1, m2, d1, d2, charge, mass, e, pt, eta, phi, x, y, z with
positions in mm), so the truth side of the skim and the surrogate run on
HepMC exactly as on Delphes output. Particles are renumbered so that the
daughters of one vertex are contiguous, which is what the feature code
assumes (d1..d2 is a range).

``write_hepmc`` does the reverse for any flat generator record with
mother/daughter indices, and is used both by ``generate --format hepmc``
(standalone Pythia) and by ``export-hepmc`` (Delphes generator record), the
latter providing an exact-events cross-check of the HepMC path.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterator

import awkward as ak
import numpy as np

FIELDS = ("pid", "status", "m1", "m2", "d1", "d2", "charge", "mass", "e", "pt", "eta", "phi", "x", "y", "z")


@lru_cache(maxsize=None)
def _charge(pid: int) -> float:
    """Electric charge from Pythia's particle data (covers the Hidden Valley ids)."""
    return _pdata().charge(int(pid))


@lru_cache(maxsize=1)
def _pdata():
    import pythia8
    return pythia8.Pythia("", False).particleData


def _hm():
    from pyHepMC3 import HepMC3
    return HepMC3


def _open_reader(path: str | Path):
    """Reader for HepMC2 or HepMC3 ASCII; gzipped files are decompressed to a
    temporary file first (pyHepMC3 reads plain files only)."""
    hm = _hm()
    path = str(path)
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        import gzip
        import shutil
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".hepmc", delete=False)
        with gzip.open(path, "rb") as fi:
            shutil.copyfileobj(fi, tmp)
        tmp.close()
        path = tmp.name
    head = Path(path).open("rb").read(200)
    if b"HepMC::Version 2" in head or b"HepMC::IO_GenEvent" in head:
        return hm.ReaderAsciiHepMC2(path)
    return hm.ReaderAscii(path)


def _event_record(evt) -> dict:
    """One HepMC GenEvent -> dict of numpy columns (FIELDS), daughters contiguous."""
    hm = _hm()
    lscale = 1.0 if evt.length_unit() == hm.Units.MM else 10.0          # cm -> mm
    mscale = 1.0 if evt.momentum_unit() == hm.Units.GEV else 1e-3       # MeV -> GeV
    parts = list(evt.particles())
    # order: by production vertex (so siblings are contiguous), then original id
    def key(p):
        v = p.production_vertex()
        return (v.id() if v is not None else 0, p.id())
    parts.sort(key=key)
    new_index = {p.id(): i for i, p in enumerate(parts)}
    n = len(parts)
    cols = {f: np.zeros(n, dtype=np.int32 if f in ("pid", "status", "m1", "m2", "d1", "d2") else np.float32)
            for f in FIELDS}
    cols["m1"][:] = -1; cols["m2"][:] = -1; cols["d1"][:] = -1; cols["d2"][:] = -1
    for i, p in enumerate(parts):
        mom = p.momentum()
        px, py, pz, e = mom.px() * mscale, mom.py() * mscale, mom.pz() * mscale, mom.e() * mscale
        pt = np.hypot(px, py)
        cols["pid"][i] = p.pid(); cols["status"][i] = p.status()
        cols["charge"][i] = _charge(p.pid())
        gm = p.generated_mass()
        if gm > 0 or p.is_generated_mass_set():
            cols["mass"][i] = gm
        else:
            m2 = e * e - px * px - py * py - pz * pz
            cols["mass"][i] = np.sqrt(m2) if m2 > 0 else 0.0
        cols["e"][i] = e; cols["pt"][i] = pt
        cols["eta"][i] = np.arcsinh(pz / pt) if pt > 0 else 0.0
        cols["phi"][i] = np.arctan2(py, px)
        pv = p.production_vertex()
        if pv is not None:
            pos = pv.position()
            cols["x"][i] = pos.x() * lscale; cols["y"][i] = pos.y() * lscale; cols["z"][i] = pos.z() * lscale
            ins = [new_index[q.id()] for q in pv.particles_in()]
            if ins:
                cols["m1"][i] = min(ins); cols["m2"][i] = max(ins)
        ev = p.end_vertex()
        if ev is not None:
            outs = [new_index[q.id()] for q in ev.particles_out()]
            if outs:
                cols["d1"][i] = min(outs); cols["d2"][i] = max(outs)
    return cols


def read_hepmc(path: str | Path, max_events: int | None = None, chunk: int = 2000) -> Iterator[ak.Array]:
    """Yield awkward batches of up to ``chunk`` events, each with a ``part`` field."""
    hm = _hm()
    reader = _open_reader(path)
    batch, n = [], 0
    try:
        while max_events is None or n < max_events:
            evt = hm.GenEvent()
            reader.read_event(evt)
            if reader.failed():
                break
            batch.append(_event_record(evt)); n += 1
            if len(batch) == chunk:
                yield _to_awkward(batch); batch = []
    finally:
        reader.close()
    if batch:
        yield _to_awkward(batch)


def _to_awkward(batch: list[dict]) -> ak.Array:
    counts = np.array([len(b["pid"]) for b in batch])
    part = ak.zip({f: ak.unflatten(np.concatenate([b[f] for b in batch]), counts) for f in FIELDS})
    return ak.zip({"part": part}, depth_limit=1)


def write_hepmc(path: str | Path, events: Iterator[dict], *, weights=None) -> int:
    """Write HepMC3 (ASCII) from flat per-event records with FIELDS columns
    (positions in mm, momenta in GeV). Each particle with daughters gets an
    end vertex at the daughters' production point; daughters sharing a
    production point with several mothers join the same vertex. Returns the
    number of events written."""
    hm = _hm()
    writer = hm.WriterAscii(str(path))
    n_written = 0
    for rec in events:
        evt = hm.GenEvent(hm.Units.GEV, hm.Units.MM)
        evt.set_event_number(n_written)
        n = len(rec["pid"])
        pt = np.asarray(rec["pt"], dtype=np.float64); eta = np.asarray(rec["eta"], dtype=np.float64)
        e = np.asarray(rec["e"], dtype=np.float64); m = np.asarray(rec["mass"], dtype=np.float64)
        px = pt * np.cos(rec["phi"]); py = pt * np.sin(rec["phi"])
        with np.errstate(over="ignore", invalid="ignore"):
            pz = np.where(pt > 0, pt * np.sinh(np.clip(eta, -20, 20)),
                          np.sign(eta) * np.sqrt(np.maximum(e * e - m * m, 0.0)))   # beams: pt = 0
        gp = []
        for i in range(n):
            g = hm.GenParticle(hm.FourVector(float(px[i]), float(py[i]), float(pz[i]), float(e[i])),
                               int(rec["pid"][i]), int(rec["status"][i]))
            g.set_generated_mass(float(m[i]))
            gp.append(g)
        # production vertex per particle, keyed by (first mother, position)
        vtx_of_child: dict[int, object] = {}
        for i in range(n):
            d1, d2 = int(rec["d1"][i]), int(rec["d2"][i])
            if d1 < 0:
                continue
            d2 = max(d2, d1)
            first = d1
            v = vtx_of_child.get(first)
            if v is None:
                v = hm.GenVertex(hm.FourVector(float(rec["x"][first]), float(rec["y"][first]), float(rec["z"][first]), 0.0))
                evt.add_vertex(v)
                for c in range(d1, min(d2, n - 1) + 1):
                    if c not in vtx_of_child:
                        v.add_particle_out(gp[c]); vtx_of_child[c] = v
            v.add_particle_in(gp[i])
        # particles without a production vertex (beams, orphans) are added directly,
        # as Pythia's own HepMC writer does; a root vertex without incoming
        # particles is not representable in the ASCII format
        for i in range(n):
            if i not in vtx_of_child:
                evt.add_particle(gp[i])
        if weights is not None:
            evt.weights().append(float(weights[n_written]))
        writer.write_event(evt)
        n_written += 1
    writer.close()
    return n_written


def delphes_particle_records(path: str | Path, max_events: int | None = None, chunk: int = 2000) -> Iterator[dict]:
    """Flat FIELDS records from a Delphes file's generator record (for export-hepmc)."""
    from .delphes_io import n_events, read_delphes
    n_tot = n_events(path) if max_events is None else min(max_events, n_events(path))
    done = 0
    while done < n_tot:
        stop = min(done + chunk, n_tot)
        ev = read_delphes(path, stop)[done:stop]
        for i in range(len(ev)):
            p = ev.part[i]
            yield {f: ak.to_numpy(p[f]) for f in FIELDS}
        done = stop
