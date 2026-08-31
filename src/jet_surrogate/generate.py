"""Pythia8 + Delphes event generation via the ``DelphesPythia8`` executable.

One call = one ROOT file holding the full generator record (``Particle``),
smeared tracks (``Track``), R=0.4 particle-flow jets (``Jet``) and R=0.4
truth jets (``GenJet``); see ``cards/delphes/delphes_card_ATLAS_tracks.tcl``.

The Pythia card is assembled from a base card plus per-run overrides
(event count, seed, dark-pion lifetime, dark-sector mass scale), written next
to the output so every file is reproducible from its own ``.cmnd``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CARD_DIR = REPO / "cards"
DELPHES_CARD = CARD_DIR / "delphes" / "delphes_card_ATLAS_tracks.tcl"

DARK_PION_ID = 4900111
DARK_QUARK_ID = 4900101
DARK_RHO_ID = 4900113

# Nominal benchmark (displaced-observables / DSTF unflavored): the whole dark
# sector is rescaled coherently with the dark-pion mass so the shower stays
# self-consistent (m_qv = 2 m_pid, m_rhod = 4 m_pid, Lambda = 2 m_pid).
NOMINAL_MPID = 5.0
CTAU_GRID_MM = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0)


NOMINAL_LAMBDA = 10.0          # HiddenValley:Lambda for m_pid = 5 GeV (m_pi / Lambda = 0.5)
NOMINAL_NFLAV = 1
DARK_PION_OFFDIAG_ID = 4900211
DARK_RHO_OFFDIAG_ID = 4900213


def sample_tag(sample: str, *, ctau_mm: float | None = None, mpid: float = NOMINAL_MPID,
               lam: float | None = None, nflav: int | None = None, mzp: float | None = None,
               mu: float | None = None) -> str:
    """Canonical file stem (without seed) for a sample point. Model variants
    (a non-nominal Lambda at fixed masses, or nFlav > 1) get a suffix."""
    suffix = f"_mu{mu:g}" if mu is not None else ""
    if sample == "qcd":
        return "qcd" + suffix
    if sample == "signal":
        if ctau_mm is None:
            raise ValueError("signal requires ctau_mm")
        tag = f"signal_m{mpid:g}_ctau{ctau_mm:g}mm"
        if lam is not None:
            tag += f"_lam{lam:g}"
        if nflav is not None:
            tag += f"_nf{nflav:d}"
        if mzp is not None:
            tag += f"_mzp{mzp:g}"
        return tag + suffix
    raise ValueError(f"unknown sample {sample!r}")


def pythia_overrides(sample: str, *, n_events: int, seed: int, ctau_mm: float | None, mpid: float,
                     lam: float | None = None, nflav: int | None = None,
                     mzp: float | None = None) -> list[str]:
    """Per-run Pythia settings.

    Nominal: the whole dark sector scales with m_pid / 5 GeV (Lambda = 2 m_pi,
    m_qv = Lambda, m_rho = 2 Lambda = 4 m_pi, pTminFSR = 1.1 Lambda).
    ``lam`` fixes Lambda at the given m_pid instead (m_qv = Lambda,
    pTminFSR = 1.1 Lambda); m_rho = max(2 Lambda, 2.2 m_pi) keeps the forced
    rho_D -> pi_D pi_D decay open once m_pi / Lambda >= 1.
    ``mzp`` overrides the Z' mass in GeV (nominal 1500, width kept at 1%%
    of the mass), changing the dark-quark and hence dark-hadron pT spectra
    at a fixed dark sector.
    ``nflav`` > 1 switches on more dark-quark flavours; the off-diagonal
    mesons (4900211, 4900213) copy the masses, lifetime and decays of the
    diagonal ones, and the Z' decays to all flavours.
    """
    lines = [
        "",
        "! ---- overrides written by jet_surrogate.generate ----",
        f"Main:numberOfEvents = {n_events}",
        f"Random:seed = {seed % 900000000}",
    ]
    if sample == "signal":
        scale = mpid / NOMINAL_MPID
        m_pi = NOMINAL_MPID * scale
        lam_eff = NOMINAL_LAMBDA * scale if lam is None else lam
        m_rho = 2.0 * lam_eff if lam is None else max(2.0 * lam_eff, 2.2 * m_pi)
        lines += [
            f"HiddenValley:Lambda = {lam_eff:g}",
            f"HiddenValley:pTminFSR = {1.1 * lam_eff:g}",
            f"{DARK_QUARK_ID}:m0 = {lam_eff:g}",
            f"{DARK_PION_ID}:m0 = {m_pi:g}",
            f"{DARK_RHO_ID}:m0 = {m_rho:g}",
            f"{DARK_PION_ID}:tau0 = {ctau_mm:g}",
        ]
        if mzp is not None:
            lines += [f"4900023:m0 = {mzp:g}", f"4900023:mWidth = {mzp / 100:g}"]
        if nflav is not None and nflav > 1:
            quarks = [DARK_QUARK_ID + i for i in range(nflav)]
            lines += [f"HiddenValley:nFlav = {nflav}"]
            lines += [f"{q}:m0 = {lam_eff:g}" for q in quarks[1:]]
            lines += [
                f"4900023:onIfAny = {','.join(str(q) for q in quarks)}",
                f"{DARK_PION_OFFDIAG_ID}:m0 = {m_pi:g}",
                f"{DARK_RHO_OFFDIAG_ID}:m0 = {m_rho:g}",
                f"{DARK_RHO_OFFDIAG_ID}:oneChannel = 1 1.0 0 {DARK_PION_OFFDIAG_ID} {DARK_PION_ID}",
                f"{DARK_PION_OFFDIAG_ID}:oneChannel = 1 1.0 91 1 -1",
                f"{DARK_PION_OFFDIAG_ID}:mayDecay = on",
                f"{DARK_PION_OFFDIAG_ID}:tau0 = {ctau_mm:g}",
            ]
    return lines


def write_card(sample: str, out_cmnd: Path, *, n_events: int, seed: int,
               ctau_mm: float | None = None, mpid: float = NOMINAL_MPID,
               lam: float | None = None, nflav: int | None = None, mzp: float | None = None) -> Path:
    base = CARD_DIR / "pythia" / ("signal_zprime_hv.cmnd" if sample == "signal" else "qcd_dijet.cmnd")
    text = base.read_text().rstrip() + "\n" + "\n".join(
        pythia_overrides(sample, n_events=n_events, seed=seed, ctau_mm=ctau_mm, mpid=mpid,
                         lam=lam, nflav=nflav, mzp=mzp)
    ) + "\n"
    out_cmnd.parent.mkdir(parents=True, exist_ok=True)
    out_cmnd.write_text(text)
    return out_cmnd


def generate_sample(sample: str, *, n_events: int, seed: int = 1,
                    ctau_mm: float | None = None, mpid: float = NOMINAL_MPID,
                    lam: float | None = None, nflav: int | None = None, mzp: float | None = None,
                    mu: float | None = None, pileup_library: str | Path | None = None,
                    out_dir: str | Path = "data/delphes",
                    delphes_card: Path = DELPHES_CARD, quiet: bool = False) -> Path:
    """Run DelphesPythia8 for one (sample, ctau, mass, seed). Returns the ROOT path."""
    exe = shutil.which("DelphesPythia8")
    if exe is None:
        raise RuntimeError("DelphesPythia8 not on PATH: run inside `pixi run`")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sample_tag(sample, ctau_mm=ctau_mm, mpid=mpid, lam=lam, nflav=nflav, mzp=mzp, mu=mu)}_seed{seed}"
    root = out_dir / f"{stem}.root"
    if mu is not None:
        if pileup_library is None:
            raise ValueError("--mu needs --pileup-library (a .pileup file from hepmc2pileup)")
        tpl = (CARD_DIR / "delphes" / "delphes_card_ATLAS_tracks_pileup.tcl").read_text()
        tpl = tpl.replace("JS_PILEUP_FILE", str(Path(pileup_library).resolve()))
        tpl = tpl.replace("JS_MEAN_PILEUP", f"{mu:g}")
        delphes_card = out_dir / f"{stem}.card.tcl"
        delphes_card.write_text(tpl)
    cmnd = write_card(sample, out_dir / f"{stem}.cmnd", n_events=n_events, seed=seed,
                      ctau_mm=ctau_mm, mpid=mpid, lam=lam, nflav=nflav, mzp=mzp)
    tmp = out_dir / f"{stem}.root.part"
    tmp.unlink(missing_ok=True)
    cmd = [exe, str(Path(delphes_card).resolve()), str(cmnd), str(tmp)]
    subprocess.run(cmd, check=True, cwd=out_dir,
                   stdout=subprocess.DEVNULL if quiet else None)
    tmp.rename(root)
    return root


def generate_hepmc(sample: str, *, n_events: int, seed: int = 1, ctau_mm: float | None = None,
                   mpid: float = NOMINAL_MPID, lam: float | None = None, nflav: int | None = None,
                   out_dir: str | Path = "data/hepmc", card: str | Path | None = None,
                   settings: list[str] | None = None, hepmc_version: int = 3) -> Path:
    """Standalone Pythia8 with the same card, written as HepMC3 (no detector
    simulation). This is the input format of ``jet-surrogate predict``."""
    import numpy as np
    import pythia8

    from .hepmc_io import FIELDS, write_hepmc

    out_dir = Path(out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    if card is not None:                    # any user card: stem from the card name
        stem = f"{Path(card).stem}_seed{seed}"
        text = Path(card).read_text().rstrip() + "\n" + "\n".join(
            ["", "! ---- overrides written by jet_surrogate.generate ----", f"Main:numberOfEvents = {n_events}",
             f"Random:seed = {seed % 900000000}", *(settings or [])]) + "\n"
        cmnd = out_dir / f"{stem}.cmnd"; cmnd.write_text(text)
    else:
        stem = f"{sample_tag(sample, ctau_mm=ctau_mm, mpid=mpid, lam=lam, nflav=nflav)}_seed{seed}"
        cmnd = write_card(sample, out_dir / f"{stem}.cmnd", n_events=n_events, seed=seed,
                          ctau_mm=ctau_mm, mpid=mpid, lam=lam, nflav=nflav)
    pythia = pythia8.Pythia("", False)
    pythia.readFile(str(cmnd))
    pythia.readString("Print:quiet = on")
    if not pythia.init():
        raise RuntimeError(f"Pythia init failed for {cmnd}")

    def records():
        n = 0
        while n < n_events:
            if not pythia.next():
                continue
            ev = pythia.event
            size = ev.size()
            rec = {f: np.zeros(size, dtype=np.int32 if f in ("pid", "status", "m1", "m2", "d1", "d2") else np.float32)
                   for f in FIELDS}
            for i in range(size):
                p = ev[i]
                rec["pid"][i] = p.id(); rec["status"][i] = p.statusHepMC()
                rec["m1"][i] = p.mother1(); rec["m2"][i] = p.mother2()
                rec["d1"][i] = p.daughter1(); rec["d2"][i] = p.daughter2()
                rec["charge"][i] = p.charge(); rec["mass"][i] = p.m(); rec["e"][i] = p.e()
                rec["pt"][i] = p.pT(); rec["eta"][i] = p.eta(); rec["phi"][i] = p.phi()
                rec["x"][i] = p.xProd(); rec["y"][i] = p.yProd(); rec["z"][i] = p.zProd()
            # Pythia: daughter1 == 0 means none; daughter2 < daughter1 means a single daughter
            rec["d1"][rec["d1"] <= 0] = -1
            rec["d2"] = np.where(rec["d1"] >= 0, np.maximum(rec["d2"], rec["d1"]), -1)
            rec["m1"][rec["m1"] <= 0] = -1
            yield rec
            n += 1

    path = out_dir / f"{stem}.hepmc"
    write_hepmc(path, records(), version=hepmc_version)
    return path
