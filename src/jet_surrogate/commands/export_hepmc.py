"""Write the generator record of a Delphes file to HepMC3.

    jet-surrogate export-hepmc data/delphes/signal_m5_ctau1mm_seed1.root [--max-events 1000] [--out data/hepmc]

Used to cross-check the HepMC entry point (``predict --hepmc``) against the
Delphes path on identical events, and to produce example inputs for users.
"""

from __future__ import annotations

from pathlib import Path


def add_arguments(ap) -> None:
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", default="data/hepmc")
    ap.add_argument("--max-events", type=int, default=None)


def run(args) -> None:
    from ..hepmc_io import delphes_particle_records, write_hepmc
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for root in args.inputs:
        dst = out / f"{root.stem}.hepmc"
        n = write_hepmc(dst, delphes_particle_records(root, args.max_events))
        print(f"wrote {dst} ({n} events)", flush=True)
