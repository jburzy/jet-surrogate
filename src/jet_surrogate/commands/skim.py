"""Skim Delphes ROOT files into per-jet HDF5 tables.

    jet-surrogate skim data/delphes/qcd_seed1.root        # -> data/skim/qcd_seed1.h5
    jet-surrogate skim --all                              # every ROOT file lacking a skim
"""

from __future__ import annotations

import time
from pathlib import Path

from ..skim import skim_file


def add_arguments(ap) -> None:
    ap.add_argument("inputs", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="skim every --indir/*.root without an output")
    ap.add_argument("--indir", default="data/delphes")
    ap.add_argument("--out", default="data/skim")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2000)


def run(args) -> None:
    inputs = list(args.inputs)
    if args.all:
        inputs += [p for p in sorted(Path(args.indir).glob("*.root"))
                   if not (Path(args.out) / f"{p.stem}.h5").exists()]
    if not inputs:
        raise SystemExit("no inputs")
    for root in inputs:
        t0 = time.time()
        out = skim_file(root, Path(args.out) / f"{root.stem}.h5", chunk=args.chunk,
                        max_events=args.max_events)
        print(f"wrote {out}  ({time.time() - t0:.0f} s)", flush=True)
