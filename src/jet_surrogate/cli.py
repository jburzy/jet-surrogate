"""``jet-surrogate``: the single entry point for every step of the chain.

    jet-surrogate generate        Pythia8 + Delphes -> ROOT
    jet-surrogate skim            ROOT -> per-jet HDF5 tables
    jet-surrogate train-tagger    track transformer (GPU)
    jet-surrogate apply-tagger    score every reco jet, derive surrogate labels
    jet-surrogate train-surrogate truth-particle transformer (GPU)
    jet-surrogate evaluate        surrogate closure -> results/summary.json
    jet-surrogate visualize       every figure, from stored outputs only
    jet-surrogate predict         HepMC (or skim) -> surrogate -> SR efficiency of a new model
    jet-surrogate export-hepmc    Delphes generator record -> HepMC3 (cross-checks, examples)
    jet-surrogate serve           web service: upload HepMC, get the SR efficiency
    jet-surrogate worker          service worker (runs queued jobs)

Compute steps never plot; ``visualize`` never computes.
"""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "generate": "jet_surrogate.commands.generate",
    "skim": "jet_surrogate.commands.skim",
    "train-tagger": "jet_surrogate.commands.train_tagger",
    "apply-tagger": "jet_surrogate.commands.apply_tagger",
    "train-surrogate": "jet_surrogate.commands.train_surrogate",
    "evaluate": "jet_surrogate.commands.evaluate",
    "visualize": "jet_surrogate.commands.visualize",
    "predict": "jet_surrogate.commands.predict",
    "export-hepmc": "jet_surrogate.commands.export_hepmc",
    "serve": "jet_surrogate.commands.serve",
    "worker": "jet_surrogate.commands.worker",
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="jet-surrogate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, modname in COMMANDS.items():
        mod = importlib.import_module(modname)
        p = sub.add_parser(name, help=mod.__doc__.strip().splitlines()[0], description=mod.__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        mod.add_arguments(p)
        p.set_defaults(run=mod.run)
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        # training commands forward every unrecognized argument, in order, to LightningCLI
        if hasattr(args, "overrides"):
            args.overrides = unknown
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return int(args.run(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
