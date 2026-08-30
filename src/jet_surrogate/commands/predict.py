"""Predict the signal-region efficiency of a new model from generator truth
with the predictor of one library analysis.

    jet-surrogate predict --analysis <id> --hepmc events.hepmc [more.hepmc ...] [--option name=value]
    jet-surrogate predict --analysis <id> --skim data/skim/<sample>.h5   (analyses whose predictor supports skims)

This is the reinterpretation entry point: no detector simulation. The
analysis directory (analyses/<id>/) supplies the model and the code that
turns the event record into a prediction; this command only dispatches.
Writes <out>/<stem>.json (summary: n_events, sr_efficiency +- error, any
analysis-specific quantities) and <out>/<stem>.h5 (per-event probabilities
plus whatever extra arrays the predictor returns).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def add_arguments(ap) -> None:
    ap.add_argument("--analysis", required=True, help="analysis id from the library (analyses/<id>/analysis.yaml)")
    ap.add_argument("--hepmc", nargs="*", type=Path, default=[], help="HepMC2/3 files")
    ap.add_argument("--skim", nargs="*", type=Path, default=[],
                    help="skim HDF5 files, for predictors that implement run_skim (closure studies)")
    ap.add_argument("--model", default=None, help="model file overriding the one named in the record (e.g. a .ckpt)")
    ap.add_argument("--out", default="results/predict")
    ap.add_argument("--option", action="append", default=[], metavar="NAME=VALUE",
                    help="per-analysis option (see the analysis record), e.g. --option selection=CR+2J")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--device", default=None)


def run(args) -> None:
    if not args.hepmc and not args.skim:
        raise SystemExit("give --hepmc and/or --skim inputs")
    from ..service import registry
    from ..service.predictors import call_run
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    a = registry.load(strict=False).get(args.analysis)
    if a is None:
        raise SystemExit(f"unknown analysis '{args.analysis}' (see analyses/)")
    if args.model:
        a.model_path = Path(args.model).resolve()
    if a.record.get("assets"):
        a.fetch_assets()
    predictor = a.predictor(args.device or "cpu")
    options = dict(kv.split("=", 1) for kv in args.option)
    for o in a.record.get("options", []):
        options.setdefault(o["name"], o.get("default", o["choices"][0]))
    max_events = args.max_events or a.record.get("max_events", 10**9)

    for path in args.hepmc:
        summary, per_event, extras = call_run(predictor, path, max_events,
                                              progress=lambda m: print(f"  {path.name}: {m}", flush=True), options=options)
        summary.update({"input": str(path), "model_file": str(a.model_path)})
        _write(out, f"{path.stem}_hepmc", summary, per_event, extras)
    if args.skim and not hasattr(predictor, "run_skim"):
        raise SystemExit(f"the predictor of '{a.id}' does not support --skim inputs")
    for path in args.skim:
        summary, per_event, extras = predictor.run_skim(path, args.max_events)
        summary.update({"input": str(path), "model_file": str(a.model_path)})
        _write(out, f"{path.stem}_skim", summary, per_event, extras)


def _write(out: Path, stem: str, summary: dict, per_event, extras: dict) -> None:
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    with h5py.File(out / f"{stem}.h5", "w") as h:
        h.create_dataset("event_probability", data=np.asarray(per_event, np.float32))
        for k, v in extras.items():
            h.create_dataset(k, data=v)
        h.attrs.update({k: v for k, v in summary.items() if isinstance(v, (int, float, str))})
    print(f"{stem}: {summary['n_events']} events, SR efficiency {summary['sr_efficiency']:.4f} "
          f"+- {summary['sr_efficiency_err']:.4f}", flush=True)
