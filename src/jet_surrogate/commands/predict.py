"""Predict the signal-region efficiency of a new model from generator truth.

    jet-surrogate predict --hepmc events.hepmc [more.hepmc ...] [--analysis emerging-jets-delphes]
    jet-surrogate predict --skim data/skim/signal_m10_ctau0.1mm_seed1.h5 --model models/surrogate/surrogate.pt

This is the reinterpretation entry point: no detector simulation, no tagger.
Generator particles (HepMC2/3, or the truth tables of an existing skim) are
clustered into truth large-R jets, the surrogate returns a probability per
jet that the detector-level tagger would select it, and the per-event
probability of the two-jet signal region follows from the Poisson binomial.
Writes <out>/<stem>.json (n_events, sr_efficiency +- error, hard-threshold
variant, mean per-jet probability) and <out>/<stem>.h5 (per-event P and the
per-jet probabilities with their jet kinematics).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from ..metrics import predicted_sr_efficiency, sr_efficiency
from ..training import load_checkpoint, pick_device, score_padded


def add_arguments(ap) -> None:
    ap.add_argument("--hepmc", nargs="*", type=Path, default=[], help="HepMC2/3 files")
    ap.add_argument("--skim", nargs="*", type=Path, default=[], help="skim HDF5 files (truth tables)")
    ap.add_argument("--analysis", default="emerging-jets-delphes",
                    help="analysis id from the registry (analyses/<id>/analysis.yaml); its model is used unless --model is given")
    ap.add_argument("--model", default=None, help="explicit .pt or .ckpt (overrides --analysis)")
    ap.add_argument("--out", default="results/predict")
    ap.add_argument("--option", action="append", default=[], metavar="NAME=VALUE",
                    help="per-analysis option (see the analysis record), e.g. --option selection=CR+2J")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--device", default=None)


def _truth_from_hepmc(path: Path, max_events, chunk):
    from ..hepmc_io import read_hepmc
    from ..skim import skim_truth
    jets, parts, n_ev = [], [], 0
    for batch in read_hepmc(path, max_events=max_events, chunk=chunk):
        tj, tp = skim_truth(batch.part)
        tj["event"] += n_ev
        jets.append(tj); parts.append(tp); n_ev += len(batch)
        print(f"  {path.name}: {n_ev} events, {sum(len(j) for j in jets)} truth jets", flush=True)
    return np.concatenate(jets), np.concatenate(parts), n_ev


def _truth_from_skim(path: Path, max_events):
    with h5py.File(path, "r") as h:
        jets, parts, n_ev = h["truth_jets"][...], h["truth_parts"][...], int(h.attrs["n_events"])
    if max_events is not None and max_events < n_ev:
        keep = jets["event"] < max_events
        jets, parts, n_ev = jets[keep], parts[keep], max_events
    return jets, parts, n_ev


def predict_sample(jets: np.ndarray, parts: np.ndarray, n_events: int, model, pre, device) -> tuple[dict, np.ndarray]:
    logit = score_padded(model, pre, parts, device) if len(parts) else np.zeros(0, np.float32)
    prob = 1.0 / (1.0 + np.exp(-logit))
    eff, err = predicted_sr_efficiency(jets["event"], prob, n_events)
    eff_thr, err_thr = sr_efficiency(jets["event"], prob > 0.5, n_events)
    per_event = np.zeros(n_events)
    order = np.argsort(jets["event"], kind="stable")
    from ..metrics import prob_at_least
    ev, pr = jets["event"][order], prob[order]
    starts = np.searchsorted(ev, np.arange(n_events), side="left")
    stops = np.searchsorted(ev, np.arange(n_events), side="right")
    per_event[:] = [prob_at_least(pr[a:b], 2) if b > a else 0.0 for a, b in zip(starts, stops)]
    summary = {"n_events": int(n_events), "n_truth_jets": int(len(jets)),
               "sr_efficiency": eff, "sr_efficiency_err": err,
               "sr_efficiency_threshold05": eff_thr, "sr_efficiency_threshold05_err": err_thr,
               "mean_jet_probability": float(prob.mean()) if len(prob) else None}
    return summary, per_event, prob


def run(args) -> None:
    if not args.hepmc and not args.skim:
        raise SystemExit("give --hepmc and/or --skim inputs")
    device = pick_device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from ..service import registry
    from ..service.predictors import finish_summary

    if args.model is None:
        # HepMC inputs go through the analysis' registered predictor type, whatever it is
        a = registry.load(strict=False).get(args.analysis)
        if a is None:
            raise SystemExit(f"unknown analysis '{args.analysis}' (see analyses/); or pass --model")
        if a.record.get("assets"):
            a.fetch_assets()
        predictor = a.predictor(str(device))
        options = dict(kv.split("=", 1) for kv in args.option)
        for o in a.record.get("options", []):
            options.setdefault(o["name"], o.get("default", o["choices"][0]))
        from ..service.predictors import call_run
        for path in args.hepmc:
            summary, per_event, extras = call_run(predictor, path, args.max_events or a.record.get("max_events", 10**9),
                                                  progress=lambda m: print(f"  {path.name}: {m}", flush=True), options=options)
            summary["input"] = str(path)
            _write(out, f"{path.stem}_hepmc", summary, per_event, extras)
        if not args.skim:
            return
        if a.predictor_type != "jet_surrogate":
            raise SystemExit("--skim inputs are only meaningful for jet_surrogate analyses")
        model, pre = predictor.model, predictor.pre
        model_path = a.model_path
    else:
        a = None
        model_path = Path(args.model)
        model, pre, _ = load_checkpoint(model_path, device)
        model.to(device)
        for path in args.hepmc:
            jets, parts, n_ev = _truth_from_hepmc(path, args.max_events, args.chunk)
            summary, per_event, prob = predict_sample(jets, parts, n_ev, model, pre, device)
            summary.update({"input": str(path), "model": str(model_path)})
            _write(out, f"{path.stem}_hepmc", summary, per_event, {"jet_probability": prob, "truth_jets": jets})
    for path in args.skim:
        jets, parts, n_ev = _truth_from_skim(path, args.max_events)
        summary, per_event, prob = predict_sample(jets, parts, n_ev, model, pre, device)
        summary.update({"input": str(path), "model": str(model_path), "analysis": a.id if a else None})
        _write(out, f"{path.stem}_skim", summary, per_event, {"jet_probability": prob, "truth_jets": jets})


def _write(out: Path, stem: str, summary: dict, per_event, extras: dict) -> None:
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    with h5py.File(out / f"{stem}.h5", "w") as h:
        h.create_dataset("event_probability", data=np.asarray(per_event, np.float32))
        for k, v in extras.items():
            h.create_dataset(k, data=v)
        h.attrs.update({k: v for k, v in summary.items() if isinstance(v, (int, float, str))})
    print(f"{stem}: {summary['n_events']} events, SR efficiency {summary['sr_efficiency']:.4f} "
          f"+- {summary['sr_efficiency_err']:.4f}", flush=True)
