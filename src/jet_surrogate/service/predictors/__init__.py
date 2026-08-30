"""Predictor types: how an analysis turns a HepMC file into per-event
signal-region probabilities. Each type is a class registered with
``@register("<type>")`` and named in ``analysis.yaml`` as ``predictor.type``.

Interface (``Predictor``):
    __init__(analysis, device="cpu")        load the model file(s) once
    run(hepmc, max_events, progress=None, options=None) -> (summary, per_event, extras)

``summary`` must contain the keys of ``REQUIRED_SUMMARY``; ``per_event`` is a
float array of length n_events with the probability that each event enters
the signal region; ``extras`` are optional arrays stored in the result
HDF5. Use ``finish_summary`` to fill the common fields (histogram,
threshold variant, analysis and model provenance).

Adding a type: an analysis ships ``analyses/<id>/predictor.py`` with a
registered class (the registry loads it for that analysis only), plus any
dependency in the ``infer`` feature of ``pixi.toml`` and in the record's
``requirements``. The HepMC reader
(``jet_surrogate.hepmc_io.read_hepmc``) yields batches with every particle
of every event (pid, status, mother/daughter links, kinematics, vertices),
so a predictor is free to select whatever objects and features it needs.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Callable

import numpy as np

REQUIRED_SUMMARY = ("n_events", "sr_efficiency", "sr_efficiency_err")
PREDICTORS: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        cls.type_name = name
        PREDICTORS[name] = cls
        return cls
    return deco


class Predictor:
    """Base class; subclasses implement ``__init__`` (model loading) and ``run``."""

    type_name = "base"

    def __init__(self, analysis, device: str = "cpu"):
        self.analysis = analysis

    def run(self, hepmc: Path, max_events: int, progress=None, options: dict | None = None):
        """``options`` are the per-job choices declared under ``options`` in
        analysis.yaml (e.g. which selection), already validated by the app."""
        raise NotImplementedError


def finish_summary(analysis, summary: dict, per_event: np.ndarray, *, objects: dict | None = None,
                   quantities: list[dict] | None = None) -> dict:
    """Add the fields every result carries: efficiency from the per-event
    probabilities if absent, the hard-threshold variant, a histogram, the
    analysis and model provenance, an optional description of the objects
    the surrogate acted on (``{label, count, mean_probability}``) and
    optional extra ``quantities`` (``[{name, value, err, unit, note}]``,
    e.g. a predicted signal-region yield for a given cross section)."""
    p = np.asarray(per_event, dtype=float)
    n = len(p)
    summary.setdefault("n_events", int(n))
    if "sr_efficiency" not in summary:
        summary["sr_efficiency"] = float(p.mean()) if n else 0.0
        summary["sr_efficiency_err"] = float(p.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    summary.setdefault("sr_efficiency_threshold05", float((p > 0.5).mean()) if n else 0.0)
    counts, edges = np.histogram(p, bins=30, range=(0.0, 1.0))
    summary["histogram"] = {"edges": edges.round(4).tolist(), "counts": counts.tolist()}
    if objects is not None:
        summary["objects"] = objects
    if quantities:
        summary["quantities"] = list(quantities)
    summary.update({"analysis": analysis.id, "version": str(analysis.record["version"]),
                    "model": analysis.record["predictor"]["model"], "predictor_type": analysis.predictor_type,
                    "signal_region": analysis.record.get("signal_region")})
    missing = [k for k in REQUIRED_SUMMARY if k not in summary]
    if missing:
        raise ValueError(f"predictor summary lacks {missing}")
    return summary


def call_run(predictor, hepmc, max_events, progress=None, options=None):
    """Invoke ``run`` with ``options`` only if the implementation accepts it."""
    import inspect
    params = inspect.signature(predictor.run).parameters
    if "options" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return predictor.run(hepmc, max_events, progress, options=options or {})
    return predictor.run(hepmc, max_events, progress)


def load_all() -> dict[str, type]:
    """Import every module in this package so their ``@register`` calls run."""
    for m in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{m.name}")
    return PREDICTORS
