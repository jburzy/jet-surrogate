"""The library of preserved analyses: one directory per analysis under
``analyses/`` (``JS_ANALYSES_DIR``), each with an ``analysis.yaml``, its
surrogate model files, a ``README.md`` model card and optional
``figures/``. New analyses arrive by pull request; ``validate()`` is what CI
and the tests run on every record.

A predictor type turns a HepMC file into (summary, per-event probabilities).
``jet_surrogate`` is the first; adding another means registering a class in
``PREDICTORS`` that follows the same interface.
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REQUIRED = ("id", "title", "short", "experiment", "status", "version", "signal_region", "inputs", "predictor")
STATUSES = ("example", "preserved", "draft")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


@dataclass
class Analysis:
    id: str
    path: Path
    record: dict
    predictor_type: str
    model_path: Path

    @property
    def example_path(self) -> Path | None:
        f = self.record.get("example")
        return (self.path / f) if f and (self.path / f).exists() else None

    def public(self, detail: bool = False) -> dict:
        r = self.record
        base = {"id": self.id, "title": r["title"], "short": r["short"], "experiment": r["experiment"],
                "status": r["status"], "version": str(r["version"]), "updated": str(r.get("updated", "")),
                "tags": list(r.get("tags", [])),
                "repo_url": f"{repo_url()}/tree/main/analyses/{self.id}",
                "default_max_events": int(r.get("default_max_events", 20000)),
                "max_events": int(r.get("max_events", 50000)),
                "example_url": f"/api/analyses/{self.id}/example" if self.example_path else None,
                "example_name": self.example_path.name if self.example_path else None}
        if not detail:
            return base
        base.update({
            "description_html": render_markdown((self.path / "README.md").read_text()) if (self.path / "README.md").exists() else "",
            "signal_region": r["signal_region"], "inputs": list(r["inputs"]),
            "references": list(r.get("references", [])), "validation": list(r.get("validation", [])),
            "figures": [f for f in r.get("figures", []) if (self.path / "figures" / f["file"]).exists()],
            "model": {"type": self.predictor_type, "file": r["predictor"]["model"], "version": str(r["version"]),
                      "training": r.get("training", "")},
            "contact": r.get("contact", ""),
        })
        return base


def repo_url() -> str:
    return os.environ.get("JS_REPO_URL", "https://github.com/jburzy/jet-surrogate")


def analyses_dir() -> Path:
    return Path(os.environ.get("JS_ANALYSES_DIR", "analyses")).resolve()


def validate(path: Path) -> list[str]:
    """Problems with one analysis directory (empty list = valid)."""
    problems = []
    rec_file = path / "analysis.yaml"
    if not rec_file.exists():
        return [f"{path.name}: analysis.yaml missing"]
    try:
        r = yaml.safe_load(rec_file.read_text()) or {}
    except yaml.YAMLError as e:
        return [f"{path.name}: analysis.yaml is not valid YAML ({e})"]
    for k in REQUIRED:
        if k not in r:
            problems.append(f"{path.name}: missing field '{k}'")
    if r.get("id") != path.name:
        problems.append(f"{path.name}: id '{r.get('id')}' must equal the directory name")
    if not ID_RE.match(path.name):
        problems.append(f"{path.name}: directory name must match {ID_RE.pattern}")
    if r.get("status") not in STATUSES:
        problems.append(f"{path.name}: status must be one of {STATUSES}")
    pred = r.get("predictor") or {}
    if pred.get("type") not in PREDICTORS:
        problems.append(f"{path.name}: predictor.type must be one of {sorted(PREDICTORS)}")
    if not pred.get("model") or not (path / pred["model"]).exists():
        problems.append(f"{path.name}: predictor.model file '{pred.get('model')}' not found")
    if r.get("example") and not (path / r["example"]).exists():
        problems.append(f"{path.name}: example file '{r['example']}' not found")
    for f in r.get("figures", []):
        if not (path / "figures" / f.get("file", "")).exists():
            problems.append(f"{path.name}: figure '{f.get('file')}' not found in figures/")
    if not (path / "README.md").exists():
        problems.append(f"{path.name}: README.md (model card) missing")
    return problems


def load(root: Path | None = None, strict: bool = True) -> dict[str, Analysis]:
    root = root or analyses_dir()
    out = {}
    for path in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        problems = validate(path)
        if problems:
            if strict:
                raise ValueError("\n".join(problems))
            continue
        r = yaml.safe_load((path / "analysis.yaml").read_text())
        out[path.name] = Analysis(path.name, path, r, r["predictor"]["type"], path / r["predictor"]["model"])
    return out


def render_markdown(text: str) -> str:
    try:
        import markdown
        return markdown.markdown(text, extensions=["tables"])
    except ImportError:                               # minimal fallback: paragraphs
        return "".join(f"<p>{html.escape(p)}</p>" for p in text.split("\n\n") if p.strip())


# ------------------------------------------------------------------ predictors
class JetSurrogatePredictor:
    """Truth large-R jets -> per-jet probabilities -> Poisson-binomial event probability."""

    def __init__(self, analysis: Analysis, device="cpu"):
        from ..training import load_checkpoint, pick_device
        self.analysis = analysis
        self.device = pick_device(device)
        self.model, self.pre, _ = load_checkpoint(analysis.model_path, self.device)
        self.model.to(self.device)

    def run(self, hepmc: Path, max_events: int, progress=None, chunk: int = 1000):
        from ..commands.predict import predict_sample
        from ..hepmc_io import read_hepmc
        from ..skim import skim_truth
        jets, parts, n_ev = [], [], 0
        for batch in read_hepmc(hepmc, max_events=max_events, chunk=chunk):
            tj, tp = skim_truth(batch.part)
            tj["event"] += n_ev
            jets.append(tj); parts.append(tp); n_ev += len(batch)
            if progress:
                progress(f"{n_ev} events read, {sum(len(j) for j in jets)} truth jets")
        if n_ev == 0:
            raise ValueError("no events could be read from the input (is it HepMC2/3 ASCII?)")
        jets = np.concatenate(jets); parts = np.concatenate(parts)
        summary, per_event, prob = predict_sample(jets, parts, n_ev, self.model, self.pre, self.device)
        counts, edges = np.histogram(per_event, bins=30, range=(0.0, 1.0))
        summary.update({"analysis": self.analysis.id, "version": str(self.analysis.record["version"]),
                        "model": self.analysis.record["predictor"]["model"],
                        "histogram": {"edges": edges.round(4).tolist(), "counts": counts.tolist()}})
        return summary, per_event, {"jet_probability": prob, "truth_jets": jets}


PREDICTORS = {"jet_surrogate": JetSurrogatePredictor}
