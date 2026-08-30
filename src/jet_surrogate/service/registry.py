"""The library of preserved analyses: one directory per analysis under
``analyses/`` (``JS_ANALYSES_DIR``), each with an ``analysis.yaml``, its
surrogate model files, a ``README.md`` model card and optional
``figures/``. New analyses arrive by pull request; ``validate()`` is what CI
and the tests run on every record.

A predictor turns a HepMC file into per-event signal-region probabilities.
The interface lives in ``service/predictors/``; nothing else about a
model is known to the service. An analysis brings its own predictor as
``analyses/<id>/predictor.py`` (a registered ``Predictor`` subclass, named
by ``predictor.type``), keeping analysis-specific code out of the package;
``requirements`` lists the Python modules it needs.
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
LOCAL_PREDICTOR = "predictor.py"       # optional, next to analysis.yaml: the analysis' own predictor class
STATUSES = ("example", "preserved", "draft")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


@dataclass
class Analysis:
    id: str
    path: Path
    record: dict
    predictor_type: str
    model_path: Path
    predictor_cls: type | None = None      # resolved at load time (analysis-local predictor.py or built-in)

    def predictor(self, device: str = "cpu"):
        return self.predictor_cls(self, device)

    @property
    def assets_dir(self) -> Path:
        """Where downloaded assets (large model files declared under ``assets``) live:
        JS_ASSET_DIR/<id> if set (the service PVC), else <analysis>/.assets (gitignored)."""
        root = os.environ.get("JS_ASSET_DIR")
        return (Path(root) / self.id) if root else (self.path / ".assets")

    def fetch_assets(self, log=print) -> Path:
        """Download and unpack every declared asset once; returns the assets directory."""
        import hashlib
        import shutil
        import tarfile
        import urllib.request
        d = self.assets_dir; d.mkdir(parents=True, exist_ok=True)
        for a in self.record.get("assets", []):
            marker = d / f".{a['name']}.done"
            if marker.exists():
                continue
            tmp = d / (a["name"] + ".download")
            log(f"[{self.id}] downloading {a['url']}")
            with urllib.request.urlopen(a["url"], timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            if a.get("sha256"):
                h = hashlib.sha256(tmp.read_bytes()).hexdigest()
                if h != a["sha256"]:
                    tmp.unlink(); raise ValueError(f"{a['name']}: sha256 mismatch ({h})")
            if a.get("extract", False):
                with tarfile.open(tmp) as t:
                    t.extractall(d / a["name"], filter="data")
                tmp.unlink()
            else:
                tmp.rename(d / a["name"])
            marker.write_text(a["url"])
        return d

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
                "example_name": self.example_path.name if self.example_path else None,
                "options": list(r.get("options", []))}
        if not detail:
            return base
        base.update({
            "description_html": render_markdown((self.path / "README.md").read_text()) if (self.path / "README.md").exists() else "",
            "signal_region": r["signal_region"], "inputs": list(r["inputs"]),
            "references": list(r.get("references", [])), "validation": list(r.get("validation", [])),
            "figures": [f for f in r.get("figures", []) if (self.path / "figures" / f["file"]).exists()],
            "model": {"type": self.predictor_type, "file": r["predictor"]["model"], "version": str(r["version"]),
                      "training": r.get("training", ""),
                      "local_predictor": (self.path / LOCAL_PREDICTOR).exists(),
                      "requirements": list(r.get("requirements", []))},
            "contact": r.get("contact", ""),
        })
        return base


def repo_url() -> str:
    return os.environ.get("JS_REPO_URL", "https://github.com/jburzy/jet-surrogate")


def analyses_dir() -> Path:
    return Path(os.environ.get("JS_ANALYSES_DIR", "analyses")).resolve()


def local_predictors(path: Path) -> dict[str, type]:
    """Predictor classes defined in ``<analysis>/predictor.py`` (registered
    under that module only, so two analyses may use the same type name)."""
    import importlib.util
    import inspect

    from .predictors import Predictor

    f = path / LOCAL_PREDICTOR
    if not f.exists():
        return {}
    spec = importlib.util.spec_from_file_location(f"analyses.{path.name.replace('-', '_')}.predictor", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {cls.type_name: cls for _, cls in inspect.getmembers(mod, inspect.isclass)
            if issubclass(cls, Predictor) and cls is not Predictor and cls.__module__ == mod.__name__}


def resolve_predictor(path: Path, record: dict) -> type | None:
    t = (record.get("predictor") or {}).get("type")
    local = local_predictors(path) if (path / LOCAL_PREDICTOR).exists() else {}
    return local.get(t) or PREDICTORS.get(t)


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
    try:
        local = local_predictors(path)
    except Exception as e:                                   # noqa: BLE001
        local = {}; problems.append(f"{path.name}: {LOCAL_PREDICTOR} failed to import ({type(e).__name__}: {e})")
    if pred.get("type") not in local and pred.get("type") not in PREDICTORS:
        problems.append(f"{path.name}: predictor.type '{pred.get('type')}' is neither defined in {LOCAL_PREDICTOR} "
                        f"({sorted(local)}) nor a built-in type ({sorted(PREDICTORS)})")
    for req in r.get("requirements", []):
        import importlib
        try:
            importlib.import_module(req)
        except ImportError:
            problems.append(f"{path.name}: requirement '{req}' is not importable (add it to the infer feature of pixi.toml)")
    if not pred.get("model"):
        problems.append(f"{path.name}: predictor.model missing")
    elif not (path / pred["model"]).exists() and not r.get("assets"):
        problems.append(f"{path.name}: predictor.model file '{pred['model']}' not found (declare it under assets if it is downloaded)")
    for a in r.get("assets", []):
        for k in ("name", "url"):
            if k not in a:
                problems.append(f"{path.name}: asset without '{k}'")
    for o in r.get("options", []):
        if not o.get("name") or not o.get("choices"):
            problems.append(f"{path.name}: option needs 'name' and 'choices'")
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
    for path in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        problems = validate(path)
        if problems:
            if strict:
                raise ValueError("\n".join(problems))
            continue
        r = yaml.safe_load((path / "analysis.yaml").read_text())
        out[path.name] = Analysis(path.name, path, r, r["predictor"]["type"], path / r["predictor"]["model"],
                                  resolve_predictor(path, r))
    return out


def render_markdown(text: str) -> str:
    try:
        import markdown
        return markdown.markdown(text, extensions=["tables"])
    except ImportError:                               # minimal fallback: paragraphs
        return "".join(f"<p>{html.escape(p)}</p>" for p in text.split("\n\n") if p.strip())


# ------------------------------------------------------------------ predictors
from .predictors import PREDICTORS, load_all as _load_predictors  # noqa: E402

_load_predictors()
