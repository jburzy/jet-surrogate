"""An analysis brings its own predictor as analyses/<id>/predictor.py: a
registered class named by predictor.type, loaded for that analysis only.
Here a toy event-level model on the two leading particles runs through the
registry, the worker and the API."""

from pathlib import Path

import pytest
import yaml

TOY_PREDICTOR = '''
from pathlib import Path
import numpy as np
import yaml
from jet_surrogate.service.predictors import Predictor, finish_summary, register


@register("two_particle_toy")
class TwoParticleToy(Predictor):
    """Probability = logistic(sum of the pT of the two leading final-state particles / scale)."""

    def __init__(self, analysis, device="cpu"):
        super().__init__(analysis, device)
        self.scale = float(yaml.safe_load(analysis.model_path.read_text())["scale_gev"])
        self.cfg = analysis.record["predictor"].get("config", {})

    def run(self, hepmc: Path, max_events: int, progress=None):
        import awkward as ak
        from jet_surrogate.hepmc_io import read_hepmc
        probs = []
        for batch in read_hepmc(hepmc, max_events=max_events):
            part = batch.part[batch.part.status == 1]
            top2 = ak.sort(part.pt, ascending=False)[:, :2]
            s = ak.to_numpy(ak.fill_none(ak.sum(top2, axis=1), 0.0))
            probs.append(1 / (1 + np.exp(-(s / self.scale - 2))))
            if progress:
                progress(f"{sum(len(p) for p in probs)} events")
        per_event = np.concatenate(probs)
        yield_ = per_event.mean() * self.cfg["xsec_pb"] * 1e3 * self.cfg["lumi_fb"]
        return finish_summary(self.analysis, {}, per_event,
                              objects={"label": "leading particles", "count": 2 * len(per_event), "mean_probability": None},
                              quantities=[{"name": "expected yield", "value": float(yield_), "unit": "events"}]), per_event, {}
'''


@pytest.fixture(scope="module")
def hepmc(tmp_path_factory):
    from jet_surrogate.generate import generate_hepmc
    return generate_hepmc("signal", n_events=6, seed=11, ctau_mm=0.1, out_dir=tmp_path_factory.mktemp("gen"))


@pytest.fixture
def library(tmp_path):
    lib = tmp_path / "analyses"
    a = lib / "toy-two-particle"; a.mkdir(parents=True)
    (a / "predictor.py").write_text(TOY_PREDICTOR)
    (a / "model.yaml").write_text("scale_gev: 100\n")
    (a / "README.md").write_text("# toy\n")
    (a / "analysis.yaml").write_text(yaml.safe_dump({
        "id": "toy-two-particle", "title": "Toy two-particle model", "short": "toy", "experiment": "other",
        "status": "draft", "version": "0.1", "signal_region": "toy", "inputs": ["HepMC"],
        "requirements": ["numpy"],
        "predictor": {"type": "two_particle_toy", "model": "model.yaml", "config": {"xsec_pb": 1.0, "lumi_fb": 140}}}))
    (lib / "_template").mkdir()                      # underscore directories are ignored by the loader
    return lib


def test_local_predictor_is_resolved_and_validated(library):
    from jet_surrogate.service import registry
    assert registry.validate(library / "toy-two-particle") == []
    lib = registry.load(library)
    assert list(lib) == ["toy-two-particle"]
    a = lib["toy-two-particle"]
    assert a.predictor_cls.type_name == "two_particle_toy" and a.public(detail=True)["model"]["local_predictor"]
    bad = library / "toy-two-particle" / "analysis.yaml"
    bad.write_text(bad.read_text().replace("two_particle_toy", "nope"))
    assert any("neither defined" in p for p in registry.validate(library / "toy-two-particle"))


def test_plugin_predictor_end_to_end(hepmc, library, tmp_path, monkeypatch):
    monkeypatch.setenv("JS_SERVICE_DIR", str(tmp_path / "svc"))
    monkeypatch.setenv("JS_ANALYSES_DIR", str(library))
    from fastapi.testclient import TestClient
    from jet_surrogate.service.app import create_app
    from jet_surrogate.service.worker import main_loop
    client = TestClient(create_app())
    assert [x["id"] for x in client.get("/api/analyses").json()] == ["toy-two-particle"]
    with open(hepmc, "rb") as f:
        job_id = client.post("/api/jobs", files={"file": ("e.hepmc", f)}, data={"analysis": "toy-two-particle"}).json()["id"]
    main_loop(once=True)
    j = client.get(f"/api/jobs/{job_id}").json()
    assert j["status"] == "done", j
    r = j["result"]
    assert r["n_events"] == 6 and r["predictor_type"] == "two_particle_toy" and r["objects"]["count"] == 12
    assert r["quantities"][0]["name"] == "expected yield" and r["quantities"][0]["value"] >= 0
    assert 0.0 <= r["sr_efficiency"] <= 1.0 and len(r["histogram"]["counts"]) == 30
