"""A new predictor type only needs a registered class and an analysis
record naming it: here a trivial event-level model on two particles per
event runs through the whole service (registry, worker, API)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from jet_surrogate.service.predictors import PREDICTORS, Predictor, finish_summary, register


@register("two_particle_toy")
class TwoParticleToy(Predictor):
    """Probability = logistic(sum of the pT of the two leading final-state particles / 100 GeV)."""

    def __init__(self, analysis, device="cpu"):
        super().__init__(analysis, device)
        self.scale = float(yaml.safe_load(analysis.model_path.read_text())["scale_gev"])

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
        return finish_summary(self.analysis, {}, per_event,
                              objects={"label": "leading particles", "count": 2 * len(per_event),
                                       "mean_probability": None}), per_event, {}


@pytest.fixture(scope="module")
def hepmc(tmp_path_factory):
    from jet_surrogate.generate import generate_hepmc
    return generate_hepmc("signal", n_events=6, seed=11, ctau_mm=0.1, out_dir=tmp_path_factory.mktemp("gen"))


def test_plugin_predictor_end_to_end(hepmc, tmp_path, monkeypatch):
    lib = tmp_path / "analyses"; a = lib / "toy-two-particle"; a.mkdir(parents=True)
    (a / "model.yaml").write_text("scale_gev: 100\n")
    (a / "README.md").write_text("# toy\n")
    (a / "analysis.yaml").write_text(yaml.safe_dump({
        "id": "toy-two-particle", "title": "Toy two-particle model", "short": "toy", "experiment": "other",
        "status": "draft", "version": "0.1", "signal_region": "toy", "inputs": ["HepMC"],
        "predictor": {"type": "two_particle_toy", "model": "model.yaml"}}))
    monkeypatch.setenv("JS_SERVICE_DIR", str(tmp_path / "svc"))
    monkeypatch.setenv("JS_ANALYSES_DIR", str(lib))
    assert "two_particle_toy" in PREDICTORS
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
    assert 0.0 <= r["sr_efficiency"] <= 1.0 and len(r["histogram"]["counts"]) == 30
