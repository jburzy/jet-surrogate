"""End-to-end test of the web service on a tiny Pythia HepMC sample
(run with `pixi run -e infer test-service`; needs the released model)."""

import os
from pathlib import Path

import pytest

MODEL = Path("models/release/surrogate.pt")


@pytest.fixture(scope="module")
def hepmc(tmp_path_factory):
    from jet_surrogate.generate import generate_hepmc
    out = tmp_path_factory.mktemp("gen")
    return generate_hepmc("signal", n_events=8, seed=3, ctau_mm=0.1, out_dir=out)


@pytest.mark.skipif(not MODEL.exists(), reason="released model missing")
def test_submit_and_process(hepmc, tmp_path, monkeypatch):
    monkeypatch.setenv("JS_SERVICE_DIR", str(tmp_path / "svc"))
    monkeypatch.setenv("JS_MODEL", str(MODEL))
    from fastapi.testclient import TestClient
    from jet_surrogate.service.app import create_app
    from jet_surrogate.service.worker import main_loop

    client = TestClient(create_app())
    assert client.get("/health").json()["status"] == "ok"
    with open(hepmc, "rb") as f:
        r = client.post("/jobs", files={"file": ("events.hepmc", f)}, data={"label": "test", "max_events": "8"})
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    assert client.get(f"/jobs/{job_id}").json()["status"] == "queued"
    main_loop(once=True)
    j = client.get(f"/jobs/{job_id}").json()
    assert j["status"] == "done", j
    res = j["result"]
    assert res["n_events"] == 8 and 0.0 <= res["sr_efficiency"] <= 1.0
    assert client.get(f"/jobs/{job_id}/result.h5").status_code == 200
    assert "events" in client.get(f"/jobs/{job_id}/log").text


def test_upload_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("JS_SERVICE_DIR", str(tmp_path / "svc"))
    monkeypatch.setenv("JS_MAX_UPLOAD_MB", "0.001")
    from fastapi.testclient import TestClient
    from jet_surrogate.service.app import create_app
    client = TestClient(create_app())
    r = client.post("/jobs", files={"file": ("big.hepmc", b"x" * 5000)})
    assert r.status_code == 413
