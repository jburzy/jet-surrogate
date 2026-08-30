"""Every analysis in the library must validate (this is what CI runs on PRs)."""

from pathlib import Path

from jet_surrogate.service import registry

ROOT = Path(__file__).resolve().parents[1] / "analyses"


def test_all_analyses_validate():
    problems = []
    for path in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        problems += registry.validate(path)
    assert not problems, "\n".join(problems)


def test_registry_loads_and_publishes():
    lib = registry.load(ROOT)
    assert "emerging-jets-delphes" in lib
    pub = lib["emerging-jets-delphes"].public(detail=True)
    for key in ("title", "signal_region", "inputs", "model", "figures", "description_html", "default_max_events"):
        assert key in pub
    assert pub["model"]["type"] == "jet_surrogate"


def test_invalid_record_is_reported(tmp_path):
    bad = tmp_path / "bad-analysis"; bad.mkdir()
    (bad / "analysis.yaml").write_text("id: other\ntitle: x\n")
    problems = registry.validate(bad)
    assert any("must equal the directory name" in p for p in problems)
    assert any("missing field" in p for p in problems)
