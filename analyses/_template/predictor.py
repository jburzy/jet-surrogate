"""Predictor template: a BDT acting on a fixed set of objects per event.

The class is loaded from this file for this analysis only. It must subclass
``Predictor``, be decorated with ``@register("<type>")`` matching
``predictor.type`` in analysis.yaml, and implement ``run``.

The example below follows a Corpe et al. style surrogate (arXiv:2502.10231):
a gradient-boosted model on per-event features that predicts the probability
that the event lands in the analysis' signal region (region A of an ABCD
plane), from which the efficiency and, given a cross section and luminosity
in the config, the expected yield follow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jet_surrogate.service.predictors import Predictor, finish_summary, register


def event_features(part, config: dict) -> np.ndarray:
    """Per-event feature matrix from the particle record of a HepMC batch.

    ``part`` is an awkward array (events -> particles) with fields pid, status,
    m1, m2, d1, d2, charge, mass, e, pt, eta, phi, x, y, z (mm). Replace this
    with the features the model was trained on.
    """
    import awkward as ak
    llp_id = config.get("llp_pdgid", 35)
    llp = part[np.abs(part.pid) == llp_id]
    lead = ak.firsts(llp[ak.argsort(llp.pt, ascending=False)])
    # decay position of the leading LLP: the production vertex of its first daughter
    dau = part[ak.local_index(part.pid) == ak.fill_none(lead.d1, -1)]
    lxy = ak.fill_none(ak.firsts(np.hypot(dau.x, dau.y)), 0.0)
    return np.stack([ak.to_numpy(ak.fill_none(lead.pt, 0.0)), ak.to_numpy(ak.fill_none(lead.eta, 0.0)),
                     ak.to_numpy(lxy), ak.to_numpy(ak.num(llp))], axis=1).astype(np.float32)


@register("my_bdt")
class MyBDT(Predictor):
    def __init__(self, analysis, device: str = "cpu"):
        super().__init__(analysis, device)
        import xgboost
        self.booster = xgboost.Booster()
        self.booster.load_model(str(analysis.model_path))
        self.config = analysis.record["predictor"].get("config", {})

    def run(self, hepmc: Path, max_events: int, progress=None):
        from jet_surrogate.hepmc_io import read_hepmc
        probs = []
        for batch in read_hepmc(hepmc, max_events=max_events):
            x = event_features(batch.part, self.config)
            probs.append(self.booster.inplace_predict(x))
            if progress:
                progress(f"{sum(len(p) for p in probs)} events")
        per_event = np.clip(np.concatenate(probs), 0.0, 1.0)
        quantities = []
        if "xsec_pb" in self.config and "lumi_fb" in self.config:
            n = per_event.mean() * self.config["xsec_pb"] * 1e3 * self.config["lumi_fb"]
            quantities.append({"name": "expected signal-region yield", "value": float(n), "unit": "events",
                               "note": f"for {self.config['xsec_pb']} pb and {self.config['lumi_fb']} fb^-1"})
        summary = finish_summary(self.analysis, {}, per_event, quantities=quantities)
        return summary, per_event, {}
