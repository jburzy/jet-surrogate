"""``jet_surrogate``: truth large-R jets -> transformer surrogate -> per-jet
probability -> Poisson-binomial probability of >= 2 tagged jets per event."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import Predictor, finish_summary, register


@register("jet_surrogate")
class JetSurrogatePredictor(Predictor):
    def __init__(self, analysis, device: str = "cpu"):
        from ...training import load_checkpoint, pick_device
        super().__init__(analysis, device)
        self.device = pick_device(device)
        self.model, self.pre, _ = load_checkpoint(analysis.model_path, self.device)
        self.model.to(self.device)

    def run(self, hepmc: Path, max_events: int, progress=None, chunk: int = 1000):
        from ...commands.predict import predict_sample
        from ...hepmc_io import read_hepmc
        from ...skim import skim_truth
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
        summary = finish_summary(self.analysis, summary, per_event,
                                 objects={"label": "truth jets", "count": int(len(jets)),
                                          "mean_probability": float(prob.mean()) if len(prob) else None})
        return summary, per_event, {"jet_probability": prob, "truth_jets": jets}
