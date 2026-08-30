"""Predictor of the emerging-jets example analysis: generator particles are
clustered into truth large-R jets, a transformer surrogate returns per jet
the probability that the detector-level tagger would select it, and the
per-event probability of the two-jet signal region follows from the Poisson
binomial. Everything specific to this analysis lives here; the service only
knows the ``Predictor`` interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jet_surrogate.service.predictors import Predictor, finish_summary, register


def predict_sample(jets: np.ndarray, parts: np.ndarray, n_events: int, model, pre, device) -> tuple[dict, np.ndarray, np.ndarray]:
    """Per-jet probabilities -> per-event signal-region probabilities and summary."""
    from jet_surrogate.metrics import predicted_sr_efficiency, prob_at_least, sr_efficiency
    from jet_surrogate.training import score_padded
    logit = score_padded(model, pre, parts, device) if len(parts) else np.zeros(0, np.float32)
    prob = 1.0 / (1.0 + np.exp(-logit))
    eff, err = predicted_sr_efficiency(jets["event"], prob, n_events)
    eff_thr, err_thr = sr_efficiency(jets["event"], prob > 0.5, n_events)
    order = np.argsort(jets["event"], kind="stable")
    ev, pr = jets["event"][order], prob[order]
    starts = np.searchsorted(ev, np.arange(n_events), side="left")
    stops = np.searchsorted(ev, np.arange(n_events), side="right")
    per_event = np.array([prob_at_least(pr[a:b], 2) if b > a else 0.0 for a, b in zip(starts, stops)])
    summary = {"n_events": int(n_events), "n_truth_jets": int(len(jets)),
               "sr_efficiency": eff, "sr_efficiency_err": err,
               "sr_efficiency_threshold05": eff_thr, "sr_efficiency_threshold05_err": err_thr,
               "mean_jet_probability": float(prob.mean()) if len(prob) else None}
    return summary, per_event, prob


@register("emerging_jets_transformer")
class EmergingJetsPredictor(Predictor):
    def __init__(self, analysis, device: str = "cpu"):
        from jet_surrogate.training import load_checkpoint, pick_device
        super().__init__(analysis, device)
        self.device = pick_device(device)
        self.model, self.pre, _ = load_checkpoint(analysis.model_path, self.device)
        self.model.to(self.device)

    def run(self, hepmc: Path, max_events: int, progress=None, options=None, chunk: int = 1000):
        from jet_surrogate.hepmc_io import read_hepmc
        from jet_surrogate.skim import skim_truth
        jets, parts, n_ev = [], [], 0
        for batch in read_hepmc(hepmc, max_events=max_events, chunk=chunk):
            tj, tp = skim_truth(batch.part)
            tj["event"] += n_ev
            jets.append(tj); parts.append(tp); n_ev += len(batch)
            if progress:
                progress(f"{n_ev} events read, {sum(len(j) for j in jets)} truth jets")
        if n_ev == 0:
            raise ValueError("no events could be read from the input (is it HepMC2/3 ASCII?)")
        return self._finish(np.concatenate(jets), np.concatenate(parts), n_ev)

    def run_skim(self, skim: Path, max_events: int | None = None):
        """Same prediction from the truth tables of an existing skim HDF5
        (used by ``jet-surrogate predict --skim`` for closure studies)."""
        import h5py
        with h5py.File(skim, "r") as h:
            jets, parts, n_ev = h["truth_jets"][...], h["truth_parts"][...], int(h.attrs["n_events"])
        if max_events is not None and max_events < n_ev:
            keep = jets["event"] < max_events
            jets, parts, n_ev = jets[keep], parts[keep], max_events
        return self._finish(jets, parts, n_ev)

    def _finish(self, jets, parts, n_ev):
        summary, per_event, prob = predict_sample(jets, parts, n_ev, self.model, self.pre, self.device)
        summary = finish_summary(self.analysis, summary, per_event,
                                 objects={"label": "truth jets", "count": int(len(jets)),
                                          "mean_probability": float(prob.mean()) if len(prob) else None})
        return summary, per_event, {"jet_probability": prob, "truth_jets": jets}
