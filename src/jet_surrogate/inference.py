"""ONNX model loading and per-jet inference."""

from typing import List, Dict
import numpy as np
import onnxruntime as ort

from .features import INT_VARS, FLOAT_VARS, N_FEATURES


class OnnxJetScorer:
    """
    Wraps an ONNX session and scores jets from their truth-particle feature dicts.

    The model is expected to consume:
      input[0]  float32[batch, max_particles, N_FEATURES]  — particle features
      input[1]  float32[batch, max_particles]               — validity mask (1=valid, 0=pad)

    If the model has only one input the mask is omitted.
    The first model output is taken as the score; if it is 2-D the last column
    (signal probability) is used.
    """

    def __init__(self, model_path: str, max_particles: int = 200) -> None:
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.max_particles = max_particles
        self._in_names = [i.name for i in self.session.get_inputs()]
        self._out_names = [o.name for o in self.session.get_outputs()]

    # ------------------------------------------------------------------
    def _build_tensors(
        self, jets_particles: List[List[Dict]]
    ) -> tuple:
        n_jets = len(jets_particles)
        features = np.zeros(
            (n_jets, self.max_particles, N_FEATURES), dtype=np.float32
        )
        mask = np.zeros((n_jets, self.max_particles), dtype=np.float32)

        for i, plist in enumerate(jets_particles):
            for j, pf in enumerate(plist[: self.max_particles]):
                for k, var in enumerate(INT_VARS):
                    features[i, j, k] = float(pf["ints"][var])
                for k, var in enumerate(FLOAT_VARS):
                    features[i, j, len(INT_VARS) + k] = pf["floats"][var]
                mask[i, j] = 1.0

        return features, mask

    # ------------------------------------------------------------------
    def score(self, jets_particles: List[List[Dict]]) -> np.ndarray:
        """
        Score a batch of jets.

        Parameters
        ----------
        jets_particles:
            List of jets; each jet is a list of feature dicts as returned by
            ``features.extract_features``.

        Returns
        -------
        np.ndarray of shape [n_jets] with a float32 score per jet.
        """
        if not jets_particles:
            return np.empty(0, dtype=np.float32)

        features, mask = self._build_tensors(jets_particles)

        feed: Dict[str, np.ndarray] = {self._in_names[0]: features}
        if len(self._in_names) >= 2:
            feed[self._in_names[1]] = mask

        raw = self.session.run(self._out_names, feed)
        scores = raw[0]

        if scores.ndim > 1:
            scores = scores[:, -1]

        return scores.flatten().astype(np.float32)
