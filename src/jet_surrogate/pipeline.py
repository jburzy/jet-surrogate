"""Full analysis pipeline: HEPMC → jets → particle matching → ONNX → acceptance."""

import math
from typing import Optional, Dict, List
import numpy as np

from .reader import iter_events
from .reconstruction import reconstruct_r04_jets, recluster_r10_jets
from .matching import match_particles_to_jet
from .features import (
    extract_features, features_to_row,
    ALL_VARS, JETS_DTYPE, PARTICLES_DTYPE,
)
from .inference import OnnxJetScorer


def _reco_event(particles, large_r_pt_cut: float, dr_match: float):
    """Reconstruct jets and extract per-particle features for one event.

    Returns a list of jets, each represented as a dict:
        pt, eta, phi, mass  – large-R jet kinematics
        features            – list of per-particle feature dicts
    """
    r04_jets = reconstruct_r04_jets(particles)
    r10_jets = recluster_r10_jets(r04_jets, ptmin=large_r_pt_cut)

    jets = []
    for jet in r10_jets:
        matched = match_particles_to_jet(particles, jet, dr_cut=dr_match)
        feats = [extract_features(p, jet_eta=jet.eta, jet_phi=jet.phi) for p in matched]
        jets.append({
            "pt": jet.pt,
            "eta": jet.eta,
            "phi": jet.phi,
            "mass": jet.mass,
            "features": feats,
        })
    return jets


def run_reco(
    hepmc_file: str,
    large_r_pt_cut: float = 200.0,
    dr_match: float = 1.4,
    max_events: Optional[int] = None,
    max_particles: int = 200,
    output_npz: Optional[str] = None,
    verbose: bool = False,
) -> Dict:
    """
    Run reconstruction and feature extraction only (no inference).

    Returns
    -------
    dict with keys:
        n_events        – events processed
        n_jets          – total large-R jets found
        jet_pt          – np.ndarray of all jet pTs
        jet_eta         – np.ndarray of all jet etas
        jet_n_particles – np.ndarray of matched particle counts per jet
        jets            – structured np.ndarray [n_jets] with JETS_DTYPE
        particles       – structured np.ndarray [n_jets, max_particles] with PARTICLES_DTYPE
        feature_names   – list of field names (from ALL_VARS)

    If output_npz is given, jets and particles arrays are saved to that HDF5 file.
    """
    all_jet_rows: List[tuple] = []
    all_particle_rows: List[List[Dict]] = []
    all_jet_npart: List[int] = []

    n_events = 0
    for event in iter_events(hepmc_file):
        if max_events is not None and n_events >= max_events:
            break

        jets = _reco_event(list(event.particles), large_r_pt_cut, dr_match)
        for jet in jets:
            e = math.sqrt(jet["pt"] ** 2 * math.cosh(jet["eta"]) ** 2 + jet["mass"] ** 2)
            all_jet_rows.append((jet["pt"], jet["eta"], e, jet["mass"], jet["phi"]))
            all_particle_rows.append(jet["features"])
            all_jet_npart.append(len(jet["features"]))

        n_events += 1
        if verbose and n_events % 100 == 0:
            print(f"  Processed {n_events} events …")

    n_jets = len(all_jet_rows)

    jets_arr = np.array(all_jet_rows, dtype=JETS_DTYPE) if n_jets > 0 else np.zeros(0, dtype=JETS_DTYPE)

    parts_arr = np.zeros((n_jets, max_particles), dtype=PARTICLES_DTYPE)
    for i, feat_list in enumerate(all_particle_rows):
        for j, feat in enumerate(feat_list[:max_particles]):
            parts_arr[i, j] = features_to_row(feat)

    result = {
        "n_events": n_events,
        "n_jets": n_jets,
        "jet_pt": jets_arr["pt"] if n_jets > 0 else np.array([], dtype=np.float32),
        "jet_eta": jets_arr["eta"] if n_jets > 0 else np.array([], dtype=np.float32),
        "jet_n_particles": np.array(all_jet_npart, dtype=np.int32),
        "jets": jets_arr,
        "particles": parts_arr,
        "feature_names": ALL_VARS,
    }

    if output_npz:
        import h5py
        with h5py.File(output_npz, "w") as hf:
            hf.create_dataset("jets", data=jets_arr)
            hf.create_dataset("particles", data=parts_arr)

    return result


def run_pipeline(
    hepmc_file: str,
    model_path: str,
    threshold: float = 0.5,
    large_r_pt_cut: float = 200.0,
    dr_match: float = 1.4,
    max_events: Optional[int] = None,
    max_particles: int = 200,
    verbose: bool = False,
) -> Dict:
    """
    Run the full analysis pipeline.

    Steps
    -----
    1. Read HEPMC events.
    2. Reconstruct anti-kt R=0.4 truth jets from stable visible particles.
    3. Recluster R=0.4 jets → anti-kt R=1.0 jets, keep pT > large_r_pt_cut.
    4. For each large-R jet associate all truth particles within dR < dr_match.
    5. Extract the INT_VARS + FLOAT_VARS feature set for each matched particle.
    6. Run ONNX model; assign a score to each jet.
    7. Compute per-event acceptance: event passes when >=2 jets exceed threshold.

    Returns
    -------
    dict with keys:
        n_events          – total events processed
        n_events_2jets    – events with >=2 jets above threshold
        acceptance        – n_events_2jets / n_events
        all_scores        – flat list of all jet scores across all events
    """
    scorer = OnnxJetScorer(model_path, max_particles=max_particles)

    n_events = 0
    n_events_2jets = 0
    all_scores = []

    for event in iter_events(hepmc_file):
        if max_events is not None and n_events >= max_events:
            break

        jets = _reco_event(list(event.particles), large_r_pt_cut, dr_match)
        jets_particle_features = [jet["features"] for jet in jets]

        if jets_particle_features:
            scores = scorer.score(jets_particle_features)
            all_scores.extend(scores.tolist())
            n_passing = int(np.sum(scores > threshold))
        else:
            n_passing = 0

        if n_passing >= 2:
            n_events_2jets += 1

        n_events += 1
        if verbose and n_events % 100 == 0:
            print(f"  Processed {n_events} events …")

    acceptance = n_events_2jets / n_events if n_events > 0 else 0.0

    return {
        "n_events": n_events,
        "n_events_2jets": n_events_2jets,
        "acceptance": acceptance,
        "all_scores": all_scores,
    }
