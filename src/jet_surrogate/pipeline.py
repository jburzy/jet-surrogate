"""Full analysis pipeline: HEPMC → jets → particle matching → ONNX → acceptance."""

from typing import Optional, Dict, List
import numpy as np

from .reader import iter_events
from .reconstruction import reconstruct_r04_jets, recluster_r10_jets
from .matching import match_particles_to_jet
from .features import extract_features, INT_VARS, FLOAT_VARS, ALL_VARS
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
        features        – np.ndarray [n_jets, max_particles, N_FEATURES] float32
        mask            – np.ndarray [n_jets, max_particles] bool

    If output_npz is given, the arrays are also saved to that file.
    """
    all_jet_pt, all_jet_eta, all_jet_phi, all_jet_mass, all_jet_npart = [], [], [], [], []
    all_features: List[List[Dict]] = []

    n_events = 0
    for event in iter_events(hepmc_file):
        if max_events is not None and n_events >= max_events:
            break

        jets = _reco_event(list(event.particles), large_r_pt_cut, dr_match)
        for jet in jets:
            all_jet_pt.append(jet["pt"])
            all_jet_eta.append(jet["eta"])
            all_jet_phi.append(jet["phi"])
            all_jet_mass.append(jet["mass"])
            all_jet_npart.append(len(jet["features"]))
            all_features.append(jet["features"])

        n_events += 1
        if verbose and n_events % 100 == 0:
            print(f"  Processed {n_events} events …")

    n_jets = len(all_jet_pt)
    n_features = len(ALL_VARS)

    def _particle_array(plist):
        arr = np.zeros((len(plist), n_features), dtype=np.float32)
        for j, pf in enumerate(plist):
            for k, var in enumerate(INT_VARS):
                arr[j, k] = float(pf["ints"][var])
            for k, var in enumerate(FLOAT_VARS):
                arr[j, len(INT_VARS) + k] = pf["floats"][var]
        return arr

    result = {
        "n_events": n_events,
        "n_jets": n_jets,
        "jet_pt": np.array(all_jet_pt, dtype=np.float32),
        "jet_eta": np.array(all_jet_eta, dtype=np.float32),
        "jet_phi": np.array(all_jet_phi, dtype=np.float32),
        "jet_mass": np.array(all_jet_mass, dtype=np.float32),
        "jet_n_particles": np.array(all_jet_npart, dtype=np.int32),
        "particle_arrays": [_particle_array(plist) for plist in all_features],
        "feature_names": ALL_VARS,
    }

    if output_npz:
        import h5py
        jet_arr = np.column_stack([
            result["jet_pt"], result["jet_eta"],
            result["jet_phi"], result["jet_mass"],
        ])
        with h5py.File(output_npz, "w") as hf:
            jets_ds = hf.create_dataset("jets", data=jet_arr)
            jets_ds.attrs["feature_names"] = ["pt", "eta", "phi", "mass"]

            parts_grp = hf.create_group("particles")
            parts_grp.attrs["feature_names"] = ALL_VARS
            for i, arr in enumerate(result["particle_arrays"]):
                parts_grp.create_dataset(str(i), data=arr)

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
