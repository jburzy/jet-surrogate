"""Unit tests for the pieces that are easy to get silently wrong."""

import awkward as ak
import numpy as np
from pathlib import Path

import pytest

from jet_surrogate import features as F
from jet_surrogate.jets import (associate, delta_phi, match_jets, recluster_large_r,
                                select_small_r)
from jet_surrogate.metrics import (predicted_sr_efficiency, prob_at_least,
                                   sr_efficiency, threshold_at_rejection)


def _small(events):
    return ak.zip({"pt": ak.Array([[float(p) for p, *_ in e] for e in events]),
                   "eta": ak.Array([[float(x) for _, x, _ in e] for e in events]),
                   "phi": ak.Array([[float(y) for *_, y in e] for e in events]),
                   "mass": ak.Array([[0.0] * len(e) for e in events])})


def test_recluster_merges_nearby_small_r_jets():
    # two 150 GeV small-R jets 0.6 apart -> one large-R jet with the vector-sum
    # pT 2*150*cos(0.3); a lone far-away 150 GeV jet fails the 200 GeV cut
    small = select_small_r(_small([[(150, 0.0, 0.0), (150, 0.0, 0.6), (150, 0.0, 3.0)]]))
    lr = recluster_large_r(small)
    assert lr.counts.tolist() == [1]
    assert lr.jets.pt[0, 0] == pytest.approx(300 * np.cos(0.3), rel=1e-3)
    assert lr.jets.nsub[0, 0] == 2


def test_associate_uses_subjet_cones():
    small = select_small_r(_small([[(150, 0.0, 0.0), (150, 0.0, 0.6)], []]))
    lr = recluster_large_r(small)
    objs = ak.zip({"pt": ak.Array([[1.0, 1.0, 1.0], [1.0]]), "eta": ak.Array([[0.0, 0.0, 0.0], [0.0]]),
                   "phi": ak.Array([[0.1, 0.8, 1.5], [0.0]])})
    jid = associate(objs, lr)
    # 0.1 and 0.8 are within 0.4 of a subjet axis; 1.5 is 0.9 from the nearest
    assert jid.tolist() == [0, 0, -1, -1]


def test_match_jets_one_to_one():
    a = recluster_large_r(select_small_r(_small([[(300, 0.0, 0.0), (400, 0.0, 2.5)]])))
    b = recluster_large_r(select_small_r(_small([[(390, 0.05, 2.45), (280, 0.1, 0.1)]])))
    a2b, b2a = match_jets(a, b)
    # both collections are pT-ordered: a = [400@2.5, 300@0], b = [390@2.45, 280@0.1]
    assert a2b.tolist() == [0, 1] and b2a.tolist() == [0, 1]


def test_pad_groups_orders_and_truncates():
    jid = np.array([1, 0, 1, 1, -1])
    cols = {"x": np.array([5.0, 1.0, 9.0, 7.0, 3.0])}
    out = F.pad_groups(jid, 2, cols["x"], cols, 2, ["x"], [])
    assert out["x"][1].tolist() == [9.0, 7.0] and out["valid"][1].tolist() == [True, True]
    assert out["x"][0].tolist() == [1.0, 0.0] and out["valid"][0].tolist() == [True, False]


def test_model_space_transforms():
    assert F.model_space("pt", np.array([np.e])) == pytest.approx(1.0)
    assert F.model_space("c0_pt", np.array([0.0])) == 0.0
    assert F.model_space("d0", np.array([-0.01])) == pytest.approx(-np.log(2))


def test_prob_at_least_matches_brute_force():
    p = np.array([0.2, 0.5, 0.9])
    brute = 0.0
    for bits in range(8):
        k = bin(bits).count("1")
        if k >= 2:
            brute += np.prod([pi if (bits >> i) & 1 else 1 - pi for i, pi in enumerate(p)])
    assert prob_at_least(p, 2) == pytest.approx(brute)
    assert prob_at_least(np.array([0.3]), 2) == 0.0


def test_sr_efficiency_counts_events():
    event = np.array([0, 0, 1, 2, 2, 2])
    passed = np.array([1, 1, 1, 0, 1, 0], bool)
    eff, _ = sr_efficiency(event, passed, 4)
    assert eff == pytest.approx(0.25)
    pred, _ = predicted_sr_efficiency(event, passed.astype(float), 4)
    assert pred == pytest.approx(0.25)


def test_threshold_at_rejection():
    b = np.linspace(0, 1, 100001)
    thr = threshold_at_rejection(b, 1000)
    assert (b > thr).mean() == pytest.approx(1e-3, abs=2e-5)


def test_delta_phi_wraps():
    assert delta_phi(3.1, -3.1) == pytest.approx(-0.0831853, abs=1e-6)


def test_skim_chunks_keep_match_indices_consistent(tmp_path):
    """Regression: ``match`` is a row index into the other jet collection and
    must be offset per chunk, or every jet past the first chunk points at an
    unrelated jet in another event (bug found 2026-08-31)."""
    import h5py
    import numpy as np

    from jet_surrogate.skim import skim_file

    root = next(iter(sorted(Path("data/delphes").glob("signal_m5_ctau0.1mm_seed28.root"))), None)
    if root is None:
        pytest.skip("no Delphes file available")
    out = skim_file(root, tmp_path / "s.h5", chunk=200, max_events=1000)
    with h5py.File(out) as h:
        tj, rj = h["truth_jets"][...], h["reco_jets"][...]
    ok = tj["match"] >= 0
    assert ok.sum() > 0
    assert (tj["event"][ok] == rj["event"][tj["match"][ok]]).all()
    okr = rj["match"] >= 0
    assert (rj["event"][okr] == tj["event"][rj["match"][okr]]).all()


def _toy_record(z_stable, z_decayed):
    """One event: a hadron that decays to two stable daughters, plus a spectator.
    ``z_stable`` and ``z_decayed`` place the two groups along the beamline."""
    import awkward as ak

    #            hadron(status 2)   daughter   daughter   spectator
    pid = [[321, 211, -211, 211]]
    status = [[2, 1, 1, 1]]
    z = [[z_decayed, z_stable, z_stable, z_stable]]
    one = lambda v: [[v] * 4]
    return ak.zip({
        "pid": pid, "status": status,
        "pt": one(10.0), "eta": one(0.1), "phi": one(0.2), "e": one(12.0), "mass": one(0.5),
        "x": one(0.0), "y": one(0.0), "z": z,
        "d1": [[1, -1, -1, -1]], "d2": [[2, -1, -1, -1]],
    })


def test_longitudinal_features_are_relative_to_the_primary_vertex():
    """A signal simulated with a displaced beam spot must give the same truth
    features as the same signal simulated at the origin, or a theorist's
    pileup-free sample would not reproduce our pileup training inputs."""
    import numpy as np

    from jet_surrogate import features as F

    jets = np.zeros(1, dtype=np.dtype([("pt", "f4"), ("eta", "f4"), ("phi", "f4"), ("mass", "f4"),
                                       ("nsub", "i4"), ("event", "i4"), ("match", "i4"), ("n_assoc", "i4")]))
    jets["pt"] = 100.0
    ref = None
    for pv in (0.0, 50.0, -120.0):                      # the same event, placed anywhere in z
        part = _toy_record(z_stable=pv + 1.0, z_decayed=pv + 1.0)
        cols, _ = F.particle_columns(part, np.zeros(4, np.int64), jets, pv_z=F.primary_vertex_z(part))
        got = np.array([cols["prod_z"], cols["decay_z"], cols["decay_len"]])
        if ref is None:
            ref = got
        assert np.allclose(got, ref, atol=1e-4), f"features moved with the beam spot at pv={pv}"


def test_delphes_pileup_vertex_shift_is_repaired():
    """Delphes' PileUpMerger displaces only the stable particles, leaving decayed
    ones at the unshifted vertex, which would give prompt hadrons a spurious
    flight length of tens of mm (bug found 2026-09-01)."""
    import numpy as np

    from jet_surrogate import features as F

    jets = np.zeros(1, dtype=np.dtype([("pt", "f4"), ("eta", "f4"), ("phi", "f4"), ("mass", "f4"),
                                       ("nsub", "i4"), ("event", "i4"), ("match", "i4"), ("n_assoc", "i4")]))
    jets["pt"] = 100.0
    import awkward as ak

    from jet_surrogate.delphes_io import _repair_pileup_vertex_shift

    part = _toy_record(z_stable=60.0, z_decayed=0.0)     # only the stable particles were moved
    part = _repair_pileup_vertex_shift(part, np.array([60.0], dtype=np.float32))
    cols, _ = F.particle_columns(part, np.zeros(4, np.int64), jets, pv_z=F.primary_vertex_z(part))
    assert cols["decay_len"][0] < 1e-3, f"prompt hadron given a flight length of {cols['decay_len'][0]} mm"
