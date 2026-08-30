# ATLAS-EXOT-2022-04: displaced hadronic jets in the calorimeter

The ATLAS Run 2 search for neutral long-lived particles decaying to
hadronic jets in the calorimeter, in association with leptons or jets
(JHEP 11 (2024) 036, arXiv:2407.09183). The collaboration published
reinterpretation BDTs with the paper (Zenodo record 12957031): one
scikit-learn random forest per selection that maps per-event generator-level
quantities to the probability of the event entering each region of the
analysis' ABCD plane. This entry wraps those published models unchanged.

## The analysis being approximated

Six selections: CR+2J (gluon-fusion-like, two additional jets), WHS low and
high ET and WALP (W boson decaying leptonically plus long-lived particles),
ZHS low and high ET (Z boson decaying leptonically). The signal region is
region A of the ABCD plane of the chosen selection.

## The surrogate

Inputs per event, computed here from the HepMC record following the
published CSV convention: for each of the two long-lived particles its pT,
eta, ET = sqrt(pT^2 + m^2), decay position Lxy and |z| in metres and the
|pdg id| of its first decay product, plus the pT and eta of the W or Z for
those selections. A single long-lived particle (ALP) is duplicated. The
models are downloaded from Zenodo on first use (about 900 MB unpacked).

The published applicability guard is applied: when the mean decay position
of the long-lived particles is outside the calorimeter range (0.25 to 16 m
in Lxy for |eta| > 1.5, 0.75 to 28 m in |z| for |eta| < 1.5) the authors'
code returns -1 for every region, and so does this predictor.

## Validation

The published `example.py` reproduces the region-A efficiencies of the
auxiliary-material tables from the authors' CSV inputs; this entry
reproduces `example.py` exactly when given those CSV features. The mapping
from HepMC to the CSV features was validated on a gluon-fusion sample
generated here (see the repository's `tests/`).

## Provenance

| field | value |
|---|---|
| models | Zenodo 12957031, scikit-learn 1.4.2 pickles, loaded with a newer scikit-learn (version warning suppressed) |
| feature code | `predictor.py` in this directory |
| maintainer of this entry | PRISM |
