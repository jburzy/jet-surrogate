# Emerging jets with a track transformer tagger (Delphes ATLAS example)

This is the example analysis of the surrogate method: a fully controlled
chain in which every element is available, so the surrogate can be
confronted with the detector-level answer on models it never saw.

## The analysis being approximated

Pythia 8.312 events are passed through Delphes 3.5.1 with the ATLAS card
and ATLAS track impact-parameter resolutions. Anti-kt R = 0.4 particle-flow
jets are reclustered into R = 1.0 jets. A transformer tagger acting on the
reconstructed tracks of each large-radius jet separates dark showers from
QCD, with the working point fixed at a QCD jet rejection of 1000. The
signal region requires two tagged jets with pT > 200 GeV.

## The surrogate

A transformer with the same architecture acting on the generator particles
inside the truth large-radius jet (stable particles and decayed hadrons,
with kinematics, production and decay vertices, decay products and PDG
identities), trained to predict the tagger decision. Per-jet probabilities
are combined into the event-level signal-region probability with the
Poisson binomial. Files: `surrogate.pt` (torch), `surrogate.onnx` +
`surrogate.preprocessor.json` (ONNX with the preprocessing spec).

## Validation

See the figures on the analysis page and the paper draft in `paper/`.
In-sample closure is within 5 to 13 percent at event level. Extrapolation
in the dark-pion mass and the confinement scale is the known weakness of
this first surrogate, which was trained at a single model point.

## Provenance

| field | value |
|---|---|
| repository | https://github.com/jburzy/jet-surrogate |
| training data | seeds 35 to 54 of the nominal signal, 6 lifetimes, 1.2M events |
| tagger | seeds 1 to 24, working point on seeds 28 to 34 |
| checkpoint | see `models/surrogate/run_dir.txt` in the training repository |
