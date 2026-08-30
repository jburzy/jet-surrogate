# Released surrogate

`surrogate.pt` (torch inference checkpoint) and `surrogate.onnx` (+ preprocessor
JSON) of the truth-level surrogate used by `jet-surrogate predict` and by the
reinterpretation workflow.

| field | value |
|---|---|
| trained on | nominal model, m_pid = 5 GeV, ctau in {0.01, 0.05, 0.1, 0.5, 1, 5} mm, surrogate seeds 35-54 (signal only) |
| target | detector-level tagger (transformer on Delphes tracks) at 10^3 QCD rejection, two-jet signal region pT > 200 GeV |
| labels | tagger trained on seeds 1-24, working point on seeds 28-34 (disjoint from the surrogate seeds) |
| status | PLACEHOLDER: pre charge-fix model; to be replaced by the retrained surrogate (job 33400491) |

Update this file and both model files together; the workflow reports the
commit hash with every result.
