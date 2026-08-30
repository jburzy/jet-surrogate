"""jet-surrogate: a truth-level surrogate for a detector-level emerging-jet tagger.

Chain:  Pythia8 -> Delphes (ATLAS card, smeared tracks)
        -> anti-kt R=0.4 jets reclustered into R=1.0 jets (reco and truth)
        -> transformer tagger on Delphes tracks  (signal region: 2 tagged jets)
        -> transformer surrogate on truth particles predicting the tagger decision
"""

__version__ = "0.2.0"
