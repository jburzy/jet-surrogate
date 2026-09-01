# Calibration plan for the truth-level surrogate

Written 2026-08-31, after the match-index bug. Nothing here is implemented
yet: this is the plan to run once the corrected surrogates land.

## Why this is not automatic

The paper argues that the surrogate output is a probability because binary
cross-entropy is a proper scoring rule. What that actually gives is an
asymptotic statement: minimizing the *population* BCE over an
*unrestricted* function class has the unique minimizer

    f*(x) = E[y | x] = P(tagged | x)

which assumes population loss, unlimited capacity, exact optimization, and
train and test drawn from the same distribution. All four are assumptions
about our setup rather than guarantees, and each has a concrete failure
mode here:

- **Capacity.** 0.58M parameters, 4 layers, no pairwise interaction
  features. If the expressible function class does not contain the true
  conditional, the constrained BCE minimum is not the conditional
  probability and nothing forces calibration.
- **Overfitting.** Deep networks fitting hard 0/1 labels are typically
  overconfident. We are probably safe (3.5M jets vs 0.58M parameters, early
  stopping on validation loss), but that is an empirical claim.
- **Label noise.** Proven, not hypothetical: the match-index bug made the
  network fit 0.2 p_true(x) + 0.8 <p>, perfectly calibrated to the noisy
  labels and compressed against the real ones (std 0.224 where resolved
  labels imply 0.43). A proper scoring rule calibrates you to whatever
  target you actually show it.
- **Distribution shift.** Calibration measured on the training model
  transfers to an unseen dark-sector model only if the *conditional*
  P(tag | x) is invariant while P(x) shifts. Physically it should be: the
  detector and the frozen tagger do not know which Lagrangian produced the
  jet, so given a sufficient description of the jet's truth content the
  tagging probability is a property of the detector. That holds only if our
  features are sufficient. Where they are not, the residual dependence sits
  on hidden variables whose distribution moves with the model, which is
  exactly the Lambda-scan failure and the m_pi = 10 GeV degradation.

**Reframing worth keeping:** the calibration curve on an *unseen* model is a
direct test of whether the truth-level feature set is sufficient, not a
cosmetic diagnostic. It is arguably the most informative plot we make.

## A stronger requirement than the reliability diagram

Our calibration plot tests reliability, E[y | p = v] = v, binning jets by
predicted probability. The Poisson binomial (Eq. 4.2) needs the per-jet
conditional E[y | x] to be right *and* the decisions conditionally
independent, because probabilities are multiplied within an event. A model
can be reliable in aggregate while being wrong jet by jet in a way that
cancels in the binning but not in the product.

The independence half is now measured and holds: for the two leading truth
jets, E[l1 l2] exceeds <l>^2 by only 0.009-0.015 (1-2% relative) at ctau
0.01 / 0.1 / 5 mm. The per-jet half can only be bounded empirically.

## The plan

1. **Expected calibration error per sample.** Report ECE (equal-frequency
   bins, on held-out data) alongside every reliability diagram, so
   "calibrated" is a number rather than an eyeball judgement. Both chains,
   every evaluation sample, including the pileup chain where the target is
   intrinsically noisier.

2. **Recalibration transfer test.** Fit an isotonic (and, for comparison, a
   one-parameter temperature/Platt) recalibration on a held-out slice of the
   *training* model, then apply it unchanged to the unseen models
   (m_pi = 10 and 2 GeV, the held-out m_pi/Lambda points, the Z' scan).
   - If it improves the unseen models: the miscalibration is a global shape
     distortion and is fixable post hoc.
   - If it does not transfer: the problem is feature sufficiency, and no
     post-hoc calibration will help. Fix the inputs or the architecture
     instead (auxiliary per-constituent targets, pairwise interaction bias).
   This cleanly separates the two failure modes and also answers whether we
   should be applying calibration at all (currently we do not).

3. **Keep the labels-only reference as the ceiling.** SR efficiency computed
   from the tagger's true per-jet decisions carried onto truth jets. It
   isolates the bookkeeping loss (passing reco jets with no truth partner:
   -0.1% without pileup, -1.1% at mu = 60 after the fix). Any remaining gap
   for a well-calibrated surrogate is per-jet resolution, not bookkeeping.
   Caveat to state: this reference assumes a surrogate that resolves the
   detector's randomness perfectly, which no truth-only predictor can do, so
   it is an optimistic ceiling.

4. **Soften the paper.** Replace "which is a proper scoring rule and
   therefore yields calibrated probabilities when the model capacity and
   statistics permit" with the asymptotic statement plus an explicit note
   that calibration is an empirical property we verify, pointing at the
   reliability figures and the ECE table.

## Related note on what to report

SR-efficiency closure on the *training* model is a weak test: it is
dominated by the per-sample mean, so a surrogate that learned only
"lifetime -> average efficiency" passes it (ours did, at 0.90-0.95, while
its per-jet AUC against correct labels was being reported as 0.71).
Closure on *unseen* models is the strong test, because there the mean must
be inferred from truth inputs. Per-jet AUC and calibration are the direct
diagnostics and should be primary results rather than supporting plots.
