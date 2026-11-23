# Follow-up 2: does removing race from the CLIP embeddings close the gap? (LEACE)

*Written 2026-11-20, before any run. Second follow-up required by the course rules. Exploratory: no Holm correction, no success verdict; unadjusted paired CIs only.*

## The observation it comes from

Follow-up 1 (`docs/FOLLOWUP_E_FORCE_RACE.md`, `results/followups/E_force_race.json`) pushed E's adversary up to λ = 10 with both targets. A fresh linear probe still read race from the head's representation at 0.805 to 0.822 balanced accuracy (threshold for "race removed" 0.40; chance 0.20) in every setting, the gap never narrowed significantly, and the Black − White over-estimation stayed at +1.7 to +2.2 years. Adversarial training fooled its adversary instead of removing race, so the question it was meant to answer is still open: **if race is actually removed, does the gap close?**

LEACE (least-squares concept erasure; Belrose et al., NeurIPS 2023) answers this directly. It is a closed-form linear edit of the embeddings that makes them uncorrelated with race, so no linear classifier can read race from them, while moving the embeddings as little as possible. Unlike the adversary, it cannot be fooled.

## Three sentences (course rule)

1. **What I wonder (Isaac's wording, 2026-11-23):** If linear race information is actually removed from CLIP's embeddings, does the White–Black age gap close, or does it stay wide?
2. **What I expect:** see Prediction below.
3. **What I change and what I keep fixed:** I change only the input: LEACE, fitted on each fold's training rows, is applied to the standardised embeddings. Everything else is model A's exact procedure: the same 8-setting grid, selection by inner-fold MAE, refit on 4 folds, seed, folds and evaluation.

## Prediction (Isaac's, 2026-11-23, before the run)

**The gap closes.** Once linear race information is removed from the embeddings, the White − Black gap falls to about 3 points or less with age error within +0.5 years of model A: linear race information was the real culprit.

Isaac's reasoning for the test: if the covariance between the embeddings and race is zeroed and the White − Black gap shrinks with an acceptable MAE (within the +0.5-year limit of the success criteria), race was the real culprit; if not, it was not. (LEACE removes linear race information only; the MLP probe shows what non-linear race information is left, and the conclusion is limited accordingly.)

Record of how this was set: on 2026-11-23 Isaac first chose "gap stays wide" by accident in a multiple-choice question; it was committed (1a8c570) and replaced at his request by "gap closes" minutes later, before any run. Claude's earlier draft (for the record): linear probe near 0.20 but MLP probe well above chance; gap narrows by a few points at most; over-estimation shrinks but stays above 0; age error below +0.5 years in the band variant.

## Design

| Variant | What is erased | Role |
|---|---|---|
| **band (primary)** | race, after removing what each group's age-band mix explains (race one-hot residualised on the 6 fold age bands) | Removes race without also removing age, which plain LEACE would do because race and age mix are confounded in UTKFace (synthetic test: plain LEACE MAE 15.4 vs band 3.0 when groups' age mixes differ) |
| plain | race (one-hot, 5 groups) | Textbook LEACE, for comparison |

Per variant: LEACE fitted on the training rows only, inside every inner and outer split (no test data touches it); then model A's procedure; out-of-fold predictions for all 23,705 images.

## Measured per variant

- Race leakage after erasure (pooled out-of-fold, balanced accuracy, 95% bootstrap CI): linear probe and MLP probe (one hidden layer, 256 units) on the erased inputs, the same two probes on the plain inputs (reference), and a linear probe on the trained head's hidden layer.
- Gap, teen TPR by group, adult FPR, MAE, Black − White age-error gap among minors; paired vs A (unadjusted).
- Checks: LEACE rank (4 = number of races − 1), train cross-covariance after erasure, head refit reproduces the out-of-fold test MAE.

## How the result will be read (fixed before the run)

- **Race linearly removed** = linear probe on the erased inputs at most 0.40 (same threshold as follow-up 1).
- If race is removed and the gap reduction CI excludes 0 with the gap at most 0.03: removing race closes the gap (Isaac's Case A for this follow-up), and the cost (MAE, adult FPR) is reported.
- If race is removed but the gap reduction CI includes 0, or the age-error gap stays clearly above 0: removing linear race information does not fix the over-estimation (Case B), pointing to how CLIP encodes age itself, i.e. model F.
- If the MLP probe stays high, the conclusion is limited to linear race information, and this is stated.
- A gap that shrinks only because White TPR falls is levelling down and is reported as such.

## Ethics and data

Embeddings and folds only (no images opened). Per-image predictions stay on Drive; the committed JSON is aggregate.
