# Follow-up 4 (screening): does adding FairFace training data narrow the gap?

Written 2026-11-26, **before any run**. Exploratory, outside the Holm family. Purpose: decide
whether the finished experiments (A, E, F) should be redone with FairFace added to training.

## Observation that led to it

Every condition so far (A, B, C, C2, E, F1-1, F1-4, F2, follow-ups 1-3) leaves the
White - Black stratified minor TPR gap at 0.095-0.115, and Black teenagers are over-aged by
about 2 years more than White teenagers. UTKFace has only 126 Black 10-17-year-olds. FairFace
train has 12,233 Black faces (86,744 in all), with race labels, so "not enough Black
teenagers in training" is the obvious remaining explanation that we have not tested.
(Against it, already observed: Asian teens are rarer in UTKFace (66) yet are detected at 74-92%;
see Session Log 2026-11-26.)

## Three sentences

- **What I wonder:** if the age head also learns from FairFace's 86,744 faces, including
  many more Black teenagers, does the White - Black gap on UTKFace shrink?
- **My prediction and why:** Isaac did not state one (the goal was set without a prediction).
  Claude's prediction, recorded here before the run: the gap shrinks by less than 2 points,
  because (a) FairFace ages are bins and its 10-19 bin straddles 18, so it cannot teach where
  18 lies; (b) FairFace ages are annotators' perceived ages, which may carry the same
  over-ageing of Black teenagers; (c) the bias is already in CLIP's embedding (follow-up 3).
- **What I change and keep fixed:** only the training data (UTKFace training folds + all of
  FairFace train, interval L1 loss on FairFace's bins). Fixed: UTKFace folds and test folds,
  CLIP L/14 embeddings, A's head, A's chosen setting per fold (from `results/models/A.json`),
  epoch chosen on the inner UTKFace validation fold by MAE (A's rule), evaluation code.

## Design

All conditions are run by one script (`scripts/followup_fairface_data.py`) on the committed
T4 embeddings on Drive (notebook 01; sha256 checked against results/embeddings), on Colab.
Every comparison is made against
**A rebuilt in the same script** (`A_local`), not the committed A; the committed A gap
(0.110) is only a sanity check.

| Condition | Training rows (per test fold k) | Loss |
|---|---|---|
| `A_local` | UTKFace folds != k | L1 |
| `A+FF` | UTKFace folds != k + all FairFace train | L1 (UTK) + interval L1 (FF) |
| `A_FF` | FairFace train only (one model, predicts all UTKFace) | interval L1 |
| `A_half`, `A_quarter` | 50% / 25% of UTKFace training rows (within group x band, seed 0) | L1 |

- **Interval L1:** for a FairFace face in bin [lo, hi] (e.g. 10-19 -> [10, 20); "more than 70"
  -> [70, 100)) the loss is 0 if the prediction lies in the bin, else the distance to the nearest
  edge. Same target standardisation as A (UTKFace training mean/SD).
- **Settings:** A's chosen setting per fold (hidden, lr, wd), epochs <= 60 chosen on UTKFace
  inner validation fold (k+1)%5 by MAE, then refit on all training rows. `A_FF`: epochs chosen
  on a 10% FairFace hold-out by interval loss (no UTKFace data used).
- **Metrics:** `src/metrics.py` unchanged (stratified TPR gap, teen TPR, adult FPR,
  B - W minor age error, MAE); paired bootstrap (2,000, seed 0) vs `A_local`.

## Reading rules (fixed before the run)

- **FairFace likely helps -> redo experiments with FairFace:** `A+FF` reduces the gap vs
  `A_local` by **>= 0.02** (point) with one-sided paired **p < 0.05**, AND the MAE difference's
  upper CI <= +0.5 yr, AND every tested group's adult FPR <= min(0.05, A_local + 0.02).
  Then E (and F if E+FF helps) would be rerun with FairFace in training.
- **FairFace does not help -> keep the existing experiments:** reduction < 0.02 or CI
  includes 0. Report as a data-quantity check (a fourth follow-up).
- **Learning curve (supporting):** if the gap at 25% / 50% of UTKFace is within 0.02 of the gap at
  100%, the gap does not depend on the amount of training data in this range, which makes more
  data an unlikely fix. `A_FF` is descriptive (does the gap exist without UTKFace's labels?).

## Ethics

Same pinned UTKFace and FairFace sources as the audit; no new data, no images or embeddings
in git; only aggregate JSON (`results/followups/fairface_data.json`) is committed.

## Result (2026-11-26, Colab T4; added after the run, plan text above unchanged)

Run: repo HEAD ce8fc11 + this script (uploaded via Drive, uncommitted at run time); Drive
embeddings sha256 = committed manifests (checked in the script); fold file sha = committed;
interval-loss path = A's fit_head exactly (max diff 0). JSON: Drive
`robust_child_filtering_model/results/followups/fairface_data.json` (48 KB; not yet copied to
the repo). Numbers below are from the script's printed summary of that JSON.

| Condition | W-B gap | reduction vs A_local (p, one-sided) | Teen TPR B / W | B-W minor age error | MAE |
|---|---|---|---|---|---|
| A_local | 0.105 | - | 0.476 / 0.792 | +2.05 | 4.41 |
| A+FF (UTK + 86,744 FairFace) | 0.112 | -0.007 (0.80) | 0.484 / 0.819 | +1.89 | 4.33 |
| A_half (50% UTK) | 0.115 | -0.010 (0.84) | 0.460 / 0.780 | +2.01 | 4.62 |
| A_quarter (25% UTK) | 0.119 | -0.014 (0.87) | 0.460 / 0.791 | +2.22 | 4.86 |
| A_FF (FairFace only) | 0.102 | +0.003 (0.37) | 0.500 / 0.797 | +2.02 | 5.31 |

**Reading (rules fixed above): FairFace does not help -> keep the existing experiments.**
A+FF lowers MAE by 0.08 yr and lifts White teen TPR, but the gap does not narrow (point -0.007).
Learning curve: 4x more UTKFace data (25% -> 100%) moves the gap only 0.119 -> 0.105 and
over-estimation 2.22 -> 2.05, so data quantity is not what drives the gap. A model trained on
FairFace alone (no UTKFace labels) shows the same gap (0.102) and the same over-ageing of
Black minors (+2.02), so the bias is not specific to UTKFace's labels. Consistent with
follow-up 3: the bias is in CLIP's embedding. Claude's prediction (< 2 points) was right.
