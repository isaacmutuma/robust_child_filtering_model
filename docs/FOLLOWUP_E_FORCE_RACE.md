# Follow-up 1 (model E): does forcing race out of z close the gap?

*Written 2026-11-20, before any run. Exploratory follow-up required by the course rules ("two own follow-up questions that arise from observed results"). Not a tested condition: no Holm correction, no success verdict; unadjusted paired CIs only.*

## The observation it comes from

Model E-invariant (notebook 06, `results/models/E_inv.json`, commit 1017a04) drove its adversary to chance (refit balanced accuracy 0.20 to 0.28 in four of five folds) but a fresh linear probe read race from E's representation z as well as from model A's (0.816 vs 0.819; chance 0.20). The gap did not move (reduction 0.4 points, 95% CI −1.7 to 2.4). Tuning chose the "maximise the adversary's error" target in every fold; the base-rate target, which removed more race in the synthetic test, mostly broke the cost limits (feasible in 14, 9, 2, 0 of 20 trials at λ 0.1, 0.3, 1, 3).

So E answered "can race be removed **within the cost limits**?" (no). It did not answer: **if race is actually removed from z, does the gap close?** If it does, the obstacle is the cost; if it does not, removing race is the wrong lever and the over-estimation is carried by age cues themselves, which points at CLIP's representation and model F.

## Three sentences (course rule)

1. **What I wonder:** if adversarial training is pushed hard enough to remove race from the age model's representation, ignoring the cost limits, does the White − Black gap at 18 shrink?
2. **What I expect:** see Prediction below.
3. **What I change and what I keep fixed:** I change only the adversarial target and its weight λ, in a dose-response sweep with no tuning rule. Everything else is fixed: UTKFace embeddings, the 5 folds, E's architecture (z = 512-wide hidden layer, adversary MLP(128), lr 1e-2, 5 adversary steps per encoder step), learning rate 3e-4, weight decay 1e-4, 60 epochs (the full training budget), seed 0 + 1000·fold, evaluation and bootstrap resamples as for every model.

## Prediction (Isaac's, adopted 2026-11-20 before the run; first drafted by Claude)

As λ rises with the base-rate target, the probe's balanced accuracy falls well below A's 0.82, and age error rises past the 0.5-year limit. The gap shrinks much less than race leakage does, and the Black − White age over-estimation (A: +1.9 years) stays clearly above zero, because the cues that make Black teenagers look older are age cues in CLIP's embedding, not a separate race signal the head can drop. Confidence: medium.

## Isaac's two cases (2026-11-20, before the run)

- **Case A (probe drops, gap closes, MAE jumps):** removing race does fix the racial gap, showing that the +0.5-year MAE budget was too tight to allow debiasing on frozen CLIP.
- **Case B (probe drops, gap stays wide, MAE jumps):** even when age accuracy is sacrificed to force race out of z, the over-estimation of Black teens does not go away, pointing to frozen CLIP embeddings that tie age cues to race, so that unfreezing CLIP (F1, F2) is the path to test next.

The prediction above expects Case B. Caveats agreed for the write-up: Case B shows that removing *linearly readable* race from the head's z does not fix the over-estimation; it does not rule out every frozen-feature method (D was deferred), and it does not show that F will work. Outcomes outside A and B (race never removed; gap closed by lowering White TPR) are read with the rules below.

## Design

| Setting | Target | λ | Purpose |
|---|---|---|---|
| control | none | 0 | Same budget (60 epochs, lr 3e-4) without an adversary; separates the adversary's effect from the longer training |
| prior-0.1 to prior-10 | base rates | 0.1, 0.3, 1, 3, 10 | Dose-response: increasing pressure to remove race |
| max_ce-1, max_ce-10 | maximise error | 1, 10 | The target tuning chose for E, pushed harder |

8 settings × 5 folds, one refit per fold on the 4 training folds (no inner tuning: the point is to apply a fixed pressure, not to pick the best trade-off), out-of-fold predictions for all 23,705 images.

## Measured per setting

- Leakage probe: balanced accuracy of a fresh logistic regression predicting race from z (fitted on training folds, scored on the test fold, pooled), with a 95% bootstrap CI. Reference: A's z 0.819 and raw CLIP 0.823 (from the E run).
- White − Black stratified TPR gap (main), teen TPR by group, adult FPR by group, MAE, Black − White age-error gap among minors (the mechanism).
- Paired vs A (same bootstrap resamples, unadjusted): gap reduction, MAE increase. Also paired vs the λ = 0 control.
- Adversary balanced accuracy per epoch (stability).

## How the result will be read (fixed before the run)

- **Race removed** = probe balanced accuracy at most 0.40 (halfway between A's 0.82 and chance 0.20 is 0.51; 0.40 is clearly closer to chance).
- If some setting removes race and its paired gap reduction CI excludes 0 and the gap falls to at most 0.03: removing race closes the gap, and the cost limits are what stopped E.
- If race is removed but the gap reduction CI includes 0 (or the age-error gap stays clearly above 0): removing race is not enough; the over-estimation is carried by age cues, which points to model F.
- If no setting removes race (probe stays above 0.40 even at λ 10): adversarial training on a frozen head cannot remove race from z at any tested strength; report as such.
- Any gap reduction that comes with a large drop in White TPR is levelling down and is reported as such.

## Ethics and data

Embeddings and folds only (no images opened). Per-image predictions stay on Drive; the committed JSON is aggregate.
