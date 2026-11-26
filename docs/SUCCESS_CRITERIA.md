# Success criteria and evaluation plan

Status: **APPROVED 2026-11-12** (drafted 2026-11-10 before any embedding, fold or model existed; §4 tightened on approval, see Decision Log). Once approved, this file is frozen; any later change goes in the Decision Log in CLAUDE.md with its date and reason.

## 1. Data and decision rule

- **Dataset:** UTKFace, all 23,705 parsed records (same source and sha256 as the audit). All five groups are used in training, including "Others". Only White, Black, Asian and Indian are reported, as in the audit.
- **Decision:** a model predicts age. An image is flagged as a child if predicted age < 18. "Minor" means a true age of 0–17. "Adult" for false positives means a true age of 20–59 (the audit's `ADULT_FPR_RANGE`).
- **Strata:** 0–9 and 10–17, weighted by the pooled-minor reference weights 0.7234 / 0.2766 (`results/phase0_counts.json`).
- **Evaluation:** 5-fold cross-validation. Folds are stratified by group × age band, use seed 0, are built once and are reused for every model. Every metric is computed on pooled out-of-fold predictions, so each image is scored by a model that never trained on it.
- **Tuning:** every hyperparameter (D's λ, C's step size, E's adversary weight, the learning rate, the epoch count) is chosen on an inner validation split inside the training folds, never on the test fold. The λ sweep for D is reported as a curve over all values; the "chosen" D uses the inner-split rule given in §4.

## 2. Reference point (from the audit, not retrained)

Zero-shot CLIP ViT-L/14 on UTKFace (`results/phase1/metrics_clip_vitl14_utkface.json`, commit 3388abd):

| | White | Black |
|---|---|---|
| Stratified TPR | 0.951 | 0.851 |
| TPR 10–17 | 0.835 (n 750) | 0.548 (n 126) |
| Adult FPR (20–59) | 0.011 | 0.009 |

White − Black stratified TPR gap: 0.100 [0.069, 0.133].

## 3. Endpoints

**Primary:** the White − Black stratified TPR gap. Each model is compared with model A through the paired difference (gap_A − gap_model) on the same images, with a 95% bootstrap CI (2,000 resamples within group, seed 0).

**Secondary** (reported for every model, by group):
1. TPR for ages 10–17, which is where the audit's gap sits.
2. Adult FPR (20–59), and the Black − White FPR difference.
3. Worst-group stratified TPR across the four tested groups.
4. Signed mean age error for minors (predicted − true), Black − White.
5. Overall age MAE, as the cost of accuracy.
6. Model E only: race-probe balanced accuracy on the learned representation (a leakage probe; chance = 0.20 for 5 groups).

## 4. What counts as success (approved 2026-11-12)

Tested groups = White, Black, Asian, Indian. A model **closes the gap** only if all four conditions hold:
1. **Fairer:** the paired test of (gap_A − gap_model) > 0 has a **Holm-adjusted p < 0.05** across the comparison family below (paired bootstrap on the same out-of-fold images, 2,000 resamples within group, seed 0). The unadjusted 95% paired CI is reported alongside.
2. **Small gap:** the model's own White − Black gap has a point estimate **≤ 0.03** *and* the upper bound of its 95% CI is **below A's point gap**. (A CI that merely contains 0 is not enough: with an MDE of ~0.07, an underpowered test would pass a gap of ~0.05.)
3. **No cost to adults:** for **every tested group**, adult FPR (20–59) ≤ 0.05 *and* ≤ A's FPR for that group + 0.02. This rules out the audit's failure mode, where equal thresholds flagged up to 58.3% of Black adults, and stops a method from moving the cost onto another group.
4. **Accuracy kept (non-inferiority):** the upper bound of the paired 95% bootstrap CI for (MAE_model − MAE_A) is ≤ **0.5 years**.

A model that meets 1, 3 and 4 but not 2 is reported as **narrowing** the gap.

**Multiple comparisons:** the primary test is run for B, C, chosen D, E-invariant, E-equal-errors, each data variant and F. Holm correction is applied across all of these.

*Amendment 2026-11-19 (Isaac approved; CLAUDE.md Decision Log):* C2 was added on 2026-11-18. For the class scope the family is now the conditions tested against A in the proposal: **E-invariant, F1, F2**; on 2026-11-24 F1 was split into two depths, so the family is **E-invariant, F1-1 (LoRA in the last ViT block), F1-4 (last 4 blocks), F2**. B, C and C2 are earlier results kept as context, reported with their unadjusted p and outside the family; D, E-equal-errors and the data variants are deferred to the paper. The earlier family-of-9 comparisons are kept in `results/comparisons_family9.json`.

**Model A itself:** if A's gap CI already contains 0, the frozen embeddings plus a trained head remove the zero-shot bias. That is a result in itself, and the comparisons in criterion 1 lose their purpose. This is reported plainly, not reframed.

## 5. Power (a planning estimate, not a result)

Normal approximation at TPR ≈ 0.85, α = 0.05 two-sided, 80% power:
- **Stratified gap, all minors:** the audit's minimum detectable effect (MDE) was 0.068 (n Black = 280). A residual gap smaller than about 0.07 can't be told apart from 0. That is why criterion 2 accepts a CI that contains 0.
- **Teen TPR (10–17):** n = 126 Black against 750 White. At TPR ≈ 0.75 / 0.85, SE ≈ √(0.75·0.25/126 + 0.85·0.15/750) ≈ 0.041, so MDE ≈ 2.8 × 0.041 ≈ 0.11. Teen results are descriptive, with CIs, and carry no test.
- **Paired differences** (criterion 1) have more power than these figures, because both models score the same images.

## 6. Reporting rules

- Every number comes from a results JSON that stores its config, its fold file hash and the git commit.
- All models are reported, including those that fail. There is no selective reporting.
- Figures show charts only. No faces, embeddings, image ids or per-image outputs go into git.
- FairFace is used only as extra training data (in the data variants). FairFace's validation set is not used as an evaluation set here, because it has no ages 10–17, where the gap is.

*Amendment 2026-11-26 (Isaac, Decision Log):* the demo (not an evaluation and not a success criterion) shows UTKFace children's faces with their exact ages and out-of-fold model verdicts in the executed notebook, report and video; faces never go into git (replaces the 'charts only, no faces' line in §6 for the demo). FairFace train is also used as extra training data in follow-up 4 (screening). The success criteria above are unchanged.
