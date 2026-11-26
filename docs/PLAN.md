# Plan: a fairer child-filtering model

Written 2026-11-10, before any code or training. The master copy of the narrative is the Google Doc "Child-Detector Skin-Tone Audit: Project Reference" (section "Current focus"); this file is the working plan for the repository.

## Why this project exists

**The motivation paper.** Crețu et al. (IEEE S&P 2026, arXiv 2512.05707) asked whether removing images of children from training data stops text-to-image models from producing child sexual abuse material (CSAM). They benchmarked more than 20 child detectors. The best filter (LLaVA-7B plus a caption keyword list) caught 93.9% of child images in CC3M but also removed 35% of images without children; face-based age estimators such as MiVOLO caught 46 to 59%. Models trained on filtered data could still produce a harmless stand-in concept (a child wearing glasses) within at most 12 prompts, and fine-tuning on 1,000 child images almost removed the protection. They noted that unequal filtering rates across sub-groups bias what remains, without measuring which sub-groups.

**Our audit (Phase 1, repository child-skintone-detection).** On UTKFace, all five detectors tested caught Black minors less often than White minors, by 4.0 to 10.0 percentage points after age adjustment, all significant after Holm correction. The gap sits almost entirely in 10 to 17-year-olds (CLIP ViT-L/14: 83% of White and 55% of Black teenagers caught), and age models estimate Black minors as older than they are. Per-group thresholds close the gap only by wrongly flagging up to 58.3% of Black adults (CLIP ViT-B/32). CLIP's text encoder is not the cause; the bias is in how faces are embedded. FairFace, which only tests under-tens, showed no gap.

**This project.** Train an age model on frozen CLIP ViT-L/14 image embeddings and test whether fairness-aware training makes the child-or-adult decision at 18 as accurate for Black teenagers as for White teenagers, without raising adult false positives and without race at filtering time. Then fine-tune part of CLIP itself (model F).

## What a better model would mean for CSAM prevention

- A filter that misses Black teenagers more often leaves proportionally more of their images in training data. Since what remains shapes what a model can still generate, protection may depend on skin tone. This last step is an inference, not a measurement.
- Closing the gap makes that protection more equal and helps filtering overall, because teenagers are the hardest age group for every detector.
- Limits: filtering alone does not stop misuse (prompts and small fine-tuning sets get around it); results on cropped faces may not carry over to web images; a better teenage age estimator is dual-use (for example surveillance), so it stays on research datasets, its weights stay private and it is not released as a general tool.

## Steps

> **Class scope (2026-11-19, CLAUDE.md Decision Log):** the term project tests one question: does a fairness fix need to act inside the transformer, or is training on frozen CLIP features enough? Conditions: A (baseline), E-invariant (adversarial), F1 (LoRA in CLIP's last ViT blocks), F2 (F1 + adversary). B, C, C2 are done and kept as brief context (reported with unadjusted p, not tested). D, E-equal-errors and the data variants are deferred to the paper. Holm family: E_inv, F1-1, F1-4, F2 (F1 at two LoRA depths since 2026-11-24).

1. Repository set-up (this repo). Embeddings and per-image data stay on Drive; the CLIP loader must use QuickGELU.
2. Write the success criteria and evaluation plan (`docs/SUCCESS_CRITERIA.md`) before any training.
3. Extract CLIP ViT-L/14 image embeddings on Colab: UTKFace, plus FairFace's training split for the extra-data variant.
4. Fix 5 cross-validation folds, balanced by group and age band, seed 0; reuse for every model.
5. Port the Phase 1 metric code: stratified TPR gap, age error by group, adult FPR, worst-group TPR, overall age error.
6. Model A (baseline).
7. Models B and C, D's lambda sweep, E's two adversarial versions with the leakage probe, data variants.
8. Model F: LoRA fine-tuning of CLIP's last vision layers with the best method from A to E.
9. Demo (charts only) and write-up; schedule fixed once the pitch and demo dates are known.

## Models

| Model | Training | Question |
|---|---|---|
| A. Plain | MSE/L1 on age from frozen embeddings | Does the bias appear in a model trained this way? |
| B. Balanced sampling | Every group × age band seen equally often | Is it caused by Black teenagers being rare? |
| C. Group DRO | Minimise the worst group's loss (Sagawa et al., 2020) | Does focusing on the worst group help? |
| D. Fairness penalty | Loss + λ × spread of mean age error across groups | How much accuracy does fairness cost? |
| E. Adversarial (GAN-style) | E-invariant: alternating updates of a race adversary on z (A's 512 hidden layer) and the encoder + age head (L1 + λ × fairness term; max-error or base-rate target); leakage probe. Equal-errors version deferred | Can an adversary remove race from the decision? |
| Data variants | FairFace training split (interval loss on age bins); within-group mixup | Is it a data-quantity problem? |
| F. Fine-tuned CLIP | LoRA on CLIP's last vision layers + best method from A to E | Does changing CLIP itself close the gap the frozen models leave? |

## Expected results (predictions, not data)

| Model | Expected result | Confidence |
|---|---|---|
| A | A gap still appears, probably smaller than zero-shot CLIP's 10 points, because the model learns from UTKFace's own age labels | Medium |
| B | Shrinks the gap somewhat, at a small accuracy cost for White and Asian teenagers | Medium |
| C | Similar to B, possibly more stable; may overfit the 126 Black teenagers | Low to medium |
| D | A smooth accuracy-versus-fairness trade-off curve | High |
| E | Small effect (as Gokhale et al., 2021 found); the leakage probe still recovers race from the invariant version; equal-errors helps more | Medium |
| Data variants | Extra FairFace data improves teenagers in general; effect on the gap unclear | Low |
| F | Best chance of closing the gap if A to E stall; gains may not hold across folds because of overfitting | Low to medium |

Most likely overall: no method fully closes the gap. Balanced sampling, group DRO and the penalty shrink it at a measurable accuracy cost, the adversary adds little, and F is the real test. That is still a clear result: the bias sits in CLIP's representation and cheap fixes have limits.

## Ethics (absolute)

- No images of children are generated, by any model, for any purpose.
- No sexual content or nudity classifiers.
- No scraping; only UTKFace and FairFace, already on Drive.
- No images, embeddings, folds with image ids, or per-image outputs in git. They stay on Drive.
- Charts only in figures, demo and slides; no faces.
- Group labels are the datasets' own; never a model's prediction of race.

*Amendment 2026-11-26 (Isaac, CLAUDE.md Decision Log):* the demo shows UTKFace children's faces with their exact ages and A/E out-of-fold verdicts (fixed random draw, seed 0) in the notebook outputs, report and video, never in git; this replaces the 'no faces' rule above and the ICCWD scenes (19-24 Nov). Plan: `docs/DEMO_FACES.md`.
