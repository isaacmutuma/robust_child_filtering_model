# robust_child_filtering_model

Course project (Programming for Deep Learning): a child-or-adult filtering model that is as accurate for Black teenagers as for White teenagers.

Automated child detectors used to filter text-to-image training data (Crețu et al., IEEE S&P 2026) miss Black teenagers more often than White teenagers. Our preregistered audit ([child-skintone-detection](https://github.com/isaacmutuma/child-skintone-detection), Phase 1) measured gaps of 4.0 to 10.0 percentage points on UTKFace, concentrated in ages 10 to 17. This project trains age models on frozen CLIP ViT-L/14 embeddings with fairness-aware methods, then fine-tunes CLIP itself, to test whether the gap can be closed without raising false positives on adults and without race at filtering time.

**Status (2026-11-19):** embeddings and folds done; model A (gap 0.110 [0.077, 0.142]) and B, C, C2 under the final tuning rule v3 done, none closes or narrows the gap (`results/`). Model E-invariant (adversarial, notebook 06) run on 19 Nov: gap 0.106 [0.071, 0.137], reduction vs A 0.004 [-0.017, 0.024], outcome neither; the adversary was driven to chance but a fresh probe still reads race from its representation as well as from A's (balanced accuracy 0.816 vs 0.819, chance 0.20). Follow-up 1 (notebook 07, `docs/FOLLOWUP_E_FORCE_RACE.md`): pushing the adversary up to lambda 10 never removed race from the representation (probe 0.81-0.82) and never significantly narrowed the gap. Follow-up 2 (LEACE concept erasure, notebook 08) is written, not yet run. Next: F1/F2 (LoRA inside CLIP). The course submission is built in `notebooks/submission/DLP_term_project.ipynb` (skeleton). Success criteria: [`docs/SUCCESS_CRITERIA.md`](docs/SUCCESS_CRITERIA.md); plan and class scope: [`docs/PLAN.md`](docs/PLAN.md).

**Layout:** `configs/config.py` (paths, pins, frozen constants) · `src/` (CLIP loader with QuickGELU check, data, folds) · `scripts/` (Colab steps) · `notebooks/` · `results/` (aggregate JSON only) · `tests/`.

**Ethics:** no images of children are generated; only UTKFace and FairFace are used; images, embeddings and per-image outputs never enter git; figures and the demo show charts only.
