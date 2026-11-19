# robust_child_filtering_model

Course project (Programming for Deep Learning): a child-or-adult filtering model that is as accurate for Black teenagers as for White teenagers.

Automated child detectors used to filter text-to-image training data (Crețu et al., IEEE S&P 2026) miss Black teenagers more often than White teenagers. Our preregistered audit ([child-skintone-detection](https://github.com/isaacmutuma/child-skintone-detection), Phase 1) measured gaps of 4.0 to 10.0 percentage points on UTKFace, concentrated in ages 10 to 17. This project trains age models on frozen CLIP ViT-L/14 embeddings with fairness-aware methods, then fine-tunes CLIP itself, to test whether the gap can be closed without raising false positives on adults and without race at filtering time.

**Status (2026-11-19):** embeddings and folds done; model A (gap 0.110 [0.077, 0.142]) and B, C, C2 under the final tuning rule v3 done, none closes or narrows the gap (`results/`). Model E-invariant (adversarial, with a leakage probe on its representation) is written and tested on synthetic data (`notebooks/06_model_E.ipynb`), not yet run. Next: F1/F2 (LoRA inside CLIP). Success criteria: [`docs/SUCCESS_CRITERIA.md`](docs/SUCCESS_CRITERIA.md); plan and class scope: [`docs/PLAN.md`](docs/PLAN.md).

**Layout:** `configs/config.py` (paths, pins, frozen constants) · `src/` (CLIP loader with QuickGELU check, data, folds) · `scripts/` (Colab steps) · `notebooks/` · `results/` (aggregate JSON only) · `tests/`.

**Ethics:** no images of children are generated; only UTKFace and FairFace are used; images, embeddings and per-image outputs never enter git; figures and the demo show charts only.
