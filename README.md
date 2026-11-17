# robust_child_filtering_model

Course project (Programming for Deep Learning): a child-or-adult filtering model that is as accurate for Black teenagers as for White teenagers.

Automated child detectors used to filter text-to-image training data (Crețu et al., IEEE S&P 2026) miss Black teenagers more often than White teenagers. Our preregistered audit ([child-skintone-detection](https://github.com/isaacmutuma/child-skintone-detection), Phase 1) measured gaps of 4.0 to 10.0 percentage points on UTKFace, concentrated in ages 10 to 17. This project trains age models on frozen CLIP ViT-L/14 embeddings with fairness-aware methods, then fine-tunes CLIP itself, to test whether the gap can be closed without raising false positives on adults and without race at filtering time.

**Status:** success criteria approved ([`docs/SUCCESS_CRITERIA.md`](docs/SUCCESS_CRITERIA.md)); embedding extraction and folds written (`notebooks/01_embeddings_and_folds.ipynb`), not yet run. See [`docs/PLAN.md`](docs/PLAN.md) for the motivation, steps, models and expected results.

**Layout:** `configs/config.py` (paths, pins, frozen constants) · `src/` (CLIP loader with QuickGELU check, data, folds) · `scripts/` (Colab steps) · `notebooks/` · `results/` (aggregate JSON only) · `tests/`.

**Ethics:** no images of children are generated; only UTKFace and FairFace are used; images, embeddings and per-image outputs never enter git; figures and the demo show charts only.
