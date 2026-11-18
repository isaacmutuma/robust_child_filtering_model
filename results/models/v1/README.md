# Models B and C, first version (selection rule v1)

These are the first runs of models B and C (commit de84da9), tuned with selection rule v1
(worst-group TPR among candidates with adult FPR <= 0.05). After seeing them, the rule was
tightened to v2 (post-hoc deviation, 2026-11-18): candidates must also stay within the
success limits for adult FPR (A's inner-fold FPR + 0.02 per group) and MAE (A's + 0.5 yr).
B and C were rerun under v2; those results are in results/models/. These v1 results are kept
and reported, not replaced.

The per-image OOF files these JSONs point to were moved on Drive to
per_image/v1_archive/ before the rerun; the sha256 in each JSON identifies them.
