# Models B, C and C2 under selection rule v2

Runs at commit cfab668, tuned with rule v2 (adult FPR <= A's inner-fold FPR + 0.02 per group
and <= 0.05; MAE <= A's + 0.5 yr). C2 here is uncapped and collapsed onto one tiny group in
folds 2 and 4. After seeing these results, the rule was extended to v3 (adds: every group's
stratified minor TPR >= A's - 0.02, against levelling down) and frozen as final, and C2's
DRO weights were capped to 0.1-10x each group's natural share (post-hoc, 2026-11-18). B, C
and C2 were rerun under v3; those results are in results/models/. These v2 results are kept
and reported, not replaced.

The per-image OOF files these JSONs point to were moved on Drive to per_image/v2_archive/
before the rerun; the sha256 in each JSON identifies them.
