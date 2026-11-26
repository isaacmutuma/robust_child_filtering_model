# Follow-up 3: at which depth of CLIP does the Black–White age bias appear?

*Written 2026-11-26, before any run. Exploratory: no Holm correction, no success verdict; bootstrap CIs only. Explains model F's result; it is not a new condition.*

## The observation it comes from

- **Model F** (LoRA in CLIP's last 1 or 4 blocks, run 1 and the GradScaler rerun) did not narrow the White − Black gap: F1-1 reduction −0.011 (run 1) / −0.003 (rerun), F1-4 −0.020 (run 1), CIs include 0 or show a wider gap.
- **LEACE** (follow-up 2) removed linear race information from the final embedding (probe 0.200) and the gap stayed (0.100); the Black − White over-estimation of minors was unchanged (+1.94 vs +1.90 years).
- The audit's MiVOLO and zero-shot CLIP show the same Black over-estimation **before** any UTKFace training.

F changes only blocks 20–23 (of 24). If the bias is already in the tokens that enter block 20, F works downstream of where it forms, and its failure has a structural reason. If the bias only forms in blocks 20–23, F could reach it, and its failure must come from the objective (L1 loss does not target the gap) or the data.

## Three sentences (course rule)

1. **What I wonder:** At which block of CLIP does the over-estimation of Black children's age, and the White − Black detection gap, appear, and is it already there before the blocks that model F fine-tunes?
2. **What I expect:** see Prediction below.
3. **What I change and what I keep fixed:** I change only the depth the age is read from (the CLS token or the mean patch token after each of CLIP's 24 blocks, plus the raw patch embeddings). Everything else is fixed: frozen CLIP ViT-L/14 (same pinned weights), the same 5 folds and seed, one simple readout for every depth (ridge regression on age, its strength chosen on the inner validation fold), and the same metrics and bootstrap as every model.

## Prediction (Isaac's, before the run)

**Formed before F** (Isaac, 2026-11-26, chosen from the four reading cases below before any run): by block 20 the over-estimation of Black children and the White − Black gap are already at least 75% of their final values, so F could only adjust a bias that was already present, which explains why it failed.

## Design

- **Representations (49 + 1):** for depth 0 (patch + position embedding after `ln_pre`) and after each block 1–24: the CLS token and the mean of the 256 patch tokens. Depth-0 CLS is skipped, because it is identical for every image. Anchor: the final CLIP embedding (model A's input).
- **Age readout:** ridge regression on standardised features, out-of-fold. For test fold k, alpha ∈ {0.1, 1, 10, 10², 10³, 10⁴, 10⁵} is chosen by MAE on validation fold (k+1) mod 5, then refit on the 4 training folds. Linear and identical at every depth, so the depths are compared fairly. It is not model A: the final-embedding row shows how close a linear readout comes to A.
- **Metrics per representation** (same code as every model, `src/metrics.py`): stratified White − Black TPR gap, teen gap, Black − White over-estimation of minors, adult FPR difference, MAE. Added: an **age-matched** Black − White error gap (per true year 0–17, pooled-minor weights), because weak readouts regress toward the mean age, and groups whose minors are older within a stratum would otherwise look different for that reason alone.
- **Race readability per depth:** multinomial logistic probe (L2 as sklearn C=1), pooled out-of-fold balanced accuracy, chance 0.20.
- **Checks:** the final embedding must equal F's stored exact embeddings; depth-20 features must equal F's stored tokens entering block 20 (fp16 rounding only).

## How the result will be read (fixed before the run)

"Share" = value at a depth / value at the final embedding, for the CLS readout (primary; mean-patch as robustness).

- **Formed before F:** at depth 20 (the input to F1-4's first LoRA block), the TPR gap and the Black − White over-estimation are each at least **75%** of the final value, and both CIs exclude 0. Then F could only adjust a bias that was already present, and its failure has a structural reason.
- **Formed in F's blocks:** both are at most **25%** of the final value at depth 20. F could reach the bias; its failure is not about depth but about the objective or the data.
- **Partly before F:** anything in between.
- **Emergence depth:** the first depth from which the CI of the over-estimation excludes 0 at every deeper layer. If it is within the first few blocks, the bias is tied to low-level image features (skin tone, lighting), not to an abstract "age" concept.
- If a gap shrinks only because the readout gets worse (MAE far above the final readout), this is reported as a readout limit, not as "no bias".
- Limits fixed in advance: linear readout (non-linear age information is not measured); UTKFace labels could themselves contain the bias (this test cannot separate label error from model bias, because every depth is trained on the same labels).

## Ethics and data

Same UTKFace images and folds as every model. Features (~2.4 GB) and per-image predictions stay on the course server / Drive, never in git. The committed JSON is aggregate only. No images are shown.

## Run

`python scripts/followup_layers.py --work ~/fwork` on the course GPU server (UTKFace and F's caches are already in `~/fwork`), after the F rerun has finished. Expected time: ~10 min extraction + ~10 min readouts (estimate). Output `results/followups/layers.json`.
