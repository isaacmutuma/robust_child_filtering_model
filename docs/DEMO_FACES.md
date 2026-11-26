# Demo closing scene: real children's faces with exact ages (plan, 2026-11-26)

**Superseded draft:** this file first planned FairFace children's faces. Isaac then chose
(2026-11-26, option 1, Decision Log) **UTKFace children's faces**, because FairFace's 10-19 age
bin includes 18- and 19-year-olds, so no single FairFace verdict can be shown to be wrong.
UTKFace has the exact age of every face and is the dataset every reported number comes from.
Written before any demo face is drawn; the selection rule below is fixed now (no cherry-picking).

## What each face shows

"**true age 15** | A: 19.2 -> not a child | E: 18.6 -> not a child", coloured by verdict
(right / wrong). Predictions are **out-of-fold** (the A and E models that never trained on that
face, same OOF predictions as every reported number), so the faces show exactly what the results
measure.

## Scene (~90 s of the video; notebook section 10)

1. **The numbers (chart, ~20 s).** Teen (10-17) detection rate, Black vs White, A and E, with
   95% CIs (from the results JSONs; existing Figure 1 or a two-model cut of it).
2. **The faces (~50 s).** Grid, two rows: Black teenagers (top), White teenagers (bottom), ages
   10-17, 6 per row. Under each: true age, A's and E's predicted age and verdict. Narration walks
   through 2-3 faces per row: "She is 15. Both models put her over 18, so the filter keeps her
   photo as an adult's."
3. **Control (~10 s).** Same layout, ages 3-9, 4 per row: young children are caught in both groups.
4. **Back to the question (~10 s).** Neither the adversary (E) nor fine-tuning (F) moved the gap;
   the bias is in CLIP's embedding before our training (follow-up 3), and more training data did
   not help (follow-up 4).

## Face-selection rule (fixed now, before looking)

- Pool: UTKFace, race Black or White; teens = true age 10-17; control = 3-9.
- `numpy.random.default_rng(0)` over the pool sorted by image id: **3 female + 3 male per race**
  (teens), **2 + 2 per race** (control). No re-draws, no manual swaps; seed and rule printed in
  the notebook. The grid shows whatever the draw gives, including faces the models get right;
  the chart carries the rate, the faces illustrate it.
- If the drawn faces happen to miss the pattern, they are shown anyway and the narration says
  so (the rate is in the chart).

## Handling rules

- Faces appear only in: the executed notebook's outputs (to the professor / Drive), the report
  PDF and the video (Isaac, 26 Nov: faces everywhere). Never in git: the repo notebook stays output-stripped; no
  face images under `figures/`. No file names or ids shown.
- Faces shown as the dataset provides them (aligned, cropped); no edits.
- Credit: UTKFace (Zhang, Song & Qi, CVPR 2017), non-commercial research use; copyright of the
  images remains with their owners (dataset terms), stated in the notebook, report and video.
- No generated images. ICCWD dropped from the demo and the server run.

## Build steps

1. Notebook section 10: replace the ICCWD demo with this scene (images from the UTKFace archive the
   notebook already downloads and sha-checks; OOF predictions of A and E from the notebook's own
   runs); synthetic test first.
2. Remove ICCWD from `scripts/server_submission_run.sh` and section 10.
3. Report demo paragraph + chart + face grid; video narration from
   the measured faces.
