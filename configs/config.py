"""
configs/config.py
-----------------
Single source of truth for environment, paths, pinned sources, seeds and the
evaluation constants fixed in docs/SUCCESS_CRITERIA.md (approved 2026-11-12).
Changing a FROZEN value needs Isaac's OK and a Decision Log row.

Environment: set CRF_ENV=colab|local (default local).
"""

import os
from pathlib import Path

# ── Environment ───────────────────────────────────────────────────────────────
ENV = os.environ.get("CRF_ENV", "local")  # 'local' | 'colab'

if ENV == "colab":
    REPO_DIR = Path("/content/robust_child_filtering_model")
    DRIVE = Path("/content/drive/MyDrive")
    AUDIT_DRIVE = DRIVE / "child-detection-audit"     # audit's pinned raw data + Phase 1 outputs
    DRIVE_ROOT = DRIVE / "robust_child_filtering_model"
    DATA_DIR = Path("/content/data")                  # unpacked each session (local disk)
    # HF cache on local disk: Drive's FUSE mount corrupted large safetensors in the audit.
    HF_CACHE_DIR = Path("/content/hf_cache")
elif ENV == "local":
    REPO_DIR = Path(__file__).resolve().parents[1]
    DRIVE_ROOT = REPO_DIR / "data" / "drive"          # gitignored; tests/fixtures only
    AUDIT_DRIVE = DRIVE_ROOT / "audit"
    DATA_DIR = REPO_DIR / "data" / "local"
    HF_CACHE_DIR = REPO_DIR / "data" / "hf"
else:
    raise ValueError(f"Unknown CRF_ENV: {ENV!r}. Must be 'local' or 'colab'.")

RAW_DIR = AUDIT_DRIVE / "raw"                          # UTKFace.tar.gz (audit Phase 0)
AUDIT_PER_IMAGE_DIR = AUDIT_DRIVE / "per_image"         # Phase 1 per-image CSVs (cross-check)
# Per-image artefacts: Drive only, never git
EMB_DIR = DRIVE_ROOT / "embeddings"
FOLDS_DIR = DRIVE_ROOT / "folds"
PER_IMAGE_DIR = DRIVE_ROOT / "per_image"
RESULTS_DIR = REPO_DIR / "results"                      # aggregate JSON only (committable)
FIGURES_DIR = REPO_DIR / "figures"                      # charts only

# ── Pinned sources ────────────────────────────────────────────────────────────
SOURCES = {
    # same archive, sha256 and parse as the audit (docs/DATA_CARD.md there)
    "utkface": {
        "hf_repo": "py97/UTKFace-Cropped", "revision": "e80f21fb21fa3631380c3cb634b79be3ac2f4cba",
        "file": "UTKFace.tar.gz",
        "sha256": "2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7",
        "n_files": 23708, "n_records": 23705,
    },
    # FairFace train split (0.25 padding), for the extra-data variant only. Same mirror and
    # revision as the audit's val split; sha256 = HF LFS oids (val oid matches the audit pin).
    "fairface_train": {
        "hf_repo": "mutuma00/FairFace", "revision": "e8e670de53803408a4eaebf503d0a35749907f54",
        "repo_type": "dataset",
        "files": {
            "0.25/train-00000-of-00002-d405faba4f4b9b85.parquet":
                "acebfc4a735050d0ee618b3c690ad91e1b11bcc25151f398f06990bc15e99d06",
            "0.25/train-00001-of-00002-dd3cb68164727418.parquet":
                "e8ec1fb27f0745c166b218708bf4923811cd41d85cb998f09f8f624b0eb40326",
        },
        "licence": "CC BY 4.0",
    },
}

# OpenAI CLIP ViT-L/14 (same pin as the audit's D4). MUST load with QuickGELU.
CLIP_ARCH = "ViT-L-14"
CLIP_WEIGHTS = {
    "hf_repo": "timm/vit_large_patch14_clip_224.openai",
    "revision": "18d0535469bb561bf468d76c1d73aa35156c922b",
    "file": "open_clip_model.safetensors",
    "sha256": "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e",
}
EMB_TAG = "clip_vitl14"

# ── Labels (FROZEN; dataset-provided perceived race) ─────────────────────────
UTKFACE_RACES = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
TESTED_GROUPS = ["White", "Black", "Asian", "Indian"]   # Others trained on, not reported
PRIMARY_CONTRAST = ("White", "Black")
FAIRFACE_AGE_BINS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69",
                     "more than 70"]
FAIRFACE_GENDERS = ["Male", "Female"]
FAIRFACE_RACES = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic",
                  "Southeast Asian"]

# ── Evaluation (FROZEN, docs/SUCCESS_CRITERIA.md) ─────────────────────────────
SEED = 0
N_FOLDS = 5
N_BOOTSTRAP = 2000
ALPHA = 0.05
CHILD_AGE_THRESHOLD = 18           # flagged as child if predicted age < 18
MINOR_MAX_AGE = 17
ADULT_FPR_RANGE = (20, 59)
MINOR_STRATA = [(0, 9), (10, 17)]
STRATUM_WEIGHTS = {"0-9": 0.7234, "10-17": 0.2766}     # pooled-minor reference (audit Phase 0)
# Fold stratification: group x age band (bands only balance folds; not an endpoint)
FOLD_AGE_BANDS = [(0, 9), (10, 17), (18, 19), (20, 39), (40, 59), (60, 200)]
# Success thresholds (§4)
SUCCESS = {"small_gap_max": 0.03, "adult_fpr_max": 0.05, "adult_fpr_vs_A": 0.02,
           "mae_noninferiority_years": 0.5}
# Tuning rule v3 (FINAL, 2026-11-18): levelling-down guard used in model selection only
TUNING = {"tpr_drop_max": 0.02}
# Holm family for criterion 1 (§4: B, C, chosen D, E-invariant, E-equal-errors, each data
# variant (FairFace extra data, within-group mixup; PLAN.md) and F; C2 added 2026-11-18). Until all have run,
# missing members count as p = 1, which is conservative ('interim').
HOLM_FAMILY = ["B", "C", "C2", "D", "E_inv", "E_eq", "data_fairface", "data_mixup", "F"]

# Audit D4 prompts: used ONLY to cross-check our embeddings against Phase 1 CLIP L/14 scores
AUDIT_CHILD_PROMPTS = ["a photo of a child", "a photo of a kid", "a photo of a baby",
                       "a photo of a teenager"]
AUDIT_ADULT_PROMPTS = ["a photo of an adult", "a photo of a man", "a photo of a woman"]


def as_dict():
    """Serialisable snapshot of this config, stored in every results JSON."""
    return {k: (str(v) if isinstance(v, Path) else v)
            for k, v in globals().items()
            if k.isupper() and not k.startswith("_")}
