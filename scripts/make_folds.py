"""Build the 5 UTKFace cross-validation folds once (seed 0) from the embedding file's ids.

  CRF_ENV=colab python scripts/make_folds.py
Per-image fold file (Drive only): FOLDS_DIR/utkface_folds_seed0.csv (id, fold).
Aggregate manifest (committable): results/folds/utkface_folds_seed0.json with the fold
file's sha256 and counts per fold by group x age band. Refuses to overwrite an existing
fold file with different contents: folds are frozen once made.
"""

import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from _common import REPO, atomic_copy, sha256
from configs import config as C
from src.folds import age_band, make_folds, strat_keys
from src.provenance import write_result


def main():
    import pandas as pd
    z = np.load(Path(C.EMB_DIR) / f"utkface__{C.EMB_TAG}.npz")
    ids, groups, ages = z["ids"].tolist(), z["group"].tolist(), z["age"].astype(float).tolist()
    folds = make_folds(ids, groups, ages)
    assert folds == make_folds(ids[::-1], groups[::-1], ages[::-1]), "folds depend on order"

    df = pd.DataFrame({"id": sorted(folds), "fold": [folds[i] for i in sorted(folds)]})
    dst = Path(C.FOLDS_DIR) / f"utkface_folds_seed{C.SEED}.csv"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / dst.name
        df.to_csv(tmp, index=False)
        digest = sha256(tmp)
        if dst.exists() and sha256(dst) != digest:
            raise SystemExit(f"{dst} exists with different contents; folds are frozen")
        if not dst.exists():
            atomic_copy(tmp, dst)

    by = Counter((folds[i], g, age_band(a)) for i, g, a in zip(ids, groups, ages))
    cells = sorted({(g, b) for _, g, b in by})
    counts = {f"{g}|{b}": [by.get((f, g, b), 0) for f in range(C.N_FOLDS)] for g, b in cells}
    merged = sorted({k for k in strat_keys(groups, ages, C.N_FOLDS) if k.endswith("rare")})
    res = write_result(REPO / "results" / "folds" / f"utkface_folds_seed{C.SEED}.json", {
        "n_images": len(df), "n_folds": C.N_FOLDS, "seed": C.SEED,
        "fold_file": {"drive_path": str(dst.relative_to(C.DRIVE_ROOT)), "sha256": digest},
        "fold_sizes": np.bincount(df["fold"], minlength=C.N_FOLDS).tolist(),
        "counts_by_group_band_per_fold": counts, "merged_rare_keys": merged,
        "inner_val_rule": "test fold k -> validate on fold (k+1) % 5, train on the other 3; "
                          "refit on all 4 training folds with chosen settings",
    }, C.as_dict(), REPO)
    print("fold sizes", np.bincount(df["fold"]).tolist(), "| Black 10-17 per fold",
          counts.get("Black|10-17"), "\n->", dst, "\n->", res)


if __name__ == "__main__":
    main()
