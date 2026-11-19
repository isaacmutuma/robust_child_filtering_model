"""Train a model with 5-fold out-of-fold predictions and write its evaluation report.

  CRF_ENV=colab python scripts/train_model.py --model A   (then B, C)
Inputs (Drive): UTKFace embeddings npz and fold CSV, each checked against the sha256 in
its manifest under DRIVE_ROOT/results/.
Per-image output (Drive only): PER_IMAGE_DIR/oof__<model>__seed0.csv (id, fold, pred_age).
Aggregate output (committable): results/models/<model>.json with tuning record, report
with bootstrap CIs, input hashes, git commit and config. No per-image fields.
Models other than A also get a paired comparison with A on the same bootstrap resamples
(A's OOF file is checked against the sha256 in results/models/A.json on Drive). The
success verdict needs Holm across the whole family: see scripts/compare_models.py.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from _common import REPO, atomic_copy, sha256
from configs import config as C
from src import metrics as M
from src.provenance import write_result
from src.leakage import summarise as leakage_summary
from src.train import (FIXED_A, FIXED_B, FIXED_C, FIXED_C2_CAP, FIXED_E, GRID_A, GRID_B, GRID_C,
                       GRID_C2_CAP, GRID_E, oof_predictions)

MODELS = {
    "A": {"grid": GRID_A, "fixed": FIXED_A, "select": "mae",
          "desc": "plain L1 regression, no group info"},
    # B, C: rule v1 (results/models/v1/), then v2 (results/models/v2/, with C2 uncapped),
    # now the final rule v3 (CLAUDE.md Decision Log, 2026-11-18)
    "B": {"grid": GRID_B, "fixed": FIXED_B, "select": "worst_group_v3",
          "desc": "race-balanced sampling within each age band (band shares kept)"},
    "C": {"grid": GRID_C, "fixed": FIXED_C, "select": "worst_group_v3",
          "desc": "group DRO over race x age band, losses scaled by band mean across races; "
                  "group weights start equal"},
    "C2": {"grid": GRID_C2_CAP, "fixed": FIXED_C2_CAP, "select": "worst_group_v3",
           "desc": "as C, but group weights start at each cell's natural share "
                   "(eta = 0 equals plain training), capped to 0.1-10x that share"},
    # E-invariant (2026-11-19): GAN-style alternating updates, adversary predicts race from z
    "E_inv": {"grid": GRID_E, "fixed": FIXED_E, "select": "worst_group_v3",
              "desc": "adversarial: encoder + age head trained with L1(age) + lambda * "
                      "(-CE(race | z) or KL(base rates || adversary)) against a race "
                      "adversary updated in alternation (5 steps, lr 1e-2); leakage probe"},
}


def load_A_record():
    """Model A's per-fold tuning record from results/models/A.json on Drive."""
    man = json.loads((Path(C.DRIVE_ROOT) / "results" / "models" / "A.json").read_text())
    return [{"chosen": f["chosen"], "epochs": f["epochs"], "inner_val_mae": f["inner_val_mae"],
             "test_mae": f["test_mae"]} for f in man["folds"]]


def load_A_oof(ids):
    """Model A's OOF predictions aligned to `ids`, after a sha256 check against A.json."""
    import pandas as pd
    man = json.loads((Path(C.DRIVE_ROOT) / "results" / "models" / "A.json").read_text())
    if man["git"]["dirty"] is not False:
        raise SystemExit("A.json came from modified code")
    p = Path(C.DRIVE_ROOT) / man["oof_file"]["drive_path"]
    if sha256(p) != man["oof_file"]["sha256"]:
        raise SystemExit(f"{p} does not match A.json")
    a = pd.read_csv(p, dtype={"id": str}).set_index("id")
    if set(a.index) != set(ids.tolist()):
        raise SystemExit("A's OOF file covers different images")
    return a.loc[ids, "pred_age"].to_numpy(), man["oof_file"]["sha256"]


def load_checked(manifest_rel, key):
    """Path of a Drive artefact after checking its sha256 against its manifest."""
    man = json.loads((Path(C.DRIVE_ROOT) / "results" / manifest_rel).read_text())
    if man["git"]["dirty"] is not False:
        raise SystemExit(f"{manifest_rel} came from modified code")
    p = Path(C.DRIVE_ROOT) / man[key]["drive_path"]
    if sha256(p) != man[key]["sha256"]:
        raise SystemExit(f"{p} does not match its manifest")
    return p, man[key]["sha256"]


def load_inputs():
    """(ids, X, ages, groups, folds, hashes), all aligned and sorted by id."""
    import pandas as pd
    emb_p, emb_h = load_checked(f"embeddings/utkface__{C.EMB_TAG}.json", "npz")
    fold_p, fold_h = load_checked(f"folds/utkface_folds_seed{C.SEED}.json", "fold_file")
    z = np.load(emb_p)
    order = np.argsort(z["ids"], kind="stable")
    ids = z["ids"][order]
    f = pd.read_csv(fold_p, dtype={"id": str}).set_index("id")
    if set(f.index) != set(ids.tolist()) or len(f) != len(ids):
        raise SystemExit("fold file and embeddings cover different images")
    folds = f.loc[ids, "fold"].to_numpy()
    assert len(ids) == C.SOURCES["utkface"]["n_records"], len(ids)
    return (ids, z["emb"][order].astype(np.float32), z["age"][order].astype(float),
            z["group"][order], folds, {"embeddings_sha256": emb_h, "folds_sha256": fold_h})


def main():
    import pandas as pd
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    a = ap.parse_args()
    spec = MODELS[a.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    ids, X, ages, groups, folds, hashes = load_inputs()
    ref_A = load_A_record() if spec["select"] in ("worst_group_v2", "worst_group_v3") else None
    probe_out = {} if spec["fixed"].get("leakage_probe") else None
    pred, record = oof_predictions(X, ages, folds, spec["grid"], spec["fixed"], device,
                                   groups=groups, select=spec["select"], ref_A=ref_A,
                                   probe_out=probe_out)

    oof = pd.DataFrame({"id": ids, "fold": folds, "pred_age": pred})
    dst = Path(C.PER_IMAGE_DIR) / f"oof__{a.model}__seed{C.SEED}.csv"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / dst.name
        oof.to_csv(tmp, index=False)
        digest = sha256(tmp)
        atomic_copy(tmp, dst)

    d = M.prepare(M.frame(ids, groups, ages, pred))
    plan = M.BootPlan(d)
    rep = M.model_report(d, plan)
    payload = {"model": a.model, "description": spec["desc"], "inputs": hashes,
               "grid": spec["grid"], "fixed": spec["fixed"], "select": spec["select"],
               "device": device,
               "torch": torch.__version__, "folds": record, "report": rep,
               "oof_file": {"drive_path": str(dst.relative_to(C.DRIVE_ROOT)), "sha256": digest},
               "seconds": round(time.time() - t0, 1)}
    if probe_out is not None:
        payload["leakage_probe"] = leakage_summary(probe_out, groups)
    if a.model == "A":
        payload["a_self_check"] = M.a_self_check(rep)
    else:
        predA, shaA = load_A_oof(ids)
        dA = M.prepare(M.frame(ids, groups, ages, predA))
        payload["paired_vs_A"] = {**M.paired_vs_A(dA, d, plan), "A_oof_sha256": shaA,
                                  "note": "unadjusted; Holm across the family in compare_models"}
    res = write_result(REPO / "results" / "models" / f"{a.model}.json", payload, C.as_dict(), REPO)

    g = rep["tpr_gap_W_minus_B"]
    print(f"\nmodel {a.model}: W-B stratified TPR gap {g['est']:.3f} "
          f"[{g['ci'][0]:.3f}, {g['ci'][1]:.3f}], MAE {rep['mae_overall']['est']:.2f}")
    for grp, r in rep["by_group"].items():
        print(f"  {grp:7s} TPR {r['tpr_strat']['est']:.3f}  teen {r['tpr_teen']['est']:.3f} "
              f"(n {r['n_teen_10_17']})  adult FPR {r['fpr_adult']['est']:.3f}")
    if "paired_vs_A" in payload:
        pr = payload["paired_vs_A"]
        gr, mi = pr["gap_reduction_A_minus_M"], pr["mae_increase_M_minus_A"]
        print(f"  vs A: gap reduction {gr['est']:.3f} [{gr['ci'][0]:.3f}, {gr['ci'][1]:.3f}] "
              f"p(one-sided, unadjusted) {gr['p_one_sided']:.4f}; MAE change {mi['est']:+.2f} "
              f"[{mi['ci'][0]:+.2f}, {mi['ci'][1]:+.2f}]")
    if probe_out is not None:
        lp = payload["leakage_probe"]
        print(f"  leakage probe, balanced accuracy (chance {lp['chance']:.2f}):")
        for n, r in lp["bal_acc"].items():
            print(f"    {n:9s} {r['est']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")
    print("->", dst, "\n->", res)


if __name__ == "__main__":
    main()
