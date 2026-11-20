"""Follow-up 1 for model E: does forcing race out of z close the gap? (docs/FOLLOWUP_E_FORCE_RACE.md)

  CRF_ENV=colab python scripts/followup_e_force_race.py [--settings control,prior-1 ...]
Dose-response sweep with no tuning rule: each setting is E's architecture with a fixed
target and lambda, trained 60 epochs on the 4 training folds of each test fold, then scored
out of fold, with the leakage probe on z. Exploratory: unadjusted paired CIs, no Holm.
Per-image OOF predictions -> Drive (PER_IMAGE_DIR); aggregate JSON ->
results/followups/E_force_race.json.
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from _common import REPO, atomic_copy, sha256
from configs import config as C
from src import leakage as LK
from src import metrics as M
from src.provenance import write_result
from src.train import FIXED_E, Standardiser, fit_head
from train_model import load_A_oof, load_inputs

BASE = {"hidden": 512, "lr": 3e-4, "weight_decay": 1e-4}
EPOCHS = 60
SETTINGS = {
    "control": {"adv_lambda": 0.0, "adv_target": "prior"},
    "prior-0.1": {"adv_lambda": 0.1, "adv_target": "prior"},
    "prior-0.3": {"adv_lambda": 0.3, "adv_target": "prior"},
    "prior-1": {"adv_lambda": 1.0, "adv_target": "prior"},
    "prior-3": {"adv_lambda": 3.0, "adv_target": "prior"},
    "prior-10": {"adv_lambda": 10.0, "adv_target": "prior"},
    "max_ce-1": {"adv_lambda": 1.0, "adv_target": "max_ce"},
    "max_ce-10": {"adv_lambda": 10.0, "adv_target": "max_ce"},
}
RACE_REMOVED_MAX = 0.40   # probe balanced accuracy counted as "race removed" (fixed in the plan)


def run_setting(X, y, groups, folds, s, device, seed, log=print):
    """Out-of-fold predictions and probe predictions for one fixed setting (no tuning)."""
    K = int(folds.max()) + 1
    pred = np.full(len(y), np.nan)
    probe = np.full(len(y), None, dtype=object)
    per_fold = []
    for k in range(K):
        t0 = time.time()
        tr, te = folds != k, folds == k
        st = Standardiser().fit(X[tr])
        pf, _, info = fit_head(st(X[tr]), y[tr], s, EPOCHS, seed + 1000 * k, device, None,
                               groups[tr], y[tr])
        pred[te] = pf(st(X[te]))
        probe[te] = LK.fit_predict_probe(pf.encode(st(X[tr])), groups[tr],
                                         pf.encode(st(X[te])), seed)
        per_fold.append({"test_fold": k, "test_mae": float(np.abs(pred[te] - y[te]).mean()),
                         "probe_bal_acc": LK.balanced_accuracy(probe[te], groups[te],
                                                               sorted(set(groups[tr].tolist()))),
                         "adv_bal_acc_by_epoch": info.get("adv_bal_acc_by_epoch"),
                         "adv_ce_by_epoch": info.get("adv_ce_by_epoch"),
                         "seconds": round(time.time() - t0, 1)})
        log(f"  fold {k}: test MAE {per_fold[-1]['test_mae']:.2f}, "
            f"probe {per_fold[-1]['probe_bal_acc']:.3f}")
    assert not np.isnan(pred).any()
    return pred, probe, per_fold


def main():
    import pandas as pd
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default=",".join(SETTINGS))
    a = ap.parse_args()
    names = a.settings.split(",")
    assert set(names) <= set(SETTINGS), names
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    ids, X, ages, groups, folds, hashes = load_inputs()
    predA, shaA = load_A_oof(ids)
    dA = M.prepare(M.frame(ids, groups, ages, predA))
    plan = M.BootPlan(dA)
    out = {"plan": "docs/FOLLOWUP_E_FORCE_RACE.md", "inputs": hashes, "A_oof_sha256": shaA,
           "base": BASE, "fixed": FIXED_E, "epochs": EPOCHS, "settings": {},
           "race_removed_max_bal_acc": RACE_REMOVED_MAX, "device": device,
           "torch": torch.__version__,
           "note": "exploratory follow-up: unadjusted paired CIs, no Holm, no success verdict"}
    d_ctrl = None
    for name in names:
        s = {**FIXED_E, **BASE, **SETTINGS[name]}
        print(f"\n== {name}: target {s['adv_target']}, lambda {s['adv_lambda']}")
        pred, probe, per_fold = run_setting(X, ages, groups, folds, s, device, C.SEED)
        dst = Path(C.PER_IMAGE_DIR) / f"oof__E_force_{name}__seed{C.SEED}.csv"
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / dst.name
            pd.DataFrame({"id": ids, "fold": folds, "pred_age": pred}).to_csv(tmp, index=False)
            digest = sha256(tmp)
            atomic_copy(tmp, dst)
        d = M.prepare(M.frame(ids, groups, ages, pred))
        rec = {"setting": SETTINGS[name], "report": M.model_report(d, plan),
               "paired_vs_A": M.paired_vs_A(dA, d, plan),
               "leakage_probe": LK.summarise({"model": probe}, groups),
               "folds": per_fold,
               "oof_file": {"drive_path": str(dst.relative_to(C.DRIVE_ROOT)), "sha256": digest}}
        if name == "control":
            d_ctrl = d
        elif d_ctrl is not None:
            rec["paired_vs_control"] = M.paired_vs_A(d_ctrl, d, plan)
        rec["race_removed"] = rec["leakage_probe"]["bal_acc"]["model"]["est"] <= RACE_REMOVED_MAX
        out["settings"][name] = rec
        g, pr = rec["report"]["tpr_gap_W_minus_B"], rec["paired_vs_A"]
        print(f"  gap {g['est']:.3f} [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}]  reduction vs A "
              f"{pr['gap_reduction_A_minus_M']['est']:+.3f}  MAE {rec['report']['mae_overall']['est']:.2f}"
              f"  probe {rec['leakage_probe']['bal_acc']['model']['est']:.3f}"
              f"  age-err gap B-W {rec['report']['age_err_gap_B_minus_W']['est']:+.2f}")
    out["seconds"] = round(time.time() - t0, 1)
    res = write_result(REPO / "results" / "followups" / "E_force_race.json", out, C.as_dict(), REPO)
    print("->", res)


if __name__ == "__main__":
    main()
