"""Follow-up 2: does removing race from the CLIP embeddings (LEACE) close the gap?
(docs/FOLLOWUP_LEACE.md)

  CRF_ENV=colab python scripts/followup_leace.py [--variants plain,band]
For each variant, LEACE is fitted on the training rows of every fold (inner and refit),
then model A's exact procedure runs on the erased embeddings (grid, 'mae' selection, refit,
out-of-fold predictions). Leakage after erasure: linear and MLP probes on the erased inputs
(and on the plain standardised inputs as reference), linear probe on the head's hidden layer.
Exploratory: unadjusted paired CIs vs A, no Holm. Per-image OOF -> Drive; aggregate JSON ->
results/followups/LEACE.json.
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
from src.leace import LeaceStandardiser, prep_factory
from src.provenance import write_result
from src.train import FIXED_A, GRID_A, Standardiser, fit_head, oof_predictions
from train_model import load_A_oof, load_inputs

VARIANTS = {"band": True, "plain": False}   # name -> conditional on fold age band; band = primary
RACE_REMOVED_MAX = 0.40                      # same threshold as follow-up 1


def mlp_probe(Ztr, gtr, Zte, seed=0):
    """Non-linear race probe: one hidden layer (256), early stopping."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ztr)
    clf = MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200,
                        random_state=seed)
    # integer class codes: newer scikit-learn's early-stopping score calls np.isnan on the
    # predictions, which fails for string labels (seen on Colab 2026-11-23)
    classes, codes = np.unique(np.asarray(gtr), return_inverse=True)
    clf.fit(sc.transform(Ztr), codes)
    return classes[clf.predict(sc.transform(Zte))]


def probes(X, y, groups, folds, conditional, record, device, seed, log=print):
    """Out-of-fold race predictions for each representation (pooled over test folds)."""
    names = ["erased_linear", "erased_mlp", "plain_linear", "plain_mlp", "head_z_linear"]
    out = {n: np.full(len(y), None, dtype=object) for n in names}
    checks = []
    for k in range(int(folds.max()) + 1):
        t0 = time.time()
        tr, te = folds != k, folds == k
        le = LeaceStandardiser(conditional).fit(X[tr], groups[tr], y[tr])
        st = Standardiser().fit(X[tr])
        Etr, Ete, Ptr, Pte = le(X[tr]), le(X[te]), st(X[tr]), st(X[te])
        out["erased_linear"][te] = LK.fit_predict_probe(Etr, groups[tr], Ete, seed)
        out["erased_mlp"][te] = mlp_probe(Etr, groups[tr], Ete, seed)
        out["plain_linear"][te] = LK.fit_predict_probe(Ptr, groups[tr], Pte, seed)
        out["plain_mlp"][te] = mlp_probe(Ptr, groups[tr], Pte, seed)
        rec = record[k]
        pf, _, _ = fit_head(Etr, y[tr], {**FIXED_A, **rec["chosen"]}, rec["epochs"],
                            seed + 1000 * k, device)
        out["head_z_linear"][te] = LK.fit_predict_probe(pf.encode(Etr), groups[tr],
                                                        pf.encode(Ete), seed)
        Z = np.eye(len(le.leace.classes))[np.searchsorted(le.leace.classes, groups[tr])]
        xcov = np.abs((Etr - Etr.mean(0)).T @ (Z - Z.mean(0)) / tr.sum()).max()
        checks.append({"test_fold": k, "leace_rank": le.leace.rank,
                       "max_abs_train_cross_cov_race_onehot_after": float(xcov),  # ~0 for plain; band keeps age-mix part
                       "head_refit_test_mae_diff_vs_oof": float(
                           np.abs(pf(Ete) - y[te]).mean() - rec["test_mae"]),
                       "seconds": round(time.time() - t0, 1)})
        log(f"  probes fold {k}: rank {le.leace.rank}, max |cross-cov| {xcov:.2e}")
    return out, checks


def main():
    import pandas as pd
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    names = a.variants.split(",")
    assert set(names) <= set(VARIANTS), names
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    ids, X, ages, groups, folds, hashes = load_inputs()
    predA, shaA = load_A_oof(ids)
    dA = M.prepare(M.frame(ids, groups, ages, predA))
    plan = M.BootPlan(dA)
    out = {"plan": "docs/FOLLOWUP_LEACE.md", "inputs": hashes, "A_oof_sha256": shaA,
           "grid": GRID_A, "fixed": FIXED_A, "select": "mae", "variants": {},
           "race_removed_max_bal_acc": RACE_REMOVED_MAX, "device": device,
           "torch": torch.__version__,
           "note": "exploratory follow-up: unadjusted paired CIs, no Holm, no success verdict"}
    for name in names:
        cond = VARIANTS[name]
        print(f"\n== LEACE {name} (conditional on age band: {cond})")
        pred, record = oof_predictions(X, ages, folds, GRID_A, FIXED_A, device, groups=groups,
                                       select="mae", prep=prep_factory(cond))
        dst = Path(C.PER_IMAGE_DIR) / f"oof__LEACE_{name}__seed{C.SEED}.csv"
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / dst.name
            pd.DataFrame({"id": ids, "fold": folds, "pred_age": pred}).to_csv(tmp, index=False)
            digest = sha256(tmp)
            atomic_copy(tmp, dst)
        pr_out, checks = probes(X, ages, groups, folds, cond, record, device, C.SEED)
        d = M.prepare(M.frame(ids, groups, ages, pred))
        lk = LK.summarise(pr_out, groups)
        rec = {"conditional_on_age_band": cond, "report": M.model_report(d, plan),
               "paired_vs_A": M.paired_vs_A(dA, d, plan), "probes": lk, "checks": checks,
               "folds": record,
               "oof_file": {"drive_path": str(dst.relative_to(C.DRIVE_ROOT)), "sha256": digest}}
        rec["race_removed_linear"] = lk["bal_acc"]["erased_linear"]["est"] <= RACE_REMOVED_MAX
        out["variants"][name] = rec
        g, pr = rec["report"]["tpr_gap_W_minus_B"], rec["paired_vs_A"]
        print(f"  gap {g['est']:.3f} [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}]  reduction vs A "
              f"{pr['gap_reduction_A_minus_M']['est']:+.3f}  MAE {rec['report']['mae_overall']['est']:.2f}"
              f"  age-err gap B-W {rec['report']['age_err_gap_B_minus_W']['est']:+.2f}")
        print("  probes:", {n: round(v["est"], 3) for n, v in lk["bal_acc"].items()})
    out["seconds"] = round(time.time() - t0, 1)
    res = write_result(REPO / "results" / "followups" / "LEACE.json", out, C.as_dict(), REPO)
    print("->", res)


if __name__ == "__main__":
    main()
