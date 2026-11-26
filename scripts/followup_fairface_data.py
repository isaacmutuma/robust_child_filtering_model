"""Follow-up 4 (screening): does adding FairFace train to the age head's training narrow the
White - Black gap on UTKFace? Plan and reading rules: docs/FOLLOWUP_FAIRFACE_DATA.md.

  python scripts/followup_fairface_data.py --emb-dir <dir with utkface__/fairface_train__ npz>

Conditions (all with A's chosen setting per fold from results/models/A.json; epochs <= 60
chosen on the inner UTKFace validation fold by MAE, then refit):
  A_local    UTKFace training folds (L1)            -> reference for every paired test
  A+FF       + all FairFace train (interval L1 on its age bins)
  A_half / A_quarter   50% / 25% of UTKFace training rows (learning curve)
  A_FF       FairFace only, epoch on a 10% FairFace hold-out; predicts all UTKFace
Output: results/followups/fairface_data.json (aggregate only). Per-image predictions are not
written to the repo.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from _common import REPO, sha256
from configs import config as C
from src import metrics as M
from src.folds import age_band, inner_val_fold, make_folds
from src.provenance import write_result
from src.train import FIXED_A, Standardiser, _make_head, fit_head, predict

FF_BIN_EDGES = {"0-2": (0, 3), "3-9": (3, 10), "10-19": (10, 20), "20-29": (20, 30),
                "30-39": (30, 40), "40-49": (40, 50), "50-59": (50, 60), "60-69": (60, 70),
                "more than 70": (70, 100)}


def fit_interval(Xtr, lo, hi, s, epochs, seed, device, y_mu, y_sd, Xval=None):
    """A's plain training loop with a per-row interval L1 loss.

    Row i costs relu(lo_i - out) + relu(out - hi_i) (standardised units); with lo = hi = age
    this is exactly |out - y| (A's L1), so UTKFace rows are trained as in A. Same RNG use
    as fit_head's plain path (head init, randperm with a seeded generator, dropout).
    """
    import torch
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = _make_head(Xtr.shape[1], s["hidden"], s["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    X = torch.from_numpy(Xtr).to(device)
    L = torch.from_numpy(((lo - y_mu) / y_sd).astype(np.float32)).to(device)
    H = torch.from_numpy(((hi - y_mu) / y_sd).astype(np.float32)).to(device)
    Xv = torch.from_numpy(Xval).to(device) if Xval is not None else None
    vp = []
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(X), generator=g).split(s["batch_size"]):
            b = b.to(device)
            opt.zero_grad()
            out = model(X[b]).squeeze(1)
            (torch.relu(L[b] - out) + torch.relu(out - H[b])).mean().backward()
            opt.step()
        if Xv is not None:
            vp.append(predict(model, Xv) * y_sd + y_mu)
    return (lambda Xn: predict(model, Xn) * y_sd + y_mu), (np.stack(vp) if vp else None)


def subsample(mask, groups, ages, frac, seed):
    """Keep `frac` of the rows in `mask`, within each group x fold age band."""
    if frac >= 1:
        return mask.copy()
    rng = np.random.default_rng(seed)
    keep = np.zeros_like(mask)
    keys = np.array([f"{g}|{age_band(a)}" for g, a in zip(groups, ages)])
    for k in np.unique(keys[mask]):
        idx = np.flatnonzero(mask & (keys == k))
        n = max(1, int(round(frac * len(idx))))
        keep[rng.choice(idx, n, replace=False)] = True
    return keep


def run_utk(X, y, groups, folds, A_rec, device, ff=None, frac=1.0, log=print):
    """OOF predictions for A_local / A+FF / subsampled A; ff = (Xff, lo, hi) or None."""
    K = int(folds.max()) + 1
    pred, rec = np.full(len(y), np.nan), []
    for k in range(K):
        t0, v = time.time(), inner_val_fold(k, K)
        pool = subsample(folds != k, groups, y, frac, C.SEED + 100 * k)
        tr_in, va, te = pool & (folds != v), folds == v, folds == k
        s = {**FIXED_A, **A_rec[k]["chosen"]}

        def fit(tr, epochs, Xval):
            Xtr, lo = X[tr], y[tr].astype(float)
            hi = lo.copy()
            if ff is not None:
                Xtr = np.concatenate([Xtr, ff[0]])
                lo, hi = np.concatenate([lo, ff[1]]), np.concatenate([hi, ff[2]])
            st = Standardiser().fit(Xtr)
            y_mu, y_sd = float(np.mean(y[tr])), float(np.std(y[tr])) + 1e-6
            fn, vp = fit_interval(st(Xtr), lo, hi, s, epochs, C.SEED + 1000 * k, device,
                                  y_mu, y_sd, None if Xval is None else st(Xval))
            return (lambda Xn: fn(st(Xn))), vp

        _, vp = fit(tr_in, s["max_epochs"], X[va])
        maes = np.abs(vp - y[va]).mean(1)
        ep = int(np.argmin(maes)) + 1
        fn, _ = fit(pool, ep, None)
        pred[te] = fn(X[te])
        rec.append({"test_fold": k, "setting": A_rec[k]["chosen"], "epochs": ep,
                    "inner_val_mae": float(maes[ep - 1]),
                    "test_mae": float(np.abs(pred[te] - y[te]).mean()),
                    "n_train_utk": int(pool.sum()),
                    "n_train_ff": 0 if ff is None else int(len(ff[0])),
                    "seconds": round(time.time() - t0, 1)})
        log(f"  fold {k}: {ep} ep, inner MAE {maes[ep - 1]:.2f}, test MAE {rec[-1]['test_mae']:.2f}")
    return pred, rec


def run_ff_only(Xff, lo, hi, X_utk, setting, device, log=print):
    """A_FF: FairFace only; epoch by interval loss on a 10% FairFace hold-out."""
    rng = np.random.default_rng(C.SEED)
    ho = np.zeros(len(Xff), bool)
    ho[rng.choice(len(Xff), len(Xff) // 10, replace=False)] = True
    mid = (lo + np.minimum(hi, 80)) / 2   # target scale from bin midpoints (no UTKFace labels)
    y_mu, y_sd = float(mid.mean()), float(mid.std()) + 1e-6
    s = {**FIXED_A, **setting}
    st = Standardiser().fit(Xff[~ho])
    _, vp = fit_interval(st(Xff[~ho]), lo[~ho], hi[~ho], s, s["max_epochs"], C.SEED, device,
                         y_mu, y_sd, st(Xff[ho]))
    vloss = (np.maximum(lo[ho] - vp, 0) + np.maximum(vp - hi[ho], 0)).mean(1)
    ep = int(np.argmin(vloss)) + 1
    st = Standardiser().fit(Xff)
    fn, _ = fit_interval(st(Xff), lo, hi, s, ep, C.SEED, device, y_mu, y_sd)
    log(f"  A_FF: {ep} ep, hold-out interval loss {vloss[ep - 1]:.2f}")
    return fn(st(X_utk)), {"setting": setting, "epochs": ep,
                           "holdout_interval_l1": float(vloss[ep - 1])}


def check_equivalence(X, y, A_rec, device):
    """fit_interval with point targets must equal A's fit_head (plain path) exactly."""
    rng = np.random.default_rng(1)
    idx = rng.choice(len(X), 2000, replace=False)
    Xs = Standardiser().fit(X[idx])(X[idx])
    s = {**FIXED_A, **A_rec[0]["chosen"]}
    fA, _, _ = fit_head(Xs, y[idx], s, 3, 5, device)
    mu, sd = float(np.mean(y[idx])), float(np.std(y[idx])) + 1e-6
    fI, _ = fit_interval(Xs, y[idx].astype(float), y[idx].astype(float), s, 3, 5, device, mu, sd)
    d = float(np.max(np.abs(fA(Xs) - fI(Xs))))
    assert d < 1e-4, f"interval loss path differs from A: {d}"
    return d


def main():
    import pandas as pd
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    t_all = time.time()
    ed = Path(a.emb_dir)
    for ds in ("utkface", "fairface_train"):   # same files as notebook 01 (T4)
        man = json.loads((REPO / "results/embeddings" / f"{ds}__{C.EMB_TAG}.json").read_text())
        assert sha256(ed / f"{ds}__{C.EMB_TAG}.npz") == man["npz"]["sha256"], f"{ds} npz sha"
    zu = np.load(ed / f"utkface__{C.EMB_TAG}.npz")
    order = np.argsort(zu["ids"])
    ids, X = zu["ids"][order], zu["emb"][order].astype(np.float32)
    y, groups = zu["age"][order].astype(float), zu["group"][order]

    # folds: rebuilt from ids/groups/ages and checked against the committed fold file hash
    fmap = make_folds(ids.tolist(), groups.tolist(), y.tolist())
    df = pd.DataFrame({"id": sorted(fmap), "fold": [fmap[i] for i in sorted(fmap)]})
    tmp = REPO / "data" / "local" / "utkface_folds_seed0.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    want = json.loads((REPO / "results/folds/utkface_folds_seed0.json").read_text())
    fold_sha = sha256(tmp)
    assert fold_sha == want["fold_file"]["sha256"], "fold file differs from the committed one"
    folds = np.array([fmap[i] for i in ids])

    A_json = json.loads((REPO / "results/models/A.json").read_text())
    A_rec = sorted(A_json["folds"], key=lambda f: f["test_fold"])
    eq = check_equivalence(X, y, A_rec, a.device)
    print(f"interval-loss path = A's fit_head (max diff {eq:.2e})")

    zf = np.load(ed / f"fairface_train__{C.EMB_TAG}.npz")
    Xff = zf["emb"].astype(np.float32)
    bins = zf["age_bin"]
    lo = np.array([FF_BIN_EDGES[b][0] for b in bins], float)
    hi = np.array([FF_BIN_EDGES[b][1] for b in bins], float)
    ff_counts = {g: dict(Counter(bins[zf["group"] == g])) for g in sorted(set(zf["group"]))}

    plan = M.BootPlan(M.prepare(M.frame(ids, groups, y, np.zeros(len(y)))))
    out, d_ref = {}, None
    conds = [("A_local", None, 1.0), ("A+FF", (Xff, lo, hi), 1.0),
             ("A_half", None, 0.5), ("A_quarter", None, 0.25)]
    for name, ff, frac in conds:
        print(name, flush=True)
        t0 = time.time()
        p, rec = run_utk(X, y, groups, folds, A_rec, a.device, ff, frac)
        d = M.prepare(M.frame(ids, groups, y, p))
        out[name] = {"folds": rec, "report": M.model_report(d, plan),
                     "seconds": round(time.time() - t0, 1)}
        if name == "A_local":
            d_ref = d
        else:
            out[name]["paired_vs_A_local"] = M.paired_vs_A(d_ref, d, plan)
    print("A_FF", flush=True)
    common = Counter(json.dumps(f["chosen"], sort_keys=True) for f in A_rec).most_common(1)[0][0]
    p, rec = run_ff_only(Xff, lo, hi, X, json.loads(common), a.device)
    d = M.prepare(M.frame(ids, groups, y, p))
    out["A_FF"] = {"fit": rec, "report": M.model_report(d, plan),
                   "paired_vs_A_local": M.paired_vs_A(d_ref, d, plan)}

    payload = {
        "followup": "4: FairFace training data (screening)", "plan": "docs/FOLLOWUP_FAIRFACE_DATA.md",
        "inputs": {"utkface_npz_sha256": sha256(ed / f"utkface__{C.EMB_TAG}.npz"),
                   "fairface_npz_sha256": sha256(ed / f"fairface_train__{C.EMB_TAG}.npz"),
                   "folds_sha256": fold_sha,
                   "embeddings_note": "T4 embeddings from Drive (notebook 01); "
                                      "compared with A_local rebuilt in the same run"},
        "A_committed_gap": A_json["report"]["tpr_gap_W_minus_B"],
        "equivalence_max_diff": eq, "fairface_bin_counts_by_group": ff_counts,
        "n_fairface": int(len(Xff)), "device": a.device, "torch": torch.__version__,
        "conditions": out, "seconds": round(time.time() - t_all, 1)}
    res = write_result(REPO / "results" / "followups" / "fairface_data.json", payload,
                       C.as_dict(), REPO)
    print("->", res)
    for name, o in out.items():
        r = o["report"]
        pr = o.get("paired_vs_A_local", {}).get("gap_reduction_A_minus_M", {})
        print(f"{name:10s} gap {r['tpr_gap_W_minus_B']['est']:.3f} "
              f"teen B/W {r['by_group']['Black']['tpr_teen']['est']:.3f}/"
              f"{r['by_group']['White']['tpr_teen']['est']:.3f} "
              f"B-W err {r['age_err_gap_B_minus_W']['est']:+.2f} MAE {r['mae_overall']['est']:.2f} "
              f"red {pr.get('est', float('nan')):+.3f} p {pr.get('p_one_sided', float('nan')):.3f}")


if __name__ == "__main__":
    main()
