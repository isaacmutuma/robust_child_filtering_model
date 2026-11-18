"""Age-regression heads on frozen CLIP embeddings and the out-of-fold (OOF) loop.

Methods (CLAUDE.md Decision Log, 2026-11-18):
  plain    (A) L1 on true age, ordinary shuffled batches, no group information.
  balanced (B) like A, but each epoch draws training rows with replacement so that, within
               every fold age band, the 5 races are drawn equally often; the bands keep
               their share of the training set.
  dro      (C) group DRO (Sagawa et al., 2020) over 30 groups = race x fold age band.
               Each group's L1 is compared with its band's mean across races (running
               averages), so a group gains weight only when it does worse than other races
               of the same age. Ordinary sampling. Group weights start equal (C) or at each
               cell's natural share of the training data (C2, dro_init='natural'; the loss
               then weights samples by q_g / share_g, so eta = 0 is plain training).

Protocol (docs/SUCCESS_CRITERIA.md §1, src/folds.py):
  for each outer test fold k:
    tune: train on folds != k, (k+1)%5; score every epoch of every grid setting on inner
          validation fold (k+1)%5 and pick one (setting, epoch) by the model's rule:
            'mae'         lowest inner MAE (model A)
            'worst_group' (v1) highest worst-group stratified minor TPR among candidates
                          whose adult FPR (20-59) <= 0.05 in every tested group; ties -> MAE
            'worst_group_v2' (post-hoc, 2026-11-18) as v1, but a candidate is allowed only
                          if every tested group's adult FPR <= min(0.05, A's FPR on the same
                          inner fold + 0.02) and its MAE <= A's inner MAE + 0.5; A's reference
                          is A's chosen setting/epochs for that fold retrained on the same
                          inner training folds
    refit: train on all four training folds with that setting and epoch count
    predict fold k (never seen in tuning or refit)
Inputs are L2-normalised, then standardised with the TRAINING folds' mean/std only.
The age target is also standardised with the training folds' mean/SD (an unscaled 0-116
target with L1 + Adam barely moves the output bias); predictions are mapped back to years.
"""

import itertools
import time

import numpy as np

from configs import config as C
from src.folds import age_band, inner_val_fold

# Model A grid (fixed before seeing any real result). hidden=0 means a linear head.
GRID_A = {"hidden": [0, 512], "lr": [1e-3, 3e-4], "weight_decay": [1e-4, 1e-2]}
FIXED_A = {"max_epochs": 60, "batch_size": 256, "dropout": 0.1, "loss": "l1",
           "method": "plain"}
# B and C reuse A's grid so differences come from the method, not from a wider search.
GRID_B, FIXED_B = GRID_A, {**FIXED_A, "method": "balanced"}
GRID_C = {**GRID_A, "dro_eta": [0.001, 0.01, 0.1, 1.0]}
FIXED_C = {**FIXED_A, "method": "dro", "dro_ema": 0.1}
GRID_C2, FIXED_C2 = GRID_C, {**FIXED_C, "dro_init": "natural"}


def grid_settings(grid):
    """List of dicts, one per combination, in a fixed order."""
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def _l2(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


class Standardiser:
    """L2-normalise, then z-score with statistics from the training rows only."""

    def fit(self, X):
        Z = _l2(X)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        return self

    def __call__(self, X):
        return ((_l2(X) - self.mu) / self.sd).astype(np.float32)


def _make_head(dim, hidden, dropout):
    import torch.nn as nn
    if hidden == 0:
        return nn.Linear(dim, 1)
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, 1))


# ── group bookkeeping for B and C ────────────────────────────────────────────
def cell_keys(groups, ages):
    """'race|band' key per row, using the fold age bands."""
    return np.array([f"{g}|{age_band(a)}" for g, a in zip(groups, ages)])


def balanced_probs(groups, ages):
    """Per-row draw probability for model B (sums to 1).

    band share kept: P(band b) = n_b / N; within band b each race present gets an equal
    share, spread evenly over that race's rows: p_i = (n_b / N) / R_b / n_{race,b}.
    """
    keys = cell_keys(groups, ages)
    bands = np.array([k.split("|")[1] for k in keys])
    N, p = len(keys), np.zeros(len(keys))
    for b in np.unique(bands):
        in_b = bands == b
        cells, counts = np.unique(keys[in_b], return_counts=True)
        for cell, n in zip(cells, counts):
            p[keys == cell] = (in_b.sum() / N) / len(cells) / n
    return p / p.sum()


def dro_groups(groups, ages):
    """(group index per row, band index per group, group names) for model C."""
    keys = cell_keys(groups, ages)
    names = sorted(set(keys.tolist()))
    gid = np.searchsorted(names, keys)
    band_names = sorted({n.split("|")[1] for n in names})
    band_of = np.array([band_names.index(n.split("|")[1]) for n in names])
    return gid, band_of, names


def fit_head(Xtr, ytr, s, epochs, seed, device, Xval=None, gtr=None, atr=None):
    """Train one head on the standardised target.

    Returns (predict_fn, per-epoch validation predictions in years [epochs x n_val] or
    None, info dict). predict_fn maps standardised inputs to ages in years.
    gtr / atr (groups, ages of the training rows) are needed by 'balanced' and 'dro'.
    """
    import torch
    method = s.get("method", "plain")
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = _make_head(Xtr.shape[1], s["hidden"], s["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    loss_fn = torch.nn.L1Loss() if s["loss"] == "l1" else torch.nn.MSELoss()
    y_mu, y_sd = float(np.mean(ytr)), float(np.std(ytr)) + 1e-6
    X = torch.from_numpy(Xtr).to(device)
    y = torch.from_numpy(((ytr - y_mu) / y_sd).astype(np.float32)).to(device)

    def predict_fn(Xn):
        return predict(model, Xn) * y_sd + y_mu

    info = {"method": method}
    if method == "balanced":
        probs = torch.from_numpy(balanced_probs(gtr, atr))
    if method == "dro":
        gid_np, band_of_np, names = dro_groups(gtr, atr)
        G = len(names)
        gid = torch.from_numpy(gid_np).to(device)
        band_of = torch.from_numpy(band_of_np).to(device)
        nb = int(band_of_np.max()) + 1
        ema = torch.full((G,), float("nan"), device=device)
        natural = s.get("dro_init", "uniform") == "natural"
        share = torch.bincount(gid, minlength=G).float() / len(gid)
        logq = torch.log(share) if natural else torch.zeros(G, device=device)
        eta, alpha = float(s["dro_eta"]), float(s["dro_ema"])

    Xv = torch.from_numpy(Xval).to(device) if Xval is not None else None
    val_preds = [] if Xval is not None else None
    for _ in range(epochs):
        model.train()
        if method == "balanced":
            order = torch.multinomial(probs, len(X), replacement=True, generator=g)
        else:
            order = torch.randperm(len(X), generator=g)
        for b in order.split(s["batch_size"]):
            b = b.to(device)
            opt.zero_grad()
            out = model(X[b]).squeeze(1)
            if method == "dro":
                per = (out - y[b]).abs() if s["loss"] == "l1" else (out - y[b]) ** 2
                gb = gid[b]
                sums = torch.zeros(G, device=device).index_add(0, gb, per)
                cnt = torch.bincount(gb, minlength=G).float()
                present = cnt > 0
                gmean = sums / cnt.clamp(min=1)
                with torch.no_grad():
                    upd = torch.where(torch.isnan(ema), gmean, (1 - alpha) * ema + alpha * gmean)
                    ema = torch.where(present, upd, ema)
                    seen = ~torch.isnan(ema)
                    band_sum = torch.zeros(nb, device=device).index_add(
                        0, band_of[seen], ema[seen])
                    band_n = torch.bincount(band_of[seen], minlength=nb).float()
                    band_mean = band_sum / band_n.clamp(min=1)
                    ratio = ema / band_mean[band_of].clamp(min=1e-8)
                    # groups at par (ratio 1) or not yet seen keep their weight
                    logq = logq + eta * torch.where(seen, ratio - 1, torch.zeros_like(ratio))
                    q = torch.softmax(logq, 0)
                    if natural:   # per-sample weight q_g / share_g (1 for all at eta = 0)
                        r = (q / share)[gb]
                    else:
                        w = torch.where(present, q, torch.zeros_like(q))
                        w = w / w.sum()
                if natural:
                    (r * per).mean().backward()
                else:
                    (w * gmean).sum().backward()
            else:
                loss_fn(out, y[b]).backward()
            opt.step()
        if Xv is not None:
            val_preds.append(predict_fn(Xv))
    if method == "dro":
        q = torch.softmax(logq, 0).cpu().numpy()
        top = np.argsort(-q)[:5]
        info["dro_final_weights_top5"] = {names[i]: float(q[i]) for i in top}
        info["dro_uniform_weight"] = 1.0 / G
        info["dro_init"] = "natural" if natural else "uniform"
        if natural:   # how far DRO moved each group from its natural share (top 5 ratios)
            ratio_q = (torch.softmax(logq, 0) / share).cpu().numpy()
            info["dro_weight_over_share_top5"] = {names[i]: float(ratio_q[i])
                                                  for i in np.argsort(-ratio_q)[:5]}
    return predict_fn, (np.stack(val_preds) if val_preds is not None else None), info


def predict(model, X):
    """Raw head output as a numpy array (X: numpy or tensor)."""
    import torch
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        Xt = X if torch.is_tensor(X) else torch.from_numpy(X)
        return model(Xt.to(dev)).squeeze(1).cpu().numpy().astype(float)


# ── selection on the inner validation fold ───────────────────────────────────
def selection_stats(pred, y, groups):
    """Inner-fold summary used by the selection rules (no bootstrap)."""
    pred, y, groups = np.asarray(pred), np.asarray(y), np.asarray(groups)
    flag = pred < C.CHILD_AGE_THRESHOLD
    lo, hi = C.ADULT_FPR_RANGE
    tprs, fprs = {}, {}
    for g in C.TESTED_GROUPS:
        m = groups == g
        t = 0.0
        for (a, b), w in zip(C.MINOR_STRATA, C.STRATUM_WEIGHTS.values()):
            s = m & (y >= a) & (y <= b)
            t += w * flag[s].mean() if s.any() else float("nan")
        tprs[g] = t
        ad = m & (y >= lo) & (y <= hi)
        fprs[g] = float(flag[ad].mean()) if ad.any() else float("nan")
    return {"mae": float(np.abs(pred - y).mean()),
            "worst_group_tpr": float(np.nanmin(list(tprs.values()))),
            "max_adult_fpr": float(np.nanmax(list(fprs.values()))),
            "feasible": bool(np.nanmax(list(fprs.values())) <= C.SUCCESS["adult_fpr_max"])}


def group_fpr_mae(pred, y, groups):
    """Per tested group adult FPR (20-59) and overall MAE on a set of predictions."""
    pred, y, groups = np.asarray(pred), np.asarray(y), np.asarray(groups)
    flag = pred < C.CHILD_AGE_THRESHOLD
    lo, hi = C.ADULT_FPR_RANGE
    fpr = {}
    for g in C.TESTED_GROUPS:
        ad = (groups == g) & (y >= lo) & (y <= hi)
        fpr[g] = float(flag[ad].mean()) if ad.any() else float("nan")
    return {"fpr": fpr, "mae": float(np.abs(pred - y).mean())}


def selection_stats_v2(pred, y, groups, ref):
    """v1 stats plus feasibility against A's inner-fold reference (see module docstring)."""
    st = selection_stats(pred, y, groups)
    cur = group_fpr_mae(pred, y, groups)
    S = C.SUCCESS
    excess = [max(0.0, cur["fpr"][g] - min(S["adult_fpr_max"], ref["fpr"][g] + S["adult_fpr_vs_A"]))
              for g in C.TESTED_GROUPS]
    mae_excess = max(0.0, cur["mae"] - (ref["mae"] + S["mae_noninferiority_years"]))
    n_viol = sum(e > 0 for e in excess) + (mae_excess > 0)
    return {**st, "fpr_by_group": cur["fpr"], "feasible": n_viol == 0, "n_violations": n_viol,
            "fpr_excess": float(sum(excess)), "mae_excess": float(mae_excess)}


def _rank_key(st, rule):
    """Sort key (smaller is better) for one (setting, epoch) candidate."""
    if rule == "mae":
        return (st["mae"],)
    if rule == "worst_group_v2":
        # allowed first; if nothing is allowed, fewest and smallest violations win
        return (not st["feasible"], st["n_violations"], st["fpr_excess"], st["mae_excess"],
                -st["worst_group_tpr"], st["mae"])
    if rule == "worst_group":
        # feasible first; then highest worst-group TPR; then lowest MAE. If nothing is
        # feasible, the candidate with the lowest max adult FPR wins.
        return (not st["feasible"], 0 if st["feasible"] else st["max_adult_fpr"],
                -st["worst_group_tpr"], st["mae"])
    raise ValueError(rule)


def _reference_A(rA, X, y, groups, tr_in, va, st, seed, k, device):
    """A's chosen setting for fold k, retrained on the same inner training folds.

    Uses A's tuning seed for that setting (seed + 1000k + its index in A's grid), so it
    reproduces A's own tuning trial; its inner MAE is checked against A's record.
    """
    settings = grid_settings(GRID_A)
    i = settings.index({kk: rA["chosen"][kk] for kk in GRID_A})
    s = {**FIXED_A, **settings[i]}
    _, vp, _ = fit_head(st(X[tr_in]), y[tr_in], s, rA["epochs"], seed + 1000 * k + i, device,
                        st(X[va]), groups[tr_in], y[tr_in])
    ref = group_fpr_mae(vp[-1], y[va], groups[va])
    ref["mae_diff_vs_A_record"] = ref["mae"] - rA["inner_val_mae"]
    if abs(ref["mae_diff_vs_A_record"]) > 0.05:
        raise RuntimeError(f"fold {k}: A reference MAE {ref['mae']:.3f} does not reproduce "
                           f"A's record {rA['inner_val_mae']:.3f}")
    return ref


def oof_predictions(X, y, folds, grid=None, fixed=None, device="cpu", seed=None, log=print,
                    groups=None, select="mae", ref_A=None):
    """Out-of-fold predictions for every row, plus a per-fold tuning record.

    X: (n, d) raw embeddings; y: true ages; folds: int array of fold ids (0..K-1);
    groups: race per row (needed by 'balanced', 'dro' and the 'worst_group' rules);
    ref_A: model A's per-fold record (chosen setting, epochs) for 'worst_group_v2'.
    """
    grid, fixed = grid or GRID_A, fixed or FIXED_A
    seed = C.SEED if seed is None else seed
    folds = np.asarray(folds)
    groups = None if groups is None else np.asarray(groups)
    K = int(folds.max()) + 1
    pred, record = np.full(len(y), np.nan), []
    for k in range(K):
        t0, v = time.time(), inner_val_fold(k, K)
        tr_in, va, tr_all, te = (folds != k) & (folds != v), folds == v, folds != k, folds == k
        gsub = (lambda m: groups[m]) if groups is not None else (lambda m: None)
        st = Standardiser().fit(X[tr_in])
        ref = None
        if select == "worst_group_v2":
            ref = _reference_A(ref_A[k], X, y, groups, tr_in, va, st, seed, k, device)
        trials = []
        for i, s in enumerate(grid_settings(grid)):
            s = {**fixed, **s}
            _, vp, _ = fit_head(st(X[tr_in]), y[tr_in], s, s["max_epochs"],
                                seed + 1000 * k + i, device, st(X[va]), gsub(tr_in), y[tr_in])
            if select == "mae":
                stats = [{"mae": float(np.abs(p - y[va]).mean())} for p in vp]
            elif select == "worst_group_v2":
                stats = [selection_stats_v2(p, y[va], groups[va], ref) for p in vp]
            else:
                stats = [selection_stats(p, y[va], groups[va]) for p in vp]
            best = min(range(len(stats)), key=lambda e: _rank_key(stats[e], select))
            trials.append({"setting": {k2: s[k2] for k2 in grid}, "best_epoch": best + 1,
                           "val_mae": stats[best]["mae"],
                           **({} if select == "mae" else
                              {"val_worst_group_tpr": stats[best]["worst_group_tpr"],
                               "val_max_adult_fpr": stats[best]["max_adult_fpr"],
                               "val_feasible": stats[best]["feasible"]}),
                           "_key": _rank_key(stats[best], select)})
        win = min(trials, key=lambda t: t["_key"])
        for t in trials:
            t.pop("_key")
        s = {**fixed, **win["setting"]}
        st = Standardiser().fit(X[tr_all])
        predict_fn, _, info = fit_head(st(X[tr_all]), y[tr_all], s, win["best_epoch"],
                                       seed + 1000 * k, device, None, gsub(tr_all), y[tr_all])
        pred[te] = predict_fn(st(X[te]))
        test_mae = float(np.abs(pred[te] - y[te]).mean())
        rec = {"test_fold": k, "inner_val_fold": v, "select": select, "chosen": win["setting"],
               "epochs": win["best_epoch"], "inner_val_mae": win["val_mae"],
               "test_mae": test_mae, "trials": trials, "refit_info": info,
               "seconds": round(time.time() - t0, 1)}
        if select != "mae":
            rec["inner_val_worst_group_tpr"] = win["val_worst_group_tpr"]
            rec["inner_val_feasible"] = win["val_feasible"]
        if ref is not None:
            rec["reference_A"] = ref
            rec["n_trials_feasible_at_best"] = int(sum(t["val_feasible"] for t in trials))
        record.append(rec)
        extra = "" if select == "mae" else \
            f", inner worst-group TPR {win['val_worst_group_tpr']:.3f}"
        log(f"fold {k}: chose {win['setting']} x {win['best_epoch']} ep, "
            f"inner-val MAE {win['val_mae']:.2f}{extra}, test MAE {test_mae:.2f}")
    assert not np.isnan(pred).any()
    return pred, record
