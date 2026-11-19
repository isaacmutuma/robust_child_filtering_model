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
  adv_inv  (E-invariant) adversarial training, GAN-style alternating updates (CLAUDE.md
               Decision Log, 2026-11-19). The MLP head is split at its hidden layer:
               encoder = Linear + GELU (z = A's hidden layer, 512 wide), age head = Dropout +
               Linear. Per batch: (1) adv_steps updates of an adversary MLP that predicts race
               from z with the encoder fixed, loss CE(race | z); (2) one update of encoder +
               head with L1(age) + adv_lambda * fairness term, where the term is either
               -CE(race | z) ('max_ce': maximise the adversary's error) or
               KL(training race shares || adversary's guess) ('prior': push the guess
               towards the base rates; Alvi et al. 2018). adv_lambda = 0 is model A exactly
               (the adversary is initialised from its own RNG stream).

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
            'worst_group_v3' (post-hoc, 2026-11-18; FINAL, frozen) as v2, plus every tested
                          group's stratified minor TPR >= A's on the same inner fold - 0.02
                          (guard against closing the gap by lowering other groups)
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
GRID_C2, FIXED_C2 = GRID_C, {**FIXED_C, "dro_init": "natural"}          # v2 run (uncapped)
# C2 rerun under v3 (2026-11-18): weights capped to [0.1, 10] x natural share; eta 1 dropped
GRID_C2_CAP = {**GRID_A, "dro_eta": [0.001, 0.01, 0.1]}
FIXED_C2_CAP = {**FIXED_C2, "dro_cap": 10.0}
# E-invariant (revised grid, 2026-11-19): A's MLP settings x adv_lambda x adversarial target.
# Only hidden 512 (a linear head has no hidden layer to hide race from; A chose 512 in all
# 5 folds). Fast adversary (lr 1e-2, 5 steps per encoder step): with a slow adversary
# (lr 1e-3) synthetic tests showed it fooled while race stayed fully readable from z.
# The leakage probe runs after every refit (src/leakage.py).
GRID_E = {"hidden": [512], "lr": [1e-3, 3e-4], "weight_decay": [1e-4, 1e-2],
          "adv_lambda": [0.1, 0.3, 1.0, 3.0], "adv_target": ["max_ce", "prior"]}
FIXED_E = {**FIXED_A, "method": "adv_inv", "adv_hidden": 128, "adv_lr": 1e-2, "adv_steps": 5,
           "adv_weight_decay": 1e-4, "leakage_probe": True}


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


def _make_adversary(dim, hidden, n_classes):
    """Race classifier on z (model E). No dropout, so it never draws from the RNG."""
    import torch.nn as nn
    if hidden == 0:
        return nn.Linear(dim, n_classes)
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))


def encode(model, X):
    """z = output of the MLP head's hidden layer (Linear + GELU), as numpy float32."""
    import torch
    model.eval()
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 4096):
            Xt = torch.from_numpy(np.ascontiguousarray(X[b:b + 4096])).to(dev)
            out.append(model[:2](Xt).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


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

    if s["hidden"] > 0:   # hidden layer z, for the leakage probe
        predict_fn.encode = lambda Xn: encode(model, Xn)
    info = {"method": method}
    if method == "adv_inv":
        assert s["hidden"] > 0, "E needs a hidden layer"
        classes = sorted(set(np.asarray(gtr).tolist()))
        rid = torch.from_numpy(np.searchsorted(classes, np.asarray(gtr))).to(device)
        with torch.random.fork_rng(devices=[]):   # keep A's RNG stream untouched
            torch.manual_seed(seed + 7919)
            adv = _make_adversary(s["hidden"], s["adv_hidden"], len(classes))
        adv = adv.to(device)
        opt_adv = torch.optim.AdamW(adv.parameters(), lr=s["adv_lr"],
                                    weight_decay=s["adv_weight_decay"])
        ce_fn = torch.nn.CrossEntropyLoss()
        lam, n_adv = float(s["adv_lambda"]), int(s["adv_steps"])
        target = s.get("adv_target", "max_ce")
        log_prior = torch.log(torch.bincount(rid, minlength=len(classes)).float() / len(rid))
        enc, head = model[:2], model[2:]
        nC = len(classes)
        hist = {"adv_bal_acc": [], "adv_ce": []}
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
        cap = s.get("dro_cap")
        if cap is not None:   # capped C2: z = log(weight / share), kept in [-ln cap, ln cap]
            assert natural, "dro_cap is defined for the natural start only"
            zcap = float(np.log(cap))
            z = torch.zeros(G, device=device)
        eta, alpha = float(s["dro_eta"]), float(s["dro_ema"])

    Xv = torch.from_numpy(Xval).to(device) if Xval is not None else None
    val_preds = [] if Xval is not None else None
    for _ in range(epochs):
        model.train()
        if method == "balanced":
            order = torch.multinomial(probs, len(X), replacement=True, generator=g)
        else:
            order = torch.randperm(len(X), generator=g)
        if method == "adv_inv":   # adversary's guesses on z during this epoch (after its step)
            conf = torch.zeros(nC, nC, dtype=torch.long, device=device)
            ce_sum = 0.0
        for b in order.split(s["batch_size"]):
            b = b.to(device)
            if method == "adv_inv":
                zb = enc(X[b])
                zd = zb.detach()
                for _ in range(n_adv):            # step 1: adversary, encoder fixed
                    opt_adv.zero_grad()
                    ce_fn(adv(zd), rid[b]).backward()
                    opt_adv.step()
                opt.zero_grad()                   # step 2: encoder + age head
                out = head(zb).squeeze(1)
                logits = adv(zb)
                ce = ce_fn(logits, rid[b])
                loss = loss_fn(out, y[b])
                if lam != 0 and target == "max_ce":     # maximise the adversary's error
                    loss = loss - lam * ce
                elif lam != 0:   # 'prior': adversary's guess -> training base rates (Alvi 2018)
                    logp = torch.log_softmax(logits, 1)
                    kl = (log_prior.exp() * (log_prior - logp)).sum(1).mean()
                    loss = loss + lam * kl
                loss.backward()                   # adv grads are cleared before its next step
                opt.step()
                with torch.no_grad():
                    conf += torch.bincount(rid[b] * nC + logits.argmax(1),
                                           minlength=nC * nC).view(nC, nC)
                    ce_sum += float(ce) * len(b)
                continue
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
                    step = eta * torch.where(seen, ratio - 1, torch.zeros_like(ratio))
                    if cap is not None:
                        z = (z + step).clamp(-zcap, zcap)
                        r = torch.exp(z)[gb]   # weight / natural share, in [1/cap, cap]
                    else:
                        logq = logq + step
                        q = torch.softmax(logq, 0)
                    if cap is not None:
                        pass
                    elif natural:   # per-sample weight q_g / share_g (1 for all at eta = 0)
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
        if method == "adv_inv":
            c = conf.cpu().numpy()
            hist["adv_bal_acc"].append(float(np.mean(np.diag(c) / np.maximum(c.sum(1), 1))))
            hist["adv_ce"].append(ce_sum / len(X))
        if Xv is not None:
            val_preds.append(predict_fn(Xv))
    if method == "adv_inv":
        info.update({"adv_classes": classes, "adv_chance_bal_acc": 1.0 / nC,
                     "adv_bal_acc_by_epoch": hist["adv_bal_acc"],
                     "adv_ce_by_epoch": hist["adv_ce"]})
    if method == "dro":
        if cap is not None:   # capped: weight proportional to share x e^z
            qt = share * torch.exp(z)
            q = (qt / qt.sum()).cpu().numpy()
        else:
            q = torch.softmax(logq, 0).cpu().numpy()
        top = np.argsort(-q)[:5]
        info["dro_final_weights_top5"] = {names[i]: float(q[i]) for i in top}
        info["dro_uniform_weight"] = 1.0 / G
        info["dro_init"] = "natural" if natural else "uniform"
        if cap is not None:
            info["dro_cap"] = cap
        if natural:   # how far DRO moved each group from its natural share (top 5 ratios)
            ratio_q = (torch.exp(z) if cap is not None
                       else torch.softmax(logq, 0) / share).cpu().numpy()
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
def selection_stats(pred, y, groups, per_group=False):
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
    out = {"mae": float(np.abs(pred - y).mean()),
           "worst_group_tpr": float(np.nanmin(list(tprs.values()))),
           "max_adult_fpr": float(np.nanmax(list(fprs.values()))),
           "feasible": bool(np.nanmax(list(fprs.values())) <= C.SUCCESS["adult_fpr_max"])}
    if per_group:
        out["tpr_by_group"] = {g: float(t) for g, t in tprs.items()}
    return out


def group_tpr(pred, y, groups):
    """Per tested group stratified minor TPR (fixed reference weights)."""
    return selection_stats(pred, y, groups, per_group=True)["tpr_by_group"]


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


def selection_stats_v3(pred, y, groups, ref):
    """v2 stats plus the levelling-down guard (group TPR >= A's inner TPR - 0.02)."""
    st = selection_stats_v2(pred, y, groups, ref)
    tpr = group_tpr(pred, y, groups)
    drop = C.TUNING["tpr_drop_max"]
    excess = [max(0.0, (ref["tpr"][g] - drop) - tpr[g]) for g in C.TESTED_GROUPS]
    n_tpr = sum(e > 0 for e in excess)
    return {**st, "tpr_by_group": tpr, "tpr_excess": float(sum(excess)),
            "n_violations": st["n_violations"] + n_tpr, "feasible": st["feasible"] and n_tpr == 0}


def _rank_key(st, rule):
    """Sort key (smaller is better) for one (setting, epoch) candidate."""
    if rule == "worst_group_v3":
        return (not st["feasible"], st["n_violations"], st["fpr_excess"], st["tpr_excess"],
                st["mae_excess"], -st["worst_group_tpr"], st["mae"])
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
    ref["tpr"] = group_tpr(vp[-1], y[va], groups[va])
    ref["mae_diff_vs_A_record"] = ref["mae"] - rA["inner_val_mae"]
    if abs(ref["mae_diff_vs_A_record"]) > 0.05:
        raise RuntimeError(f"fold {k}: A reference MAE {ref['mae']:.3f} does not reproduce "
                           f"A's record {rA['inner_val_mae']:.3f}")
    return ref


def _leakage_fold(X, y, groups, tr_all, te, st, predict_fn, recA, seed, k, device, probe_out):
    """Fit the race probe on the training folds, predict test fold k, for 3 representations."""
    from src.leakage import fit_predict_probe
    Xtr, Xte = st(X[tr_all]), st(X[te])
    sA = {**FIXED_A, **recA["chosen"]}
    pfA, _, _ = fit_head(Xtr, y[tr_all], sA, recA["epochs"], seed + 1000 * k, device)
    maeA = float(np.abs(pfA(Xte) - y[te]).mean())
    reps = {"model": (predict_fn.encode(Xtr), predict_fn.encode(Xte)),
            "A": (pfA.encode(Xtr), pfA.encode(Xte)),
            "clip_raw": (Xtr, Xte)}
    out = {"A_refit_test_mae": maeA}
    if recA.get("test_mae") is not None:
        out["A_refit_test_mae_diff_vs_A_record"] = maeA - recA["test_mae"]
    for name, (Ztr, Zte) in reps.items():
        p = fit_predict_probe(Ztr, groups[tr_all], Zte, seed)
        if probe_out is not None:
            probe_out.setdefault(name, np.full(len(y), None, dtype=object))[te] = p
        out[f"bal_acc_{name}"] = float(np.mean([np.mean(p[groups[te] == c] == c)
                                                for c in np.unique(groups[tr_all])]))
    return out


def oof_predictions(X, y, folds, grid=None, fixed=None, device="cpu", seed=None, log=print,
                    groups=None, select="mae", ref_A=None, probe_out=None):
    """Out-of-fold predictions for every row, plus a per-fold tuning record.

    X: (n, d) raw embeddings; y: true ages; folds: int array of fold ids (0..K-1);
    groups: race per row (needed by 'balanced', 'dro' and the 'worst_group' rules);
    ref_A: model A's per-fold record (chosen setting, epochs) for 'worst_group_v2'.
    probe_out: dict filled with out-of-fold race predictions of the leakage probe
    (fixed['leakage_probe']) on z of this model, z of model A (A's chosen setting and
    epochs refit on the same folds and seed, i.e. A's own refit) and the raw CLIP inputs.
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
        if select in ("worst_group_v2", "worst_group_v3"):
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
            elif select == "worst_group_v3":
                stats = [selection_stats_v3(p, y[va], groups[va], ref) for p in vp]
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
        probe_rec = None
        if fixed.get("leakage_probe"):
            probe_rec = _leakage_fold(X, y, groups, tr_all, te, st, predict_fn, ref_A[k], seed,
                                      k, device, probe_out)
        rec = {"test_fold": k, "inner_val_fold": v, "select": select, "chosen": win["setting"],
               "epochs": win["best_epoch"], "inner_val_mae": win["val_mae"],
               "test_mae": test_mae, "trials": trials, "refit_info": info,
               "seconds": round(time.time() - t0, 1)}
        if select != "mae":
            rec["inner_val_worst_group_tpr"] = win["val_worst_group_tpr"]
            rec["inner_val_feasible"] = win["val_feasible"]
        if probe_rec is not None:
            rec["leakage_probe"] = probe_rec
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
