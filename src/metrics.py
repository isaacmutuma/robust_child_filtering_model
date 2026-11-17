"""Evaluation metrics and success rule (docs/SUCCESS_CRITERIA.md §1-4).

Input: one row per UTKFace image with columns id, group, age (true) and pred_age
(out-of-fold prediction). Flagged as child if pred_age < 18.

Conventions (ported from the audit's src/analysis/metrics.py):
- Minor TPR is age-stratified (0-9 / 10-17) with the FIXED pooled-minor weights
  0.7234 / 0.2766, so a group with older children is not penalised.
- Bootstrap: images resampled with replacement within each group (2,000 draws, seed 0),
  implemented as multinomial count matrices; percentile 95% CIs.
- Pairing: a BootPlan is built once from the image set (sorted by id) and reused for
  every model, so model A and model M are scored on the SAME resamples. Because the plan
  is deterministic given (ids, groups, seed), separate runs reproduce the same pairing.
- Paired test (criterion 1): one-sided bootstrap p for (gap_A - gap_M) > 0,
  p = (#{replicates <= 0} + 1) / (B + 1); Holm across the comparison family.
"""

import numpy as np
import pandas as pd

from configs import config as C

STRATA = list(C.STRATUM_WEIGHTS)                     # ["0-9", "10-17"]


# ── data ─────────────────────────────────────────────────────────────────────
def prepare(df):
    """Sorted copy with derived columns: flagged, minor, teen, adult, stratum, abs_err."""
    need = {"id", "group", "age", "pred_age"}
    if not need <= set(df.columns):
        raise ValueError(f"missing columns {need - set(df.columns)}")
    if df["id"].duplicated().any():
        raise ValueError("duplicate ids")
    if df["pred_age"].isna().any():
        raise ValueError("missing predictions")
    d = df.sort_values("id", kind="stable").reset_index(drop=True).copy()
    lo, hi = C.ADULT_FPR_RANGE
    d["flagged"] = (d.pred_age < C.CHILD_AGE_THRESHOLD).astype(float)
    d["minor"] = d.age <= C.MINOR_MAX_AGE
    d["teen"] = (d.age >= 10) & d.minor
    d["adult"] = (d.age >= lo) & (d.age <= hi)
    d["stratum"] = np.where(d.age <= 9, "0-9", np.where(d.minor, "10-17", ""))
    d["err"] = d.pred_age - d.age
    d["abs_err"] = d.err.abs()
    return d


class BootPlan:
    """Within-group multinomial resample counts, shared by every model (pairing).

    W[g] has shape (B, n_g): how often each image of group g is drawn in each replicate.
    """

    def __init__(self, d, B=None, seed=None):
        self.B = B or C.N_BOOTSTRAP
        rng = np.random.default_rng(C.SEED if seed is None else seed)
        self.ids = d["id"].to_numpy()
        self.idx, self.W = {}, {}
        for g in sorted(d.group.unique()):
            idx = np.flatnonzero(d.group.to_numpy() == g)
            n = len(idx)
            self.idx[g] = idx
            self.W[g] = rng.multinomial(n, np.full(n, 1.0 / n), size=self.B).astype(np.float32)

    def check(self, d):
        """Raise unless `d` (prepared) has exactly the images this plan was built on."""
        if not np.array_equal(d["id"].to_numpy(), self.ids):
            raise ValueError("model predictions cover a different image set than the plan")


# ── estimates: point value + bootstrap replicates ────────────────────────────
def _mean(d, plan, g, values, mask):
    """(point, reps) for mean(values | mask) within group g."""
    i = plan.idx[g]
    v = np.asarray(values, float)[i]
    m = np.asarray(mask, float)[i]
    if m.sum() == 0:
        return float("nan"), np.full(plan.B, np.nan)
    W = plan.W[g]
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = (W @ (v * m)) / (W @ m)
    return float((v * m).sum() / m.sum()), reps


def _strat(d, plan, g, values):
    """(point, reps) for the stratified minor mean of `values` within group g."""
    pt, reps = 0.0, np.zeros(plan.B)
    for s, w in C.STRATUM_WEIGHTS.items():
        p, r = _mean(d, plan, g, values, d.stratum.to_numpy() == s)
        pt, reps = pt + w * p, reps + w * r
    return pt, reps


def ci(reps):
    """Percentile 95% interval of bootstrap replicates (NaNs ignored)."""
    reps = np.asarray(reps, float)
    reps = reps[~np.isnan(reps)]
    if len(reps) == 0:
        return [None, None]
    a = C.ALPHA / 2
    return [float(np.quantile(reps, a)), float(np.quantile(reps, 1 - a))]


def _block(pt, reps):
    return {"est": None if np.isnan(pt) else float(pt), "ci": ci(reps)}


def estimates(d, plan):
    """All point estimates + replicates for one model, keyed by metric name."""
    plan.check(d)
    E = {}
    for g in plan.idx:
        E[f"tpr_strat|{g}"] = _strat(d, plan, g, d.flagged)
        E[f"tpr_teen|{g}"] = _mean(d, plan, g, d.flagged, d.teen)
        E[f"tpr_0_9|{g}"] = _mean(d, plan, g, d.flagged, d.stratum == "0-9")
        E[f"fpr_adult|{g}"] = _mean(d, plan, g, d.flagged, d.adult)
        E[f"age_err_minor_strat|{g}"] = _strat(d, plan, g, d.err)
        E[f"mae|{g}"] = _mean(d, plan, g, d.abs_err, np.ones(len(d), bool))
    a, b = C.PRIMARY_CONTRAST
    E["tpr_gap_W_minus_B"] = tuple(x - y for x, y in zip(E[f"tpr_strat|{a}"], E[f"tpr_strat|{b}"]))
    E["tpr_teen_gap_W_minus_B"] = tuple(x - y for x, y in
                                        zip(E[f"tpr_teen|{a}"], E[f"tpr_teen|{b}"]))
    E["fpr_diff_B_minus_W"] = tuple(x - y for x, y in zip(E[f"fpr_adult|{b}"], E[f"fpr_adult|{a}"]))
    E["age_err_gap_B_minus_W"] = tuple(x - y for x, y in zip(E[f"age_err_minor_strat|{b}"],
                                                             E[f"age_err_minor_strat|{a}"]))
    tested = [g for g in C.TESTED_GROUPS if g in plan.idx]
    E["worst_group_tpr_strat"] = (min(E[f"tpr_strat|{g}"][0] for g in tested),
                                  np.min([E[f"tpr_strat|{g}"][1] for g in tested], axis=0))
    # overall MAE over ALL images (incl. Others): group means weighted by group size,
    # so each replicate keeps the observed group sizes (resampling is within group)
    n = {g: len(i) for g, i in plan.idx.items()}
    N = sum(n.values())
    E["mae_overall"] = (sum(n[g] * E[f"mae|{g}"][0] for g in n) / N,
                        sum(n[g] * E[f"mae|{g}"][1] for g in n) / N)
    return E


def model_report(d, plan):
    """JSON-ready report for one model: per tested group + contrasts, with CIs."""
    E = estimates(d, plan)
    groups = {}
    for g in C.TESTED_GROUPS:
        if g not in plan.idx:
            continue
        gd = d.iloc[plan.idx[g]]
        groups[g] = {"n": int(len(gd)), "n_minor": int(gd.minor.sum()),
                     "n_teen_10_17": int(gd.teen.sum()), "n_adult_20_59": int(gd.adult.sum()),
                     **{k.split("|")[0]: _block(*v) for k, v in E.items() if k.endswith(f"|{g}")}}
    keys = ["tpr_gap_W_minus_B", "tpr_teen_gap_W_minus_B", "fpr_diff_B_minus_W",
            "age_err_gap_B_minus_W", "worst_group_tpr_strat", "mae_overall"]
    return {"by_group": groups, **{k: _block(*E[k]) for k in keys},
            "n_images": int(len(d)), "bootstrap": {"B": plan.B, "seed": C.SEED,
                                                   "scheme": "within group, multinomial"}}


# ── paired comparison with model A and the success rule ──────────────────────
def paired_vs_A(dA, dM, plan):
    """Paired bootstrap of (gap_A - gap_M) and (MAE_M - MAE_A) on the same resamples."""
    for col in ("group", "age"):
        if not np.array_equal(dA[col].to_numpy(), dM[col].to_numpy()):
            raise ValueError(f"{col} differs between models")
    EA, EM = estimates(dA, plan), estimates(dM, plan)
    gA, gM = EA["tpr_gap_W_minus_B"], EM["tpr_gap_W_minus_B"]
    diff_pt, diff_reps = gA[0] - gM[0], gA[1] - gM[1]
    p = (np.sum(diff_reps <= 0) + 1) / (plan.B + 1)
    mae_pt, mae_reps = EM["mae_overall"][0] - EA["mae_overall"][0], \
        EM["mae_overall"][1] - EA["mae_overall"][1]
    return {"gap_reduction_A_minus_M": {"est": float(diff_pt), "ci": ci(diff_reps),
                                        "p_one_sided": float(p)},
            "mae_increase_M_minus_A": {"est": float(mae_pt), "ci": ci(mae_reps)}}


def holm(pvals):
    """Holm step-down adjusted p-values for a dict name -> p."""
    keys = list(pvals)
    ps = np.array([pvals[k] for k in keys], float)
    m, adj, running = len(ps), np.zeros(len(ps)), 0.0
    for rank, i in enumerate(np.argsort(ps, kind="stable")):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return {k: float(adj[i]) for i, k in enumerate(keys)}


def verdict(rep_A, rep_M, paired, p_holm):
    """Apply SUCCESS_CRITERIA.md §4. Returns each criterion and 'closes'/'narrows'/'neither'."""
    S = C.SUCCESS
    gapA, gapM = rep_A["tpr_gap_W_minus_B"], rep_M["tpr_gap_W_minus_B"]
    c1 = bool(p_holm < C.ALPHA)
    c2 = bool(gapM["est"] <= S["small_gap_max"] and gapM["ci"][1] < gapA["est"])
    fpr = {g: (rep_M["by_group"][g]["fpr_adult"]["est"], rep_A["by_group"][g]["fpr_adult"]["est"])
           for g in rep_M["by_group"]}
    c3_by = {g: bool(m <= S["adult_fpr_max"] and m <= a + S["adult_fpr_vs_A"])
             for g, (m, a) in fpr.items()}
    c3 = all(c3_by.values())
    c4 = bool(paired["mae_increase_M_minus_A"]["ci"][1] <= S["mae_noninferiority_years"])
    outcome = "closes" if (c1 and c2 and c3 and c4) else \
        "narrows" if (c1 and c3 and c4) else "neither"
    return {"c1_fairer_holm": c1, "p_holm": float(p_holm), "c2_small_gap": c2,
            "c3_adult_fpr": c3, "c3_by_group": c3_by, "c4_mae_noninferior": c4,
            "outcome": outcome}


def a_self_check(rep_A):
    """§4 'Model A itself': does A's own gap CI already contain 0?"""
    lo, hi = rep_A["tpr_gap_W_minus_B"]["ci"]
    return {"gap_ci_contains_0": bool(lo <= 0 <= hi), "gap": rep_A["tpr_gap_W_minus_B"]}


def frame(ids, groups, ages, preds):
    """Convenience: DataFrame in the format prepare() expects."""
    return pd.DataFrame({"id": ids, "group": groups, "age": np.asarray(ages, float),
                         "pred_age": np.asarray(preds, float)})
