"""Follow-up 3: at which depth of CLIP does the Black-White age bias appear? (docs/FOLLOWUP_LAYERS.md)

  python scripts/followup_layers.py --work ~/fwork            # course GPU server (UTKFace already in WORK)

Runs every UTKFace image once through frozen CLIP ViT-L/14 and keeps, at every depth
(0 = patch + position embedding after ln_pre, l = output of block l, l = 1..24), the CLS token
and the mean of the 256 patch tokens (fp16 memmap, ~2.4 GB, never in git). The final CLIP
embedding (model A's input) is kept as an anchor.

For each representation, with the same 5 folds and seed as every model:
- age readout: ridge regression on standardised features; alpha chosen on the inner
  validation fold ((k+1) % 5) by MAE, refit on the 4 training folds, predict fold k;
- the same metrics as every model (src/metrics.py: stratified White-Black TPR gap, teen gap,
  Black-White over-estimation of minors, MAE; within-group bootstrap, paired plan), plus an
  age-matched Black-White error gap (per true year 0-17, pooled-minor weights);
- race readability: multinomial logistic probe (L2 as sklearn C=1), pooled out-of-fold
  balanced accuracy (chance 0.20) with bootstrap CIs (src/leakage.summarise).
Exploratory: no Holm, no success verdict. Aggregate JSON -> results/followups/layers.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _common import REPO
from configs import config as C
from src import leakage as LK
from src import metrics as M
from src.folds import make_folds
from src.provenance import write_result

DEPTHS = 25                     # 0 = embeddings after ln_pre, 1..24 = after block l
POOLS = ("cls", "mean")
ALPHAS = [1e-1, 1, 10, 100, 1e3, 1e4, 1e5]
F_INPUT_DEPTH = {"F1_4": 20, "F1_1": 23}   # tokens entering F's first LoRA block
SHARE_FORMED, SHARE_LATE = 0.75, 0.25      # reading rule thresholds (fixed before the run)


# ── extraction ───────────────────────────────────────────────────────────────
def layer_features(visual, images):
    """(batch, DEPTHS, 2, width) CLS and patch-mean at every depth + the final CLIP embedding."""
    x = visual._embeds(images)
    out = [x]
    for blk in visual.transformer.resblocks:
        x = blk(x)
        out.append(x)
    pooled, _ = visual._pool(x)
    emb = pooled @ visual.proj if visual.proj is not None else pooled
    import torch
    feats = [[t[:, 0], t[:, 1:].mean(1)] for t in out]
    return torch.stack([torch.stack(f, 1) for f in feats], 1), emb


def extract(rows, work, device, bs=64):
    import torch
    from PIL import Image
    from src.clip_model import load_clip
    model, preprocess, _, winfo = load_clip(device)
    visual = model.visual.float()
    n, d = len(rows), visual.transformer.width
    f_path, e_path = work / "layers_cls_mean.f16", work / "layers_final_emb.npy"
    shape = (n, DEPTHS, 2, d)
    if f_path.exists() and e_path.exists():
        return np.memmap(f_path, np.float16, "r", shape=shape), np.load(e_path), visual, winfo
    feats = np.memmap(f_path, np.float16, "w+", shape=shape)
    emb = np.empty((n, visual.output_dim), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, n, bs):
            x = torch.stack([preprocess(Image.open(r["path"]).convert("RGB")) for r in rows[b:b + bs]]).to(device)
            f, e = layer_features(visual, x)
            feats[b:b + bs] = f.cpu().numpy().astype(np.float16)
            emb[b:b + bs] = e.float().cpu().numpy()
            print(f"\rlayers {min(b + bs, n)}/{n} ({time.time() - t0:.0f} s)", end="", flush=True)
    print()
    feats.flush()
    np.save(e_path, emb)
    return np.memmap(f_path, np.float16, "r", shape=shape), emb, visual, winfo


def consistency_checks(feats, emb, work, visual, device, n_check=256):
    """Final embedding vs F's stored exact embeddings; depth-20 features vs F's token cache."""
    import torch
    out = {}
    ef = work / "emb_fp32.npy"
    if ef.exists():
        ref = np.load(ef, mmap_mode="r")
        out["final_emb_max_abs_diff_vs_F_cache"] = float(np.abs(emb - ref).max())
    tf = work / "tokens_b20.f16"
    if tf.exists():
        tok = np.memmap(tf, np.float16, "r", shape=(len(emb), 257, visual.transformer.width))[:n_check]
        tok = tok.astype(np.float32)
        mine = feats[:n_check, 20].astype(np.float32)
        out["depth20_cls_max_abs_diff_vs_F_cache"] = float(np.abs(mine[:, 0] - tok[:, 0]).max())
        out["depth20_mean_max_abs_diff_vs_F_cache"] = float(np.abs(mine[:, 1] - tok[:, 1:].mean(1)).max())
        out["depth20_feature_scale"] = float(np.abs(mine).mean())
    return out


# ── readouts (torch, float64 for ridge) ──────────────────────────────────────
def _std(Xtr):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def ridge_solutions(Xtr, ytr, alphas):
    """{alpha: (w, b)} for ridge on (already standardised) Xtr, intercept unpenalised."""
    import torch
    ym = ytr.mean()
    G = Xtr.T @ Xtr
    r = Xtr.T @ (ytr - ym)
    eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    return {a: (torch.linalg.solve(G + a * eye, r), ym) for a in alphas}


def ridge_oof(F, y, folds, device, alphas=ALPHAS):
    """Out-of-fold age predictions; alpha chosen on the inner validation fold by MAE."""
    import torch
    X = torch.as_tensor(np.asarray(F, np.float64), device=device)
    Y = torch.as_tensor(y, dtype=torch.float64, device=device)
    K = int(folds.max()) + 1
    pred, record = np.full(len(y), np.nan), []
    for k in range(K):
        v = (k + 1) % K
        tr_in = torch.as_tensor((folds != k) & (folds != v), device=device)
        va = torch.as_tensor(folds == v, device=device)
        tr_all = torch.as_tensor(folds != k, device=device)
        te = folds == k
        mu, sd = _std(X[tr_in])
        sols = ridge_solutions((X[tr_in] - mu) / sd, Y[tr_in], alphas)
        Xva = (X[va] - mu) / sd
        maes = {a: float(((Xva @ w + b) - Y[va]).abs().mean()) for a, (w, b) in sols.items()}
        best = min(maes, key=maes.get)
        mu, sd = _std(X[tr_all])
        w, b = ridge_solutions((X[tr_all] - mu) / sd, Y[tr_all], [best])[best]
        Xte = (X[torch.as_tensor(te, device=device)] - mu) / sd
        pred[te] = (Xte @ w + b).cpu().numpy()
        record.append({"test_fold": k, "alpha": best, "inner_val_mae": maes[best],
                       "alpha_at_grid_edge": best in (alphas[0], alphas[-1])})
    return pred, record


def logistic_probe_oof(F, groups, folds, device, iters=100, seed=0):
    """Race predictions, multinomial logistic regression (loss = sum CE + 0.5 |W|^2, like sklearn C=1)."""
    import torch
    torch.manual_seed(seed)
    classes, codes = np.unique(groups, return_inverse=True)
    X = torch.as_tensor(np.asarray(F, np.float32), device=device)
    Yc = torch.as_tensor(codes, device=device)
    out = np.empty(len(groups), dtype=object)
    for k in range(int(folds.max()) + 1):
        tr, te = torch.as_tensor(folds != k, device=device), folds == k
        mu, sd = _std(X[tr])
        Xtr, ytr = (X[tr] - mu) / sd, Yc[tr]
        W = torch.zeros(X.shape[1], len(classes), device=device, requires_grad=True)
        b = torch.zeros(len(classes), device=device, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
        n = Xtr.shape[0]

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr) + 0.5 * (W ** 2).sum() / n
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            Xte = (X[torch.as_tensor(te, device=device)] - mu) / sd
            out[te] = classes[(Xte @ W + b).argmax(1).cpu().numpy()]
    return out


# ── extra metric: age-matched Black - White over-estimation ──────────────────
def age_matched_gap(d, plan, a="Black", b="White"):
    """Sum over true ages 0..17 of w_age * (mean err_a - mean err_b); w = pooled minor age shares
    over ages where both groups have images. Same bootstrap resamples as the plan."""
    age = d.age.to_numpy().astype(int)
    err = d.err.to_numpy()
    minor = d.minor.to_numpy()
    years = [t for t in range(C.MINOR_MAX_AGE + 1)
             if all(((age[plan.idx[g]] == t) & minor[plan.idx[g]]).any() for g in (a, b))]
    w = np.array([np.sum((age == t) & minor) for t in years], float)
    w /= w.sum()

    def per_year(g):
        i = plan.idx[g]
        pt, reps = [], []
        for t in years:
            m = ((age[i] == t) & minor[i]).astype(float)
            v = err[i] * m
            pt.append(v.sum() / m.sum())
            with np.errstate(invalid="ignore", divide="ignore"):
                reps.append((plan.W[g] @ v) / (plan.W[g] @ m))
        return np.array(pt), np.stack(reps, 1)

    pa, ra = per_year(a)
    pb, rb = per_year(b)
    diff = ra - rb
    ok = ~np.isnan(diff)                               # a resample can miss a year: renormalise
    reps = np.nansum(np.where(ok, diff, 0) * w, 1) / (ok * w).sum(1)
    return {"est": float(((pa - pb) * w).sum()), "ci": M.ci(reps), "years": years}


# ── per-representation evaluation ────────────────────────────────────────────
def evaluate(name, F, ids, y, groups, folds, plan, device):
    t0 = time.time()
    pred, rec = ridge_oof(F, y, folds, device)
    d = M.prepare(M.frame(ids, groups, y, pred))
    rep = M.model_report(d, plan)
    race = logistic_probe_oof(F, groups, folds, device)
    keep = ["tpr_gap_W_minus_B", "tpr_teen_gap_W_minus_B", "age_err_gap_B_minus_W", "fpr_diff_B_minus_W",
            "worst_group_tpr_strat", "mae_overall"]
    by_group = {g: {k: rep["by_group"][g][k] for k in ("tpr_strat", "tpr_teen", "fpr_adult",
                                                       "age_err_minor_strat", "mae")}
                for g in rep["by_group"]}
    return {"name": name, **{k: rep[k] for k in keep}, "by_group": by_group,
            "age_err_gap_B_minus_W_age_matched": age_matched_gap(d, plan),
            "ridge_folds": rec, "seconds": round(time.time() - t0, 1)}, race


def share(x, final):
    return None if final is None or abs(final) < 1e-9 or x is None else float(x / final)


def reading(res, pool="cls"):
    """Apply the reading rules fixed in docs/FOLLOWUP_LAYERS.md before the run."""
    fin = res["final_embedding"]
    out = {}
    for metric in ("tpr_gap_W_minus_B", "age_err_gap_B_minus_W"):
        fv = fin[metric]["est"]
        depths = [l for l in range(DEPTHS) if f"{pool}_{l}" in res]
        sig = [(res[f"{pool}_{l}"][metric]["ci"][0] or 0) > 0 for l in depths]
        emerge = next((depths[i] for i in range(len(depths)) if all(sig[i:])), None)
        at = {f: share(res[f"{pool}_{l}"][metric]["est"], fv) for f, l in F_INPUT_DEPTH.items()}
        out[metric] = {"final": fv, "share_of_final_at_F_input": at,
                       "first_depth_from_which_ci_excludes_0": emerge}
    s20 = [out[m]["share_of_final_at_F_input"]["F1_4"] for m in out]
    l20 = F_INPUT_DEPTH["F1_4"]
    ci20 = [(res[f"{pool}_{l20}"][m]["ci"][0] or 0) > 0 for m in out]
    if all(s is not None and s >= SHARE_FORMED for s in s20) and all(ci20):
        case = "formed_before_F"       # bias already present at the input of F's LoRA blocks
    elif all(s is not None and s <= SHARE_LATE for s in s20):
        case = "formed_in_F_blocks"    # F could reach it; its failure is not about depth
    else:
        case = "partly_before_F"
    out["case"] = case
    return out


def main():
    import pandas as pd
    import torch
    from run_model_f import get_utkface
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    work = Path(a.work).expanduser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_all = time.time()
    rows = get_utkface(work)
    ids = np.array([r["id"] for r in rows])
    y = np.array([r["age"] for r in rows], float)
    groups = np.array([r["group"] for r in rows])
    fold_of = make_folds(list(ids), list(groups), list(y))
    folds = np.array([fold_of[i] for i in ids])
    feats, emb, visual, winfo = extract(rows, work, device, a.bs)
    checks = consistency_checks(feats, emb, work, visual, device)
    print("checks:", checks)

    plan = M.BootPlan(M.prepare(M.frame(ids, groups, y, y)))
    results, race_out = {}, {}
    reps = [("final_embedding", lambda: emb)]
    for l in range(DEPTHS):
        for p, pool in enumerate(POOLS):
            if l == 0 and pool == "cls":
                continue                 # CLS before block 1 is the same for every image
            reps.append((f"{pool}_{l}", lambda l=l, p=p: np.asarray(feats[:, l, p], np.float32)))
    for name, get in reps:
        r, race = evaluate(name, get(), ids, y, groups, folds, plan, device)
        results[name], race_out[name] = r, race
        print(f"{name:16s} gap {r['tpr_gap_W_minus_B']['est']:+.3f} "
              f"B-W err {r['age_err_gap_B_minus_W']['est']:+.2f} "
              f"(age-matched {r['age_err_gap_B_minus_W_age_matched']['est']:+.2f}) "
              f"MAE {r['mae_overall']['est']:.2f} ({r['seconds']:.0f} s)", flush=True)
    race = LK.summarise(race_out, groups)
    for name in results:
        results[name]["race_probe_bal_acc"] = race["bal_acc"][name]
    A = json.loads((REPO / "results/models/A.json").read_text())["report"]
    out = {"what": "layer-wise ridge age readout + race probe on frozen CLIP ViT-L/14 (UTKFace)",
           "depths": "0 = embeddings after ln_pre; l = output of block l (1..24)",
           "pools": POOLS, "alphas": ALPHAS, "f_input_depth": F_INPUT_DEPTH,
           "thresholds": {"formed": SHARE_FORMED, "late": SHARE_LATE},
           "checks": checks, "clip_weights": winfo, "device": device, "torch": torch.__version__,
           "model_A_reference": {k: A[k] for k in ("tpr_gap_W_minus_B", "age_err_gap_B_minus_W", "mae_overall")},
           "race_probe_chance": race["chance"], "results": results,
           "reading_cls": reading(results, "cls"), "reading_mean": reading(results, "mean"),
           "seconds_total": round(time.time() - t_all, 1)}
    res = write_result(REPO / "results" / "followups" / "layers.json", out, C.as_dict(), REPO)
    print("reading (CLS):", json.dumps(out["reading_cls"], indent=1))
    print("->", res)


if __name__ == "__main__":
    main()
