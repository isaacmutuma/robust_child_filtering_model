"""Build the course submission notebook: notebooks/submission/DLP_term_project.ipynb.

  python scripts/build_submission_notebook.py

The notebook is self-contained: it downloads the pinned data and CLIP weights, computes the
embeddings, builds the folds and trains every condition, so it can be reproduced on Colab
from the notebook alone. Evaluation code (metrics, folds, leakage probe, LEACE) is copied
verbatim from src/ (only the import lines are removed); the training code is a shorter
version of src/train.py that keeps only what the submitted conditions use.
tests/test_submission_notebook.py checks that both give identical results to the repo.
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "submission" / "DLP_term_project.ipynb"
GITHUB_RAW = "https://raw.githubusercontent.com/isaacmutuma/robust_child_filtering_model/main"


def module_code(rel):
    """Source of a repo module without its local imports and module docstring."""
    src = (REPO / rel).read_text()
    src = re.sub(r'^"""[\s\S]*?"""\n', "", src, count=1)
    keep = [l for l in src.split("\n")
            if not re.match(r"^(from (configs|src)[\w.]* import|import (configs|src))", l)]
    return "\n".join(keep).strip() + "\n"


def followup3_code():
    """Library part of scripts/followup_layers.py for the notebook (same functions, notebook names)."""
    src = (REPO / "scripts/followup_layers.py").read_text()
    body = src.split("# ── extraction")[1].split("\ndef extract(")[0]
    rest = src.split("# ── readouts")[1].split("\ndef main(")[0]
    code = "# ── follow-up 3: features at every depth" + body + "\n# ── readouts" + rest
    code = code.replace("M.", "").replace("LK.", "").replace("def evaluate(", "def evaluate_depth(")
    head = src.split("DEPTHS = ")[1].split("\n\n\n")[0]
    return "DEPTHS = " + head + "\n\n\n" + code


LAYER_FIG = r'''# figure: one panel per measure, CLS and mean-patch readouts across depth; F's LoRA blocks shaded;
# grey diamond = linear readout of the final CLIP embedding (with 95% CI)
COL = {"cls": "#257de0", "mean": "#f47230"}
MARK = {"cls": "o", "mean": "s"}
LABEL = {"cls": "CLS token", "mean": "mean patch token"}
panels = [("tpr_gap_W_minus_B", "White − Black TPR gap (minors)"),
          ("age_err_gap_B_minus_W_age_matched", "Black − White age over-estimation,\nage-matched (years)"),
          ("race_probe_bal_acc", "race readable by a linear probe\n(balanced accuracy, chance 0.20)")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (key, title) in zip(axes, panels):
    ax.axvspan(19.5, 24.5, color="#000000", alpha=0.06, lw=0)
    for pool in POOLS:
        ls = [l for l in range(DEPTHS) if f"{pool}_{l}" in LAYERS]
        est = np.array([LAYERS[f"{pool}_{l}"][key]["est"] for l in ls])
        ci_ = np.array([LAYERS[f"{pool}_{l}"][key]["ci"] for l in ls], float)
        ax.fill_between(ls, ci_[:, 0], ci_[:, 1], color=COL[pool], alpha=0.15, lw=0)
        ax.plot(ls, est, color=COL[pool], lw=2, marker=MARK[pool], ms=4, label=LABEL[pool])
    fin = LAYERS["final_embedding"][key]
    ax.errorbar([26.5], [fin["est"]], yerr=[[fin["est"] - fin["ci"][0]], [fin["ci"][1] - fin["est"]]],
                fmt="D", color="#555555", ms=6, capsize=3, lw=1.2)
    ax.set_xticks([0, 5, 10, 15, 20, 24, 26.5]); ax.set_xticklabels(["0", "5", "10", "15", "20", "24", "final\nemb."])
    if key != "race_probe_bal_acc":
        ax.axhline(0, color="#999999", lw=0.8)
    ax.set_title(title, fontsize=10); ax.set_xlabel("depth (block of CLIP ViT-L/14)")
    ax.set_xlim(-0.5, 28); ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    lo, hi = ax.get_ylim()
    ax.text(22, hi, "blocks F fine-tunes", ha="center", va="top", fontsize=8, color="#333333")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h + [plt.Line2D([], [], color="#555555", marker="D", ls="")], l + ["linear readout of the final CLIP embedding"],
           loc="upper center", ncol=3, frameon=False, fontsize=9)
plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.show()'''

TEEN_FIG = r'''# figure: teenagers (10-17) flagged as children, per group and model, with 95% CIs
FIG_MODELS = ["A", "E_inv", "F1_1", "F1_4", "F2"]
FIG_NAMES = {"A": "A\nbaseline", "E_inv": "E\nadversary", "F1_1": "F1-1\nLoRA, 1 block",
             "F1_4": "F1-4\nLoRA, 4 blocks", "F2": "F2\nLoRA + adversary"}
GCOL = {"White": "#257de0", "Black": "#f47230", "Asian": "#00b894", "Indian": "#f2a600"}
GMARK = {"White": "o", "Black": "s", "Asian": "^", "Indian": "D"}
OFF = {"White": -0.24, "Asian": -0.08, "Indian": 0.08, "Black": 0.24}
fig, ax = plt.subplots(figsize=(9, 4.4))
for g in ["White", "Asian", "Indian", "Black"]:
    est = np.array([FIG_RESULTS[m]["by_group"][g]["tpr_teen"]["est"] for m in FIG_MODELS])
    ci_ = np.array([FIG_RESULTS[m]["by_group"][g]["tpr_teen"]["ci"] for m in FIG_MODELS], float)
    ax.errorbar(np.arange(len(FIG_MODELS)) + OFF[g], est, yerr=[est - ci_[:, 0], ci_[:, 1] - est], fmt=GMARK[g],
                color=GCOL[g], ms=7, capsize=3, lw=1.5,
                label=f"{g} (n = {FIG_RESULTS['A']['by_group'][g]['n_teen_10_17']})")
ax.set_xticks(range(len(FIG_MODELS))); ax.set_xticklabels([FIG_NAMES[m] for m in FIG_MODELS], fontsize=9)
ax.set_ylabel("share of 10-17-year-olds flagged as children"); ax.set_ylim(0, 1)
ax.set_title("Teenagers detected, by group and model (95% CI)", fontsize=11)
ax.grid(axis="y", color="#e6e6e6", lw=0.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.tight_layout(); plt.show()'''


CONSTANTS = '''# Fixed settings (chosen before any model was trained; same for every condition)
from types import SimpleNamespace
C = SimpleNamespace(
    SEED=0, N_FOLDS=5, N_BOOTSTRAP=2000, ALPHA=0.05,
    CHILD_AGE_THRESHOLD=18,                 # flagged as a child if predicted age < 18
    MINOR_MAX_AGE=17, ADULT_FPR_RANGE=(20, 59),
    MINOR_STRATA=[(0, 9), (10, 17)],
    STRATUM_WEIGHTS={"0-9": 0.7234, "10-17": 0.2766},   # same age weights for every group
    FOLD_AGE_BANDS=[(0, 9), (10, 17), (18, 19), (20, 39), (40, 59), (60, 200)],
    UTKFACE_RACES={0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"},
    TESTED_GROUPS=["White", "Black", "Asian", "Indian"],  # "Others" is trained on, not reported
    PRIMARY_CONTRAST=("White", "Black"),
    SUCCESS={"small_gap_max": 0.03, "adult_fpr_max": 0.05, "adult_fpr_vs_A": 0.02,
             "mae_noninferiority_years": 0.5},
    TUNING={"tpr_drop_max": 0.02},
    HOLM_FAMILY=["E_inv", "F1_1", "F1_4", "F2"],
)
UTKFACE = {"hf_repo": "py97/UTKFace-Cropped", "revision": "e80f21fb21fa3631380c3cb634b79be3ac2f4cba",
           "file": "UTKFace.tar.gz",
           "sha256": "2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7",
           "n_records": 23705}
CLIP_ARCH = "ViT-L-14"
CLIP_WEIGHTS = {"hf_repo": "timm/vit_large_patch14_clip_224.openai",
                "revision": "18d0535469bb561bf468d76c1d73aa35156c922b",
                "file": "open_clip_model.safetensors",
                "sha256": "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"}
'''

TRAIN = r'''# ── Age head, model E and the out-of-fold loop (my implementation) ───────────
import itertools, time
import numpy as np
import torch
import torch.nn as nn

GRID_A = {"hidden": [0, 512], "lr": [1e-3, 3e-4], "weight_decay": [1e-4, 1e-2]}
FIXED_A = {"max_epochs": 60, "batch_size": 256, "dropout": 0.1, "loss": "l1", "method": "plain"}
GRID_E = {"hidden": [512], "lr": [1e-3, 3e-4], "weight_decay": [1e-4, 1e-2],
          "adv_lambda": [0.1, 0.3, 1.0, 3.0], "adv_target": ["max_ce", "prior"]}
FIXED_E = {**FIXED_A, "method": "adv_inv", "adv_hidden": 128, "adv_lr": 1e-2, "adv_steps": 5,
           "adv_weight_decay": 1e-4, "leakage_probe": True}


def grid_settings(grid):
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


def make_head(dim, hidden, dropout):
    """Age head: linear, or MLP = Linear + GELU (= z, 512 numbers) -> Dropout -> Linear."""
    if hidden == 0:
        return nn.Linear(dim, 1)
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


def make_adversary(dim, hidden, n_classes):
    """Race classifier on z (model E)."""
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))


def predict(model, X):
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        Xt = X if torch.is_tensor(X) else torch.from_numpy(X)
        return model(Xt.to(dev)).squeeze(1).cpu().numpy().astype(float)


def encode(model, X):
    """z = output of the MLP's hidden layer."""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 4096):
            out.append(model[:2](torch.from_numpy(np.ascontiguousarray(X[b:b + 4096])).to(dev)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def fit_head(Xtr, ytr, s, epochs, seed, device, Xval=None, gtr=None, atr=None):
    """Train one age head. Returns (predict_fn, per-epoch validation predictions, info).

    method 'plain' (model A): L1 on the standardised age.
    method 'adv_inv' (model E): per batch, (1) adv_steps updates of the race adversary on
    z with the MLP fixed, then (2) one update of the MLP on
    L1(age) - lambda * CE(race | z)  [target 'max_ce']  or
    L1(age) + lambda * KL(race base rates || adversary's guess)  [target 'prior'].
    lambda = 0 gives exactly model A (the adversary uses its own random-number stream).
    """
    method = s.get("method", "plain")
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = make_head(Xtr.shape[1], s["hidden"], s["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    loss_fn = nn.L1Loss()
    y_mu, y_sd = float(np.mean(ytr)), float(np.std(ytr)) + 1e-6
    X = torch.from_numpy(Xtr).to(device)
    y = torch.from_numpy(((ytr - y_mu) / y_sd).astype(np.float32)).to(device)

    def predict_fn(Xn):
        return predict(model, Xn) * y_sd + y_mu

    if s["hidden"] > 0:
        predict_fn.encode = lambda Xn: encode(model, Xn)
    predict_fn.model, predict_fn.y_mu, predict_fn.y_sd = model, y_mu, y_sd   # model F starts from A's head
    info = {"method": method}
    if method == "adv_inv":
        classes = sorted(set(np.asarray(gtr).tolist()))
        rid = torch.from_numpy(np.searchsorted(classes, np.asarray(gtr))).to(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 7919)
            adv = make_adversary(s["hidden"], s["adv_hidden"], len(classes))
        adv = adv.to(device)
        opt_adv = torch.optim.AdamW(adv.parameters(), lr=s["adv_lr"], weight_decay=s["adv_weight_decay"])
        ce_fn = nn.CrossEntropyLoss()
        lam, n_adv, target = float(s["adv_lambda"]), int(s["adv_steps"]), s.get("adv_target", "max_ce")
        log_prior = torch.log(torch.bincount(rid, minlength=len(classes)).float() / len(rid))
        enc, head = model[:2], model[2:]
        nC = len(classes)
        hist = {"adv_bal_acc": [], "adv_ce": []}

    Xv = torch.from_numpy(Xval).to(device) if Xval is not None else None
    val_preds = [] if Xval is not None else None
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X), generator=g)
        if method == "adv_inv":
            conf = torch.zeros(nC, nC, dtype=torch.long, device=device)
            ce_sum = 0.0
        for b in order.split(s["batch_size"]):
            b = b.to(device)
            if method == "adv_inv":
                zb = enc(X[b])
                zd = zb.detach()
                for _ in range(n_adv):                    # (1) adversary, MLP fixed
                    opt_adv.zero_grad()
                    ce_fn(adv(zd), rid[b]).backward()
                    opt_adv.step()
                opt.zero_grad()                           # (2) MLP
                out = head(zb).squeeze(1)
                logits = adv(zb)
                ce = ce_fn(logits, rid[b])
                loss = loss_fn(out, y[b])
                if lam != 0 and target == "max_ce":
                    loss = loss - lam * ce
                elif lam != 0:
                    logp = torch.log_softmax(logits, 1)
                    loss = loss + lam * (log_prior.exp() * (log_prior - logp)).sum(1).mean()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    conf += torch.bincount(rid[b] * nC + logits.argmax(1), minlength=nC * nC).view(nC, nC)
                    ce_sum += float(ce) * len(b)
                continue
            opt.zero_grad()
            out = model(X[b]).squeeze(1)
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
                     "adv_bal_acc_by_epoch": hist["adv_bal_acc"], "adv_ce_by_epoch": hist["adv_ce"]})
    return predict_fn, (np.stack(val_preds) if val_preds is not None else None), info


# ── tuning rule on the validation fold ───────────────────────────────────────
def selection_stats(pred, y, groups, per_group=False):
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
    return selection_stats(pred, y, groups, per_group=True)["tpr_by_group"]


def group_fpr_mae(pred, y, groups):
    pred, y, groups = np.asarray(pred), np.asarray(y), np.asarray(groups)
    flag = pred < C.CHILD_AGE_THRESHOLD
    lo, hi = C.ADULT_FPR_RANGE
    fpr = {}
    for g in C.TESTED_GROUPS:
        ad = (groups == g) & (y >= lo) & (y <= hi)
        fpr[g] = float(flag[ad].mean()) if ad.any() else float("nan")
    return {"fpr": fpr, "mae": float(np.abs(pred - y).mean())}


def selection_stats_v3(pred, y, groups, ref):
    """A setting is allowed only if, on the validation fold, every group's adult FPR is at
    most min(5%, A's + 2 points), MAE is at most A's + 0.5 years and no group's minor TPR
    falls more than 2 points below A's. Among allowed: best worst-group TPR, then lower MAE."""
    st = selection_stats(pred, y, groups)
    cur = group_fpr_mae(pred, y, groups)
    S = C.SUCCESS
    excess = [max(0.0, cur["fpr"][g] - min(S["adult_fpr_max"], ref["fpr"][g] + S["adult_fpr_vs_A"]))
              for g in C.TESTED_GROUPS]
    mae_excess = max(0.0, cur["mae"] - (ref["mae"] + S["mae_noninferiority_years"]))
    n_viol = sum(e > 0 for e in excess) + (mae_excess > 0)
    tpr = group_tpr(pred, y, groups)
    tpr_excess = [max(0.0, (ref["tpr"][g] - C.TUNING["tpr_drop_max"]) - tpr[g]) for g in C.TESTED_GROUPS]
    n_tpr = sum(e > 0 for e in tpr_excess)
    return {**st, "fpr_by_group": cur["fpr"], "tpr_by_group": tpr,
            "feasible": n_viol == 0 and n_tpr == 0, "n_violations": n_viol + n_tpr,
            "fpr_excess": float(sum(excess)), "tpr_excess": float(sum(tpr_excess)),
            "mae_excess": float(mae_excess)}


def rank_key(st, rule):
    if rule == "mae":
        return (st["mae"],)
    return (not st["feasible"], st["n_violations"], st["fpr_excess"], st["tpr_excess"],
            st["mae_excess"], -st["worst_group_tpr"], st["mae"])


def reference_A(rA, X, y, groups, tr_in, va, st, seed, k, device):
    """A's chosen setting for fold k, retrained on the same training folds (A's own seed)."""
    i = grid_settings(GRID_A).index({kk: rA["chosen"][kk] for kk in GRID_A})
    s = {**FIXED_A, **grid_settings(GRID_A)[i]}
    _, vp, _ = fit_head(st(X[tr_in]), y[tr_in], s, rA["epochs"], seed + 1000 * k + i, device, st(X[va]))
    ref = group_fpr_mae(vp[-1], y[va], groups[va])
    ref["tpr"] = group_tpr(vp[-1], y[va], groups[va])
    ref["mae_diff_vs_A_record"] = ref["mae"] - rA["inner_val_mae"]
    return ref


def leakage_fold(X, y, groups, tr_all, te, st, predict_fn, recA, seed, k, device, probe_out):
    """Race probe on z of this model, on z of model A, and on the CLIP inputs (test fold k)."""
    Xtr, Xte = st(X[tr_all]), st(X[te])
    pfA, _, _ = fit_head(Xtr, y[tr_all], {**FIXED_A, **recA["chosen"]}, recA["epochs"], seed + 1000 * k, device)
    reps = {"model": (predict_fn.encode(Xtr), predict_fn.encode(Xte)),
            "A": (pfA.encode(Xtr), pfA.encode(Xte)), "clip_raw": (Xtr, Xte)}
    out = {"A_refit_test_mae": float(np.abs(pfA(Xte) - y[te]).mean())}
    for name, (Ztr, Zte) in reps.items():
        p = fit_predict_probe(Ztr, groups[tr_all], Zte, seed)
        probe_out.setdefault(name, np.full(len(y), None, dtype=object))[te] = p
        out[f"bal_acc_{name}"] = balanced_accuracy(p, groups[te])
    return out


def oof_predictions(X, y, folds, grid, fixed, device="cpu", seed=None, log=print, groups=None,
                    select="mae", ref_A=None, probe_out=None, prep=None):
    """Out-of-fold predictions: for each test fold k, tune on folds != k, (k+1)%5 with
    validation fold (k+1)%5, refit on all four training folds, predict fold k."""
    seed = C.SEED if seed is None else seed
    K = int(folds.max()) + 1
    pred, record = np.full(len(y), np.nan), []
    for k in range(K):
        t0, v = time.time(), (k + 1) % K
        tr_in, va, tr_all, te = (folds != k) & (folds != v), folds == v, folds != k, folds == k
        fit_prep = (lambda m: Standardiser().fit(X[m])) if prep is None else \
            (lambda m: prep(X[m], None if groups is None else groups[m], y[m]))
        st = fit_prep(tr_in)
        ref = reference_A(ref_A[k], X, y, groups, tr_in, va, st, seed, k, device) if select != "mae" else None
        trials = []
        for i, s in enumerate(grid_settings(grid)):
            s = {**fixed, **s}
            _, vp, _ = fit_head(st(X[tr_in]), y[tr_in], s, s["max_epochs"], seed + 1000 * k + i, device,
                                st(X[va]), None if groups is None else groups[tr_in], y[tr_in])
            stats = [{"mae": float(np.abs(p - y[va]).mean())} if select == "mae"
                     else selection_stats_v3(p, y[va], groups[va], ref) for p in vp]
            best = min(range(len(stats)), key=lambda e: rank_key(stats[e], select))
            trials.append({"setting": {k2: s[k2] for k2 in grid}, "best_epoch": best + 1,
                           "val_mae": stats[best]["mae"], "_key": rank_key(stats[best], select),
                           **({} if select == "mae" else {"val_worst_group_tpr": stats[best]["worst_group_tpr"],
                                                         "val_feasible": stats[best]["feasible"]})})
        win = min(trials, key=lambda t: t["_key"])
        s = {**fixed, **win["setting"]}
        st = fit_prep(tr_all)
        predict_fn, _, info = fit_head(st(X[tr_all]), y[tr_all], s, win["best_epoch"], seed + 1000 * k, device,
                                       None, None if groups is None else groups[tr_all], y[tr_all])
        pred[te] = predict_fn(st(X[te]))
        rec = {"test_fold": k, "chosen": win["setting"], "epochs": win["best_epoch"],
               "inner_val_mae": win["val_mae"], "test_mae": float(np.abs(pred[te] - y[te]).mean()),
               "refit_info": info, "seconds": round(time.time() - t0, 1)}
        if select != "mae":
            rec["inner_val_worst_group_tpr"] = win["val_worst_group_tpr"]
            rec["reference_A_mae_diff"] = ref["mae_diff_vs_A_record"]
        if fixed.get("leakage_probe"):
            rec["leakage_probe"] = leakage_fold(X, y, groups, tr_all, te, st, predict_fn, ref_A[k], seed, k,
                                                device, probe_out)
        record.append(rec)
        log(f"fold {k}: chose {win['setting']} x {win['best_epoch']} epochs, test MAE {rec['test_mae']:.2f} "
            f"({rec['seconds']:.0f} s)")
    return pred, record
'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")}


def build():
    cells = []
    A = cells.append
    A(md("""
# DLP Term Project: Does a Fairness Fix Have to Change CLIP? Adversarial Training vs. Fine-Tuning for Child/Adult Detection

---

**Due date:** 2026.12.17 23:59

**Submission:** `202555404_IsaacMutuma.ipynb`, `202555404_IsaacMutuma_Report.pdf` and a 10-minute video.

**Track:** C (free topic), linked to Track A (ViT fine-tuning) and Track B (adversarial training).
"""))
    A(md('<h2><span style="color:blue">[202555404] [Isaac Mutuma]</span> </h2>'))
    A(md("""
| Step | What it does |
|---|---|
| **0. Setup** | Packages, GPU check, which parts to run |
| **1. Data** | Downloads UTKFace (pinned version, checked by SHA-256) and reads age and race from the file names |
| **2. CLIP features** | Pinned CLIP ViT-L/14 (OpenAI weights, QuickGELU checked), one 768-number embedding per image |
| **3. Folds** | 5 fixed folds (seed 0), balanced by race and age |
| **4. Evaluation code** | Detection rates, the White − Black gap, paired bootstrap, Holm |
| **5. Model code** | Age head (A), adversarial training (E), leakage probe, LEACE: *my implementation* |
| **6. Experiments** | A and E, each started with the three sentences |
| **7. Fine-tuning (F1-1, F1-4, F2)** | LoRA inside CLIP |
| **8. Re-check** | Main comparison with 3 seeds |
| **9. Follow-up questions** | Three questions that came from my results (9.3: where in CLIP the bias appears) |
| **10. Visual cases and demo** | A few clearly harmless ICCWD photos run through the models |
| **11. Limits and conclusion** | |
| **12. Experiment log** | Settings, epochs, updates, seeds and run time of every run |

**Ethics.** No images of children are generated. UTKFace photos are only read by the model and never shown. Real photos appear only in Step 10: a few clearly harmless ICCWD scenes, shown as they are; no captions or links.
"""))

    A(md("""
## Question, prediction and success criteria

**Why.** Datasets for text-to-image models are filtered with automatic child detectors, to lower the risk that the models learn to generate child sexual abuse material (Crețu et al., 2026). In my earlier audit on UTKFace, all five detectors I tested missed Black minors more often than White minors, by 4 to 10 points after adjusting for age, mostly among 10 to 17-year-olds, whom they judge older than they are.

**Question.** If I build a child/adult classifier on CLIP, can I make it as accurate for Black teenagers as for White teenagers by training only on top of CLIP's frozen features, or do I have to change CLIP itself?

**Prediction (written before the experiments).** An adversary on top of the frozen features (E) will have only a small effect, and a probe will still read race from its features, because the bias is already in CLIP's embeddings. Fine-tuning CLIP's last blocks with LoRA (F) has the best chance of narrowing the gap, but may overfit, because there are only 126 Black teenagers.

**Success criteria (fixed before any model was trained).** A model *closes* the gap if (1) its gap is significantly smaller than A's (one-sided paired bootstrap, Holm-corrected p < 0.05), (2) the gap is at most 3 points and clearly below A's, (3) adult false alarms stay at most 5% in every group and at most 2 points above A's, and (4) age error rises by at most 0.5 years (upper CI bound). It *narrows* the gap if (1), (3) and (4) hold.
"""))
    A(md("## 0. Setup\n\nRuns on Colab (Runtime → Change runtime type → T4) or in Jupyter on the course GPU server (set `GPU_ID` below to your assigned GPU). Section 10 needs the ICCWD photos on Google Drive, so it runs on Colab. Approximate time on a T4: data and CLIP features 25 min, A 5 min, E 90 min, follow-up 1 60 min, follow-up 2 25 min; fine-tuning (F1-1, F1-4, F2) about 2 hours on the server's RTX A5000, or it loads the saved F predictions when `RUN[\"F\"]` is False. Set a part to `False` to skip it."))
    A(code("""
import datetime
print("This code is written at " + str(datetime.datetime.now()))
"""))
    A(code("""
!pip -q install open_clip_torch==3.3.0
GPU_ID = None   # course GPU server: set to your assigned GPU id, e.g. "2"; on Colab leave None
import os
if GPU_ID is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)   # must be set before torch starts
RUN = {"A": True, "E": True, "followup1": True, "followup2": True, "F": False, "recheck": False,
       "followup3": False}
import os, json, math, hashlib, tarfile, re, time, urllib.request
from pathlib import Path, PurePosixPath
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
%matplotlib inline
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORK = Path("/content/work") if Path("/content").exists() else Path("work")
WORK.mkdir(parents=True, exist_ok=True)
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "",
      "| torch", torch.__version__)
"""))
    A(code(CONSTANTS))

    A(md("## 1. Data: UTKFace\n\n23,705 face photos. Age, gender and race come from the file name `age_gender_race_date.jpg`. Three files have malformed names and are skipped. The photos are never displayed."))
    A(code(r'''
def sha256(path, bs=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()

from huggingface_hub import hf_hub_download
tar_path = hf_hub_download(UTKFACE["hf_repo"], UTKFACE["file"], revision=UTKFACE["revision"], repo_type="dataset")
assert sha256(tar_path) == UTKFACE["sha256"], "UTKFace archive does not match the pinned version"
img_dir = WORK / "UTKFace"
if not img_dir.exists():
    with tarfile.open(tar_path) as t:
        safe = [m for m in t.getmembers() if m.isfile() and ".." not in PurePosixPath(m.name).parts]
        t.extractall(WORK, members=safe)
    cand = [p for p in WORK.iterdir() if p.is_dir() and p.name.lower().startswith("utkface") and p != img_dir]
    if cand: cand[0].rename(img_dir)

NAME = re.compile(r"^(\d+)_([01])_([0-4])_(\d+)")
rows = []
for p in sorted(img_dir.iterdir()):
    m = NAME.match(p.name)
    if m and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
        rows.append({"id": p.name.split(".")[0], "age": float(m[1]),
                     "group": C.UTKFACE_RACES[int(m[3])], "path": str(p)})
df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
assert len(df) == UTKFACE["n_records"], len(df)
df["band"] = pd.cut(df.age, [-1, 9, 17, 19, 39, 59, 200], labels=["0-9", "10-17", "18-19", "20-39", "40-59", "60+"])
display(pd.crosstab(df.group, df.band, margins=True))
'''))

    A(md("## 2. CLIP ViT-L/14 features\n\nThe OpenAI weights were trained with QuickGELU; loading them from a file makes `open_clip` silently use plain GELU, so I force QuickGELU and check that no GELU layer is left. Each image becomes one 768-number embedding (before L2 normalisation). The result is cached in `WORK`."))
    A(code(r'''
import open_clip
from PIL import Image

def load_clip(device):
    path = hf_hub_download(CLIP_WEIGHTS["hf_repo"], CLIP_WEIGHTS["file"], revision=CLIP_WEIGHTS["revision"])
    assert sha256(path) == CLIP_WEIGHTS["sha256"], "CLIP weights do not match the pinned version"
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=path, force_quick_gelu=True)
    names = [type(m).__name__ for m in model.modules()]
    assert names.count("QuickGELU") > 0 and names.count("GELU") == 0, "CLIP must use QuickGELU"
    return model.eval().to(device), preprocess

emb_file = WORK / "utkface_clip_vitl14.npz"
if emb_file.exists():
    z = np.load(emb_file, allow_pickle=False)
    assert list(z["ids"]) == list(df.id)
    X = z["emb"]
else:
    clip_model, preprocess = load_clip(DEVICE)
    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(df)
        def __getitem__(self, i): return preprocess(Image.open(df.path[i]).convert("RGB"))
    out, t0 = [], time.time()
    with torch.no_grad():
        for x in torch.utils.data.DataLoader(DS(), batch_size=128, num_workers=2):
            out.append(clip_model.encode_image(x.to(DEVICE)).float().cpu().numpy())
    X = np.concatenate(out).astype(np.float32)
    np.savez(emb_file, ids=df.id.values, emb=X)
    print(f"embedded {len(X)} images in {time.time() - t0:.0f} s")
ages, groups = df.age.values.astype(float), df.group.values
print("embeddings:", X.shape, "| sha256", sha256(emb_file)[:16])
'''))

    A(md("## 3. Folds\n\nStratified 5-fold split over race × age band (seed 0, ids sorted). Every image is predicted by a model that never saw it. For test fold *k*, settings are tuned on fold (*k*+1) mod 5 and the model is refit on the other four folds."))
    A(code(module_code("src/folds.py") + '''
fold_of = make_folds(list(df.id), list(groups), list(ages))
folds = np.array([fold_of[i] for i in df.id])
display(pd.crosstab(folds, groups))
'''))

    A(md("## 4. Evaluation code\n\nAge-stratified detection rate (TPR) of minors with the same age weights for every group, adult false-positive rate (FPR, ages 20–59), age error, and the White − Black gap. Confidence intervals: 2,000 bootstrap resamples within each group; every model is scored on the same resamples, so comparisons with model A are paired. Ported from my earlier audit."))
    A(code(module_code("src/metrics.py")))
    A(md("### Leakage probe (my implementation)\n\nAfter training, a new logistic regression tries to read race from a representation. Score: balanced accuracy over the 5 groups (chance 0.20)."))
    A(code(module_code("src/leakage.py")))

    A(md("## 5. Model code (my implementation)"))
    A(code(TRAIN))

    A(md("""
## 6. Experiments

### 6.1 Model A (baseline)
1. **I wonder:** does an age head trained on UTKFace's own ages, on frozen CLIP features, keep the zero-shot gap?
2. **I predict, because:** a gap still appears, probably smaller than zero-shot CLIP's 10 points, because the head learns from UTKFace's own age labels.
3. **To check, I will change** nothing (baseline) **and keep** the folds, the 8-setting grid and the selection by validation MAE fixed for every later model.
"""))
    A(code(f'''
GITHUB = "{GITHUB_RAW}/results"
def committed(rel):
    """My committed result for comparison (aggregate numbers only)."""
    with urllib.request.urlopen(f"{{GITHUB}}/{{rel}}") as r:
        return json.loads(r.read())

d_ref = None
def evaluate(name, pred):
    d = prepare(frame(df.id.values, groups, ages, pred))
    global d_ref, plan
    if d_ref is None:
        d_ref, plan = d, BootPlan(d)
    rep = model_report(d, plan)
    g = rep["tpr_gap_W_minus_B"]
    print(f"{{name}}: W-B gap {{g['est']:.3f}} [{{g['ci'][0]:.3f}}, {{g['ci'][1]:.3f}}], MAE {{rep['mae_overall']['est']:.2f}}")
    return d, rep

RESULTS, RECORDS = {{}}, {{}}
if RUN["A"]:
    predA, RECORDS["A"] = oof_predictions(X, ages, folds, GRID_A, FIXED_A, DEVICE, groups=groups, select="mae")
    dA, RESULTS["A"] = evaluate("A", predA)
    refA = [{{"chosen": r["chosen"], "epochs": r["epochs"], "inner_val_mae": r["inner_val_mae"]}} for r in RECORDS["A"]]
    print("committed result:", round(committed("models/A.json")["report"]["tpr_gap_W_minus_B"]["est"], 3))
'''))
    A(md("**After running** (committed run, `results/models/A.json`) — **I observed:** the White − Black gap is 11.0 points (95% CI 7.7 to 14.2), larger than zero-shot CLIP's 10.0. A catches 47.6% of Black and 80.9% of White 10–17-year-olds, and over-estimates Black minors' ages by 1.9 years more than White minors' (95% CI 1.2 to 2.7). Adult false alarms stay at most 0.6% in every group; MAE 4.44 years. **Interpretation:** training on UTKFace's own ages does not remove the gap; my prediction that it would shrink was wrong. The gap comes with over-estimation of Black minors' ages, which pushes teenagers over 18."))
    A(md("""
### 6.2 Model E (adversarial training on frozen CLIP features)
1. **I wonder:** can adversarial training on frozen CLIP features remove the race information that makes Black teenagers look older, and close the gap at 18?
2. **I predict, because:** a small effect; a probe will still recover race from its representation, because the bias is in CLIP's embeddings, which E cannot change.
3. **To check, I will change** only the training (a race adversary on the MLP's hidden layer, 5 adversary steps per MLP step, two targets, λ ∈ {0.1, 0.3, 1, 3}) **and keep** the folds, loss, budget, tuning rule and evaluation of A. Control: λ = 0 gives exactly model A.
"""))
    A(code('''
if RUN["E"]:
    probe_E = {}
    predE, RECORDS["E_inv"] = oof_predictions(X, ages, folds, GRID_E, FIXED_E, DEVICE, groups=groups,
                                              select="worst_group_v3", ref_A=refA, probe_out=probe_E)
    dE, RESULTS["E_inv"] = evaluate("E", predE)
    PAIRED = {"E_inv": paired_vs_A(dA, dE, plan)}
    LEAK_E = summarise(probe_E, groups)
    for n, r in LEAK_E["bal_acc"].items():
        print(f"race probe on {n:9s}: {r['est']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  (chance 0.20)")
'''))
    A(md("**After running** (committed run, `results/models/E_inv.json`) — **I observed:** the gap is 10.6 points (95% CI 7.1 to 13.7); the reduction against A is 0.4 points (95% CI −1.7 to 2.4, p = 0.36), at almost no cost (MAE +0.08 years). During training the adversary's race accuracy fell to chance (0.20–0.28 in four of five folds), yet a new linear probe reads race from E's hidden layer as well as from A's (balanced accuracy 0.816 vs 0.819; chance 0.20). The tuning rule picked the target 'make the adversary wrong' in every fold. **Interpretation:** the adversary was fooled, but race was not removed, as I predicted. **Not yet certain:** whether removing race would close the gap; this became follow-up 1."))
    A(md("### 6.3 Results so far"))
    A(code('''
def row(name, rep, paired=None):
    g, bg = rep["tpr_gap_W_minus_B"], rep["by_group"]
    out = {"model": name, "gap (points)": f"{100*g['est']:.1f} ({100*g['ci'][0]:.1f} to {100*g['ci'][1]:.1f})",
           "teen TPR Black / White": f"{bg['Black']['tpr_teen']['est']:.3f} / {bg['White']['tpr_teen']['est']:.3f}",
           "max adult FPR": f"{max(x['fpr_adult']['est'] for x in bg.values()):.3f}",
           "MAE": f"{rep['mae_overall']['est']:.2f}", "Black - White age error": f"{rep['age_err_gap_B_minus_W']['est']:+.2f}"}
    if paired:
        r = paired["gap_reduction_A_minus_M"]
        out["gap reduction vs A"] = f"{100*r['est']:+.1f} ({100*r['ci'][0]:+.1f} to {100*r['ci'][1]:+.1f}), p={r['p_one_sided']:.3f}"
    return out
display(pd.DataFrame([row(m, RESULTS[m], PAIRED.get(m) if m != "A" else None) for m in RESULTS]))
'''))
    A(md("### 6.4 Figures"))
    A(code('''
PRED = {"A": predA, "E": predE}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for name, pr in PRED.items():                                   # detection rate of minors by age
    for g, ls in [("White", "-"), ("Black", "--")]:
        m = groups == g
        tpr = [np.mean(pr[m & (ages == a)] < 18) for a in range(0, 18)]
        ax[0].plot(range(18), tpr, ls, label=f"{name} {g}")
ax[0].set(xlabel="true age", ylabel="share flagged as child", title="Minors flagged as children"); ax[0].legend(fontsize=8)
for g in ["White", "Black"]:                                     # predicted vs true age (model A)
    m = (groups == g) & (ages <= 30)
    mean_pred = [predA[m & (ages == a)].mean() for a in range(0, 31)]
    ax[1].plot(range(31), mean_pred, label=g)
ax[1].plot([0, 30], [0, 30], "k:", lw=1); ax[1].axhline(18, c="grey", lw=1)
ax[1].set(xlabel="true age", ylabel="mean predicted age (A)", title="Over-estimation near 18"); ax[1].legend()
for r in RECORDS["E_inv"]:                                       # E: adversary accuracy during training
    ax[2].plot(range(1, len(r["refit_info"]["adv_bal_acc_by_epoch"]) + 1), r["refit_info"]["adv_bal_acc_by_epoch"], label=f"fold {r['test_fold']}")
ax[2].axhline(0.2, ls="--", c="grey", lw=1); ax[2].set(xlabel="epoch", ylabel="adversary balanced accuracy", ylim=(0, 1), title="E: adversary is fooled")
ax[2].legend(fontsize=8); plt.tight_layout(); plt.show()
'''))

    A(md("""
## 7. Fine-tuning inside CLIP (F1-1, F1-4, F2)

### 7.1 F1-1 and F1-4
1. **I wonder:** if CLIP's last vision blocks are allowed to change, does the White − Black gap at 18 narrow, and how deep does the change need to go?
2. **I predict, because:** fine-tuning has the best chance of narrowing the gap, because it can change how CLIP encodes age itself, which E and LEACE could not; but it may overfit, since there are only 126 Black teenagers.
3. **To check, I will change** only which weights can learn: LoRA (rank 8, α = 16) on the query, key and value projections of the last block (F1-1) or the last 4 blocks (F1-4), with the rest of CLIP frozen, **and keep** the folds, L1 loss, age head (started from A's head, so step 0 equals A), the tuning rule (epoch ≤ 10 chosen on the validation fold) and the evaluation of A and E.

### 7.2 F2 (the better depth + E's adversary)
1. **I wonder:** does the adversary help once CLIP itself can change?
2. **I predict, because:** F2 narrows the gap more than F1, because once CLIP's blocks can change, the adversary can actually remove race from the features instead of only being fooled, as it was on frozen features. *(Recorded 25 Nov 2026, before any F run.)*
3. **To check, I will change** only one thing against F1: add E's adversary (make it wrong, λ = 1, 5 adversary steps per step), **and keep** everything else; F2 against F1 is the ablation of the adversary.

**Implementation (my implementation).** open_clip's attention keeps q, k and v in one packed weight; `LoRAAttention` computes the same attention plus the low-rank update. Because blocks 1–20 are frozen, each image's tokens entering block 21 are computed once and stored (fp16, ~12.5 GB), and training runs only the last 4 blocks.
"""))
    A(code(module_code("src/lora_clip.py")))
    A(code(module_code("src/train_f.py")))
    A(code('''
START, F_CONFIGS = 20, {"F1_1": [23], "F1_4": [20, 21, 22, 23]}
tok_file = WORK / "tokens_b20.f16"
if RUN["F"]:
    clip_model, preprocess = load_clip(DEVICE)
    visual = clip_model.visual.float()
    if not tok_file.exists():
        cache = np.memmap(tok_file, np.float16, "w+", shape=(len(df), 257, visual.transformer.width))
        with torch.no_grad():
            for b in range(0, len(df), 64):
                x = torch.stack([preprocess(Image.open(q).convert("RGB")) for q in df.path[b:b + 64]]).to(DEVICE)
                cache[b:b + 64] = tokens_before(visual, x, START).cpu().numpy().astype(np.float16)
        cache.flush()
    cache = np.memmap(tok_file, np.float16, "r", shape=(len(df), 257, visual.transformer.width))

def run_F(blocks, adversary, seed=C.SEED):
    """Same protocol as E: epochs chosen on the validation fold (rule v3), refit on 4 folds."""
    pred, recs = np.full(len(ages), np.nan), []
    for k in range(C.N_FOLDS):
        v = (k + 1) % C.N_FOLDS
        tr_in, va, tr_all, te = (folds != k) & (folds != v), folds == v, folds != k, folds == k
        rA = refA[k]
        i = grid_settings(GRID_A).index({kk: rA["chosen"][kk] for kk in GRID_A})
        sA = {**FIXED_A, **rA["chosen"]}
        st_in, st_all = Standardiser().fit(X[tr_in]), Standardiser().fit(X[tr_all])
        ref_fn, vpA, _ = fit_head(st_in(X[tr_in]), ages[tr_in], sA, rA["epochs"], seed + 1000 * k + i, DEVICE, st_in(X[va]))
        ref = group_fpr_mae(vpA[-1], ages[va], groups[va]); ref["tpr"] = group_tpr(vpA[-1], ages[va], groups[va])
        refit_fn, _, _ = fit_head(st_all(X[tr_all]), ages[tr_all], sA, rA["epochs"], seed + 1000 * k, DEVICE)
        r_in, r_va, r_all, r_te = (np.flatnonzero(m) for m in (tr_in, va, tr_all, te))
        _, vp, _ = fit_f(cache, r_in, ages[r_in], groups[r_in], ref_fn, st_in.mu, st_in.sd, visual, START, blocks,
                         FIXED_F, FIXED_F["max_epochs"], seed + 1000 * k, DEVICE, rows_val=r_va, adversary=adversary)
        stats = [selection_stats_v3(p_, ages[r_va], groups[r_va], ref) for p_ in vp]
        best = min(range(len(stats)), key=lambda e: rank_key(stats[e], "v3"))
        fn, _, info = fit_f(cache, r_all, ages[r_all], groups[r_all], refit_fn, st_all.mu, st_all.sd, visual, START,
                            blocks, FIXED_F, best + 1, seed + 1000 * k, DEVICE, adversary=adversary)
        pred[r_te] = fn(r_te)
        recs.append({"test_fold": k, "chosen": {"lora_blocks": blocks}, "epochs": best + 1, "batch": FIXED_F["batch_size"],
                     "inner_val_worst_group_tpr": stats[best]["worst_group_tpr"],
                     "test_mae": float(np.abs(pred[r_te] - ages[r_te]).mean()), "seconds": sum(info["epoch_seconds"])})
        print(f"fold {k}: {best + 1} epochs, test MAE {recs[-1]['test_mae']:.2f}")
    return pred, recs

if RUN["F"]:
    for name, blocks in F_CONFIGS.items():
        predF, RECORDS[name] = run_F(blocks, adversary=False)
        dF, RESULTS[name] = evaluate(name, predF)
        PAIRED[name] = paired_vs_A(dA, dF, plan)
    best_f1 = max(F_CONFIGS, key=lambda n: np.mean([r["inner_val_worst_group_tpr"] for r in RECORDS[n]]))
    predF2, RECORDS["F2"] = run_F(F_CONFIGS[best_f1], adversary=True)
    dF2, RESULTS["F2"] = evaluate("F2", predF2)
    PAIRED["F2"] = paired_vs_A(dA, dF2, plan)
else:
    # Reuse the F run made with scripts/run_model_f.py (same code, same folds and seeds):
    # its per-image predictions are in F_OOF_DIR and its per-fold settings in results/models/<F>.json.
    F_OOF_DIR = Path.home() / "fwork"
    for name in ["F1_1", "F1_4", "F2"]:
        f = F_OOF_DIR / f"oof__{name}.csv"
        if not f.exists():
            print(f"{name}: no saved predictions at {f}; set RUN['F'] = True to train it here"); continue
        o = pd.read_csv(f, dtype={"id": str}).set_index("id").loc[df.id.values]
        assert (o.fold.values == folds).all(), "saved predictions use different folds"
        dF, RESULTS[name] = evaluate(name, o.pred_age.values)
        PAIRED[name] = paired_vs_A(dA, dF, plan)
        RECORDS[name] = [{**r, "chosen": {"lora_blocks": r["lora_blocks"]}, "batch": FIXED_F["batch_size"]}
                         for r in committed(f"models/{name}.json")["folds"]]
display(pd.DataFrame([row(m, RESULTS[m], PAIRED.get(m)) for m in RESULTS]))
'''))
    A(md("""**After running** (`results/models/F1_1.json`, `F1_4.json`, `F2.json`; class GPU server, RTX A5000, 26 Nov 2026) — **I observed:** fine-tuning inside CLIP did not change the gap. F1-1: gap 10.7 points (95% CI 7.5 to 13.8), change against A −0.3 points (CI −1.8 to +1.2), MAE −0.08 years. F1-4: 11.5 points (CI 8.3 to 14.7), change −1.1 (CI −2.6 to +0.5), MAE −0.10 years. F2 used F1-1's depth (higher mean validation worst-group rate) plus the adversary: 10.3 points (CI 7.1 to 13.3), change +0.1 (CI −1.3 to +1.7), MAE −0.03 years; its adversary was pushed to chance (0.21–0.22 balanced accuracy in four of five folds). In all three, Black minors were still over-estimated by 1.8 to 1.9 years more than White minors, and 47.6–49.2% of Black teenagers were flagged as children against 79.9–82.7% of White teenagers. Unadjusted p = 0.36 (E), 0.66 (F1-1), 0.90 (F1-4), 0.45 (F2); after Holm correction over these four, p = 1.0 for every model: none reduces the gap.
**Interpretation:** letting CLIP's last blocks change improves age error a little but leaves the group difference where it was. My F2 prediction was wrong: even when CLIP itself can change, the adversary is fooled and the gap does not move. Why fine-tuning cannot reach the bias is follow-up 3 (Section 9.3).
**A bug I found and fixed.** My first F run (25 Nov) trained in half precision (fp16) without loss scaling. In the earlier LoRA blocks the gradients (about 5·10⁻⁵) fell below fp16's smallest normal number and lost their direction: their cosine with full-precision gradients was between −0.1 and 0.3 in blocks 20–21, against 0.96–1.00 in block 23. I added a gradient scaler (`torch.amp.GradScaler`) and reran all three conditions; the numbers above are from the rerun (its JSONs record commit c131300 with this one file changed, committed afterwards as e932332). The first run led to the same conclusion (F1-4 even widened the gap by 2.0 points).
**Not yet certain:** F is compared with A rebuilt on the server (gap 10.5 points, against 11.0 on Colab; GPU arithmetic is not bit-for-bit repeatable); only one LoRA rank, learning rate and adversary weight were tried, and F ran once (seed 0)."""))
    A(md("### 7.3 Figure: teenagers detected, by group and model\n\nOne point per group and model with its 95% CI. Models not run in this session are loaded from their committed result files."))
    A(code('''
FIG_RESULTS = {m: RESULTS[m] if m in RESULTS else committed(f"models/{m}.json")["report"]
               for m in ["A", "E_inv", "F1_1", "F1_4", "F2"]}
''' + TEEN_FIG))
    A(md("## 8. Re-check of the main comparison (3 seeds)\n\nThe key comparison (A against the best model) is repeated with seeds 1 and 2 (seed 0 is the main run): new random initialisation and batch order, same folds and rule. Set `BEST` to the best model after Section 7."))
    A(code('''
BEST = "E_inv"          # largest point reduction in Section 7 (E_inv +0.4 points, F2 +0.1); none significant
if RUN["recheck"]:
    rows_rc = []
    for seed in (0, 1, 2):
        pA_s, recA_s = oof_predictions(X, ages, folds, GRID_A, FIXED_A, DEVICE, seed=seed, groups=groups, select="mae")
        refA_s = [{"chosen": r["chosen"], "epochs": r["epochs"], "inner_val_mae": r["inner_val_mae"]} for r in recA_s]
        if BEST == "E_inv":
            pB_s, _ = oof_predictions(X, ages, folds, GRID_E, FIXED_E, DEVICE, seed=seed, groups=groups,
                                      select="worst_group_v3", ref_A=refA_s, probe_out={})
        else:
            refA_saved, refA = refA, refA_s
            pB_s, _ = run_F(F_CONFIGS.get(BEST, F_CONFIGS["F1_4"]), adversary=(BEST == "F2"), seed=seed)
            refA = refA_saved
        dA_s, dB_s = prepare(frame(df.id.values, groups, ages, pA_s)), prepare(frame(df.id.values, groups, ages, pB_s))
        r = paired_vs_A(dA_s, dB_s, plan)["gap_reduction_A_minus_M"]
        rows_rc.append({"seed": seed, "gap A": round(model_report(dA_s, plan)["tpr_gap_W_minus_B"]["est"], 3),
                        f"gap {BEST}": round(model_report(dB_s, plan)["tpr_gap_W_minus_B"]["est"], 3),
                        "reduction (95% CI)": f"{r['est']:+.3f} ({r['ci'][0]:+.3f} to {r['ci'][1]:+.3f})"})
    display(pd.DataFrame(rows_rc))
'''))

    A(md("""
## 9. Follow-up questions

### 9.1 Follow-up 1
**Observation it came from:** E's adversary fell to chance while a new probe still read race at 0.82.
1. **I wonder:** if the adversary is pushed much harder, ignoring the cost limits, is race removed, and does the gap close?
2. **I predict, because:** the probe drops well below A's 0.82 and age error passes +0.5 years, but the gap shrinks far less than the race signal and the Black − White over-estimation stays above 0, because what makes Black teenagers look older is age cues in CLIP, not a separate race signal.
3. **To check, I will change** only the adversary's target and weight (8 fixed settings from λ = 0 to 10, 60 epochs each, no tuning) **and keep** everything else fixed.
"""))
    A(code('''
FU1_BASE = {"hidden": 512, "lr": 3e-4, "weight_decay": 1e-4}
FU1_SETTINGS = {"control": (0.0, "prior"), "prior-0.1": (0.1, "prior"), "prior-0.3": (0.3, "prior"),
                "prior-1": (1.0, "prior"), "prior-3": (3.0, "prior"), "prior-10": (10.0, "prior"),
                "max_ce-1": (1.0, "max_ce"), "max_ce-10": (10.0, "max_ce")}

def run_fixed(s, epochs=60, seed=C.SEED):
    """Out-of-fold predictions and race-probe predictions for one fixed setting (no tuning)."""
    pred, probe = np.full(len(ages), np.nan), np.full(len(ages), None, dtype=object)
    for k in range(C.N_FOLDS):
        tr, te = folds != k, folds == k
        st = Standardiser().fit(X[tr])
        pf, _, _ = fit_head(st(X[tr]), ages[tr], s, epochs, seed + 1000 * k, DEVICE, None, groups[tr], ages[tr])
        pred[te] = pf(st(X[te]))
        probe[te] = fit_predict_probe(pf.encode(st(X[tr])), groups[tr], pf.encode(st(X[te])), seed)
    return pred, probe

if RUN["followup1"]:
    fu1 = []
    for name, (lam, target) in FU1_SETTINGS.items():
        pred, probe = run_fixed({**FIXED_E, **FU1_BASE, "adv_lambda": lam, "adv_target": target})
        d, rep = evaluate(f"follow-up 1, {name}", pred)
        fu1.append({**row(name, rep, paired_vs_A(dA, d, plan)),
                    "race probe": round(summarise({"model": probe}, groups)["bal_acc"]["model"]["est"], 3)})
    display(pd.DataFrame(fu1))
'''))
    A(md("**I observed:** in all 8 settings a new probe still reads race at 0.805–0.822 (threshold for 'race removed' 0.40), although the adversary itself was held near 0.50 or pushed to chance. Gaps range from 9.6 to 12.0 points and no setting narrows the gap significantly against A or against the λ = 0 control; the strongest settings raise MAE by 0.7–1.0 years and lower detection for both groups. **Interpretation:** adversarial training on a frozen head fools its adversary instead of removing race; my prediction that the probe would drop was wrong. **Not yet certain:** whether actually removing race would close the gap; this became follow-up 2."))
    A(md("""
### 9.2 Follow-up 2
**Observation it came from:** follow-up 1 could not remove race with an adversary at any strength.
1. **I wonder:** if linear race information is actually removed from CLIP's embeddings, does the White–Black age gap close, or does it stay wide?
2. **I predict, because:** the gap closes: once linear race information is removed, the gap falls to about 3 points or less with age error within +0.5 years, because linear race information is the real culprit.
3. **To check, I will change** only the input, with LEACE concept erasure fitted on each fold's training rows, **and keep** model A's exact procedure fixed.

LEACE (Belrose et al., 2023) edits the embeddings so that their covariance with race is exactly zero, which means no linear classifier can read race from them, while changing them as little as possible. My implementation follows the paper's closed form. Variant *band* first removes what each group's different age mix explains, so that age is not erased together with race.
"""))
    A(code(module_code("src/leace.py") + '''

def mlp_probe(Ztr, gtr, Zte, seed=0):
    """Non-linear race probe (one hidden layer of 256)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ztr)
    classes, codes = np.unique(np.asarray(gtr), return_inverse=True)
    clf = MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200, random_state=seed)
    clf.fit(sc.transform(Ztr), codes)
    return classes[clf.predict(sc.transform(Zte))]

if RUN["followup2"]:
    fu2 = []
    for name, cond in {"band": True, "plain": False}.items():
        pred, rec = oof_predictions(X, ages, folds, GRID_A, FIXED_A, DEVICE, groups=groups, select="mae",
                                    prep=prep_factory(cond))
        d, rep = evaluate(f"LEACE {name}", pred)
        lin, mlp = np.full(len(ages), None, dtype=object), np.full(len(ages), None, dtype=object)
        for k in range(C.N_FOLDS):
            tr, te = folds != k, folds == k
            le = LeaceStandardiser(cond).fit(X[tr], groups[tr], ages[tr])
            lin[te] = fit_predict_probe(le(X[tr]), groups[tr], le(X[te]), C.SEED)
            mlp[te] = mlp_probe(le(X[tr]), groups[tr], le(X[te]), C.SEED)
        pr = summarise({"linear": lin, "mlp": mlp}, groups)["bal_acc"]
        fu2.append({**row(f"LEACE {name}", rep, paired_vs_A(dA, d, plan)),
                    "linear probe": round(pr["linear"]["est"], 3), "MLP probe": round(pr["mlp"]["est"], 3)})
    display(pd.DataFrame(fu2))
'''))
    A(md("**I observed:** LEACE removed linear race information completely (linear probe 0.200, exactly chance) at no cost in age error (MAE −0.03 years). The gap stayed at 10.0 points (95% CI 6.7 to 13.2; reduction 0.9 points, CI −1.1 to 2.9) and Black minors were still over-estimated by 1.9 years more than White minors. **Interpretation:** the bias is not carried by a linear race direction in CLIP's embeddings; my prediction that the gap would close was wrong. **Not yet certain:** race is still readable non-linearly (MLP probe 0.81) and the age head partly rebuilds it (0.60), so this test cannot separate a non-linear race signal from age cues that differ between groups. Changing CLIP itself (F1, F2) is the next test."))

    A(md("""
### 9.3 Follow-up 3: where in CLIP does the bias appear?
**Observation it came from:** fine-tuning CLIP's last 1 or 4 blocks (F1-1, F1-4, F2) did not narrow the gap, and LEACE showed that the bias is not a linear race direction in the final embedding.
1. **I wonder:** at which block of CLIP does the over-estimation of Black children's age, and the White − Black gap, appear, and is it already there before the blocks that F fine-tunes (20–23)?
2. **I predict, because:** it is formed before F: by block 20 the over-estimation and the gap are already at least 75% of their final values, so F could only adjust a bias that was already present.
3. **To check, I will change** only the depth the age is read from (the CLS token or the mean patch token after each of CLIP's 24 blocks, and the raw patch embeddings) **and keep** frozen CLIP, the folds, one simple readout for every depth (ridge regression, strength chosen on the validation fold) and the metrics fixed.

Reading rule, fixed before the run: "formed before F" if at block 20 both the gap and the over-estimation are at least 75% of the final-embedding value with CIs above 0; "formed in F's blocks" if both are at most 25%. A weak readout (high MAE) at early depths is reported as a readout limit, not as "no bias". The age-matched over-estimation compares Black and White minors of the same true age, because a weak readout pulls every prediction toward the mean age.
"""))
    A(code(followup3_code() + '''

if RUN["followup3"]:
    clip_model, preprocess = load_clip(DEVICE)
    visual = clip_model.visual.float()
    feats = np.lib.format.open_memmap(WORK / "layers_cls_mean.npy", "w+", np.float16,
                                      (len(df), DEPTHS, 2, visual.transformer.width))
    with torch.no_grad():
        for b in range(0, len(df), 64):
            x = torch.stack([preprocess(Image.open(q).convert("RGB")) for q in df.path[b:b + 64]]).to(DEVICE)
            feats[b:b + 64] = layer_features(visual, x)[0].cpu().numpy().astype(np.float16)
    LAYERS, race_out = {}, {}
    reps = [("final_embedding", X)] + [(f"{p}_{l}", None) for l in range(DEPTHS) for p in POOLS
                                       if not (l == 0 and p == "cls")]
    for name, F in reps:
        if F is None:
            pool, l = name.rsplit("_", 1)
            F = np.asarray(feats[:, int(l), POOLS.index(pool)], np.float32)
        LAYERS[name], race_out[name] = evaluate_depth(name, F, df.id.values, ages, groups, folds, plan, DEVICE)
    race = summarise(race_out, groups)
    for name in LAYERS:
        LAYERS[name]["race_probe_bal_acc"] = race["bal_acc"][name]
    READING = reading(LAYERS, "cls")
else:
    saved = committed("followups/layers.json")          # run of scripts/followup_layers.py (same code)
    LAYERS, READING = saved["results"], saved["reading_cls"]
print("reading rule (CLS):", READING["case"])

''' + LAYER_FIG + '''

show = ["final_embedding"] + [f"cls_{l}" for l in (1, 4, 8, 12, 16, 20, 23, 24)]
display(pd.DataFrame([{"readout": n, "gap": round(LAYERS[n]["tpr_gap_W_minus_B"]["est"], 3),
                       "over-estimation (yrs)": round(LAYERS[n]["age_err_gap_B_minus_W"]["est"], 2),
                       "age-matched (yrs)": round(LAYERS[n]["age_err_gap_B_minus_W_age_matched"]["est"], 2),
                       "MAE": round(LAYERS[n]["mae_overall"]["est"], 2),
                       "race probe": round(LAYERS[n]["race_probe_bal_acc"]["est"], 3)} for n in show]))
'''))
    A(md("""**I observed** (`results/followups/layers.json`, RTX A5000, 12 minutes): the block-20 features are exactly the ones F starts from, and a linear readout of the final embedding reproduces A's pattern (gap 12.8 points, 95% CI 8.9 to 16.7; A 11.0). At block 20, where F1-4's LoRA starts, the gap is already 11.4 points (CI 7.8 to 14.9), 89% of the final value, and Black minors are over-estimated by 2.6 years more than White minors (CI 1.7 to 3.5), more than in the final embedding (1.5, CI 0.7 to 2.5). The reading rule gives "formed before F" for both the CLS and the mean-patch readout. Inside the network the bias is largest in the middle: the over-estimation peaks at 5.1 years at block 6 and the gap at 23.2 points at block 18. Along the CLS token, which becomes CLIP's embedding, the last four blocks reduce the over-estimation from 2.6 to 1.4 years. Race becomes steadily easier to read with depth (linear probe 0.37 on the raw patches, 0.79 at block 20, 0.82 at the end; chance 0.20).
**Interpretation:** my prediction was right. The bias is already in the tokens that enter the blocks F fine-tunes, so F could only adjust a bias it was handed, which explains why F1-1, F1-4 and F2 left the gap unchanged. CLIP's own last blocks already shrink the over-estimation somewhat; a fix that acts only there starts from a biased input.
**Not yet certain:** in the early blocks the readout is weak (CLS readout MAE 6.7 to 11.1 years up to block 16, raw patches 12.1, against 4.8 at the end) and flags few teenagers in any group (41% of White and 20% of Black teenagers at block 12), so the large early values say little. The bias appears from the first blocks, which suggests low-level cues such as skin tone or lighting, but this test cannot show that. The readout is linear; the age-matched over-estimation of the final embedding has a CI that includes 0 (0.7 years, CI −0.1 to 1.6); and the mean-patch readout does not show the late reduction (2.3 years at block 20, 3.0 at block 24). Every depth learns from the same UTKFace labels, so label error and model bias are still not separated."""))

    A(md("""
## 10. Real web photos (ICCWD): overall check and demo

ICCWD has no age or race labels, so it cannot measure the gap. It checks whether a model trained on face crops still finds children in ordinary web photos (child / no-child labels by the dataset authors). The final models are trained on all of UTKFace with A's most frequently chosen setting and the median chosen number of epochs. The images were downloaded to Google Drive with the authors' script (notebook 09); this section runs on Colab. The demo shows a few clearly harmless scenes as they are, with no captions or links.
"""))
    A(code('''
RUN_ICCWD = False
DEMO_ROWS = []          # row numbers of a few clearly harmless ICCWD scenes, picked by me
if RUN_ICCWD:
    from google.colab import drive
    drive.mount("/content/drive")
    IC = Path("/content/drive/MyDrive/robust_child_filtering_model/iccwd")
    ic = pd.read_csv(hf_hub_download("amcretu/iccwd", "image_caption_children_in_the_wild_dataset.csv", repo_type="dataset",
                                     revision="9f76b200c40a8a88c19dc17af8b842581f10f5f5"))
    ic["file"] = [IC / "images" / f"{i}.jpg" for i in range(len(ic))]
    ic = ic[ic.file.map(Path.exists) & ic.label.isin(["Final_Child", "Final_NoChild"])].reset_index(drop=True)
    clip_model, preprocess = load_clip(DEVICE)
    Xi = []
    with torch.no_grad():
        for b in range(0, len(ic), 64):
            x = torch.stack([preprocess(Image.open(f).convert("RGB")) for f in ic.file[b:b + 64]]).to(DEVICE)
            Xi.append(clip_model.encode_image(x).float().cpu().numpy())
    Xi = np.concatenate(Xi)
    from collections import Counter
    chosen = Counter(json.dumps(r["chosen"], sort_keys=True) for r in refA).most_common(1)[0][0]
    sA = {**FIXED_A, **json.loads(chosen)}
    epochs_A = int(np.median([r["epochs"] for r in refA]))
    st_full = Standardiser().fit(X)
    A_full, _, _ = fit_head(st_full(X), ages, sA, epochs_A, C.SEED, DEVICE)
    ic["pred_A"] = A_full(st_full(Xi))
    is_child = ic.label == "Final_Child"
    print(f"ICCWD (n={len(ic)}): A flags {np.mean(ic.pred_A[is_child] < 18):.3f} of photos with a child (TPR) "
          f"and {np.mean(ic.pred_A[~is_child] < 18):.3f} of photos without one (FPR)")
    for i in DEMO_ROWS:                                         # demo: a few clearly harmless scenes, shown as they are
        plt.figure(figsize=(4, 4)); plt.imshow(Image.open(ic.file[i]).convert("RGB")); plt.axis("off")
        plt.title(f"label: {'child' if is_child[i] else 'no child'} | A: age {ic.pred_A[i]:.0f} -> "
                  f"{'child' if ic.pred_A[i] < 18 else 'adult'}"); plt.show()
'''))
    A(md("""## 11. Failure cases, limits and conclusion

**Main result.** Across every method tried, the White − Black gap in flagging minors as children stayed between 9.5 and 11.5 points in every model (A 11.0). Removing race from the age head (E), erasing every linear trace of race from CLIP's embeddings (LEACE), and fine-tuning CLIP's last blocks with and without the adversary (F1-1, F1-4, F2) all failed to close it; none of the four planned tests is significant after Holm correction (all Holm p = 1.0).

**What this says about the cause.** E hid race from its adversary without removing it, and LEACE removed every linear trace of race from the embeddings, yet the gap did not move in either; in all eight trained models (A, the context models B, C and C2, E, F1-1, F1-4, F2) Black minors' ages were over-estimated by 1.6 to 2.3 years more than White minors'. The bias therefore does not look like a separate race signal that can be switched off. It is also not caused by having few examples: Asian (66) and Indian (102) teenagers are scarcer than Black teenagers (126), yet the eight models flagged 67% to 92% of them as children against 48% to 56% of Black teenagers, and over-estimated Asian minors' ages by at most 0.6 years. The problem is specific to Black faces being judged older, and it is present before any training on UTKFace: in my audit, zero-shot CLIP flagged 54.8% of Black teenagers (Asian 84.8%), and MiVOLO, an age model trained on other data, over-estimated Black minors' ages by 1.75 years (White 0.23, Asian −0.19). Either these pretrained models see Black teenagers as older than they are, or UTKFace's age labels for Black minors are lower than the faces look; these experiments cannot separate the two.

**Why fine-tuning could not fix it (follow-up 3).** Reading age from every block of CLIP shows that the gap and the over-estimation of Black minors are already present, at 89% and 168% of their final size, in the tokens that enter block 20, the first block F fine-tunes. F could only adjust a bias it was handed. The bias is largest in CLIP's middle blocks (over-estimation up to 5.1 years at block 6, gap up to 23 points at block 18), and CLIP's last blocks partly reduce it. A fix would have to act earlier in the network, or on the data and labels themselves.

**Failure cases.** The gap is almost entirely in the 10–17 band: Black teenagers are flagged as children about half as often as White teenagers in every model (teen gap 0.25–0.35). No model flagged more than 56% of Black teenagers, while every model flagged at least 77% of White teenagers.

**Limits.** One dataset with perceived-race labels from file names; one CLIP model; one LoRA rank, learning rate and adversary weight; F run once (seed 0) because each run takes about 2 hours; follow-up 3 uses a linear readout, which is weak in the early blocks; GPU results vary slightly between machines (A's gap 10.5 on the server, 11.0 on Colab). Equal detection rates were not forced by per-group thresholds, which would trade the gap for adult false alarms.

**What would help next.** More, carefully labelled photos of Black teenagers (which would also show whether the labels or the pretrained models are at fault), or training objectives that target the 10–17 band directly and act on CLIP's middle blocks, where follow-up 3 finds the bias largest, rather than removing race from the final features."""))
    A(md("## 12. Experiment log"))
    A(code('''
log_rows = []
for name, recs in RECORDS.items():
    for r in recs:
        log_rows.append({"run": name, "test fold": r["test_fold"], "chosen setting": r["chosen"], "epochs": r["epochs"],
                         "updates": r["epochs"] * math.ceil((folds != r["test_fold"]).sum() / r.get("batch", 256)),
                         "batch": r.get("batch", 256), "seed": f"{C.SEED} + 1000 x fold", "seconds": r["seconds"], "device": DEVICE})
display(pd.DataFrame(log_rows))
'''))
    A(md("""
## 13. Sources and AI use

- Crețu et al., *Evaluating concept filtering defenses against child sexual abuse material generation by text-to-image models*, IEEE S&P 2026 (arXiv 2512.05707): motivation and the detectors' age-18 rule.
- Kireev, Crețu et al., *A manually annotated image-caption dataset for detecting children in the wild* (ICCWD), arXiv 2506.10117, and their download script (spring-epfl/iccwd): the real-photo check.
- Zhang, Song and Qi, *Age progression/regression by conditional adversarial autoencoder*, CVPR 2017: UTKFace.
- Radford et al., *Learning transferable visual models from natural language supervision* (CLIP), ICML 2021; Dosovitskiy et al., *An image is worth 16x16 words* (ViT), ICLR 2021: the backbone.
- Goodfellow et al., *Generative adversarial nets*, NeurIPS 2014; Ganin and Lempitsky, ICML 2015; Zhang, Lemoine and Mitchell, AIES 2018; Isola et al., pix2pix, CVPR 2017: adversarial training and the L1 + adversarial loss.
- Elazar and Goldberg, *Adversarial removal of demographic attributes from text data*, EMNLP 2018: why a fooled adversary does not mean race is gone.
- Belrose et al., *LEACE: perfect linear concept erasure in closed form*, NeurIPS 2023: follow-up 2.
- Hu et al., *LoRA: low-rank adaptation of large language models*, ICLR 2022; Zanella and Ben Ayed, *Low-rank few-shot adaptation of vision-language models*, CVPRW 2024: model F.
- Sagawa et al., *Distributionally robust neural networks for group shifts*, ICLR 2020: group DRO (context models C and C2).
- Gokhale et al., arXiv 2104.01781 (2021): closest earlier work (adversarial debiasing of age estimates on UTKFace).
- Castleman and Korolova, *Adultification bias in LLMs and text-to-image models*, FAccT 2025: the same bias on the generation side.

**AI use:** an AI coding assistant (Claude) helped draft code, tests and text; I chose the question, wrote down my predictions before each run, ran the experiments and checked the results myself. *(Draft: confirm or edit.)*
"""))
    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    for c in nb["cells"]:
        lines = c["source"].split("\n")
        c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    return nb


if __name__ == "__main__":
    nb = build()
    print(OUT, len(nb["cells"]), "cells")
