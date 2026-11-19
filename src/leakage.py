"""Leakage probe for model E (and F2): how much race can a NEW classifier read from z?

An adversary can be fooled while race stays in z in a form it misses. After each refit the
encoder is frozen and a fresh multinomial logistic regression (standardised z, C = 1,
fitted on the training folds) predicts race on the test fold. Pooled over the 5 test folds
this gives out-of-fold race predictions for every image.

Score = balanced accuracy over the 5 races (mean per-race recall); chance = 0.20 whatever
the group sizes (plain accuracy would reward always guessing White). Compared on the same
images: z of the model, z of model A (A's hidden layer) and the raw CLIP inputs.
CIs: 2,000 resamples within each true race (seed 0), shared by all representations, so the
differences are paired.
"""

import numpy as np

from configs import config as C


def fit_predict_probe(Ztr, gtr, Zte, seed=0):
    """Race predictions on Zte from a logistic regression fitted on (Ztr, gtr)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ztr)
    clf = LogisticRegression(C=1.0, max_iter=3000, random_state=seed)
    clf.fit(sc.transform(Ztr), np.asarray(gtr))
    return clf.predict(sc.transform(Zte))


def balanced_accuracy(pred, true, classes=None):
    """Mean per-class recall."""
    pred, true = np.asarray(pred), np.asarray(true)
    classes = sorted(set(true.tolist())) if classes is None else classes
    return float(np.mean([np.mean(pred[true == c] == c) for c in classes]))


def summarise(probe_out, groups, B=None, seed=None):
    """Pooled balanced accuracy per representation with paired bootstrap CIs."""
    B = C.N_BOOTSTRAP if B is None else B
    seed = C.SEED if seed is None else seed
    groups = np.asarray(groups)
    classes = sorted(set(groups.tolist()))
    names = list(probe_out)
    for n in names:
        if any(v is None for v in probe_out[n]):
            raise ValueError(f"probe '{n}' is missing predictions")
    rng = np.random.default_rng(seed)
    # per class: (B, n_reps) resampled recall
    rec_boot = np.zeros((len(classes), B, len(names)))
    point = {n: {} for n in names}
    for ci, c in enumerate(classes):
        idx = np.flatnonzero(groups == c)
        hit = np.stack([(np.asarray(probe_out[n])[idx] == c).astype(float) for n in names], 1)
        for j, n in enumerate(names):
            point[n][c] = float(hit[:, j].mean())
        W = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=B)
        rec_boot[ci] = (W @ hit) / len(idx)
    bal_boot = rec_boot.mean(0)                      # (B, n_reps)
    lo, hi = 100 * C.ALPHA / 2, 100 * (1 - C.ALPHA / 2)
    out = {"chance": 1 / len(classes), "classes": classes, "n": int(len(groups)),
           "majority_share": float(max(np.mean(groups == c) for c in classes)),
           "bal_acc": {}, "recall_by_race": point, "diff": {}}
    for j, n in enumerate(names):
        est = float(np.mean(list(point[n].values())))
        out["bal_acc"][n] = {"est": est, "ci": np.percentile(bal_boot[:, j], [lo, hi]).tolist()}
    if "model" in names:
        jm = names.index("model")
        for n in names:
            if n == "model":
                continue
            d = bal_boot[:, jm] - bal_boot[:, names.index(n)]
            out["diff"][f"model_minus_{n}"] = {
                "est": out["bal_acc"]["model"]["est"] - out["bal_acc"][n]["est"],
                "ci": np.percentile(d, [lo, hi]).tolist()}
    return out
