"""Age-regression heads on frozen CLIP embeddings and the out-of-fold (OOF) loop.

Model A (plain): L1 loss on true age, no group information anywhere.

Protocol (docs/SUCCESS_CRITERIA.md §1, src/folds.py):
  for each outer test fold k:
    tune: train on folds != k, (k+1)%5; pick the grid setting and epoch count with the
          lowest MAE on inner validation fold (k+1)%5
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
from src.folds import inner_val_fold

# Model A grid (fixed before seeing any real result). hidden=0 means a linear head.
GRID_A = {"hidden": [0, 512], "lr": [1e-3, 3e-4], "weight_decay": [1e-4, 1e-2]}
FIXED_A = {"max_epochs": 60, "batch_size": 256, "dropout": 0.1, "loss": "l1"}


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


def fit_head(Xtr, ytr, s, epochs, seed, device, Xval=None, yval=None):
    """Train one head on the standardised target.

    Returns (predict_fn, per-epoch validation MAE in years, or None). predict_fn maps
    standardised inputs to ages in years.
    """
    import torch
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

    Xv = torch.from_numpy(Xval).to(device) if Xval is not None else None
    hist = [] if Xval is not None else None
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(X), generator=g).split(s["batch_size"]):
            b = b.to(device)
            opt.zero_grad()
            loss_fn(model(X[b]).squeeze(1), y[b]).backward()
            opt.step()
        if Xv is not None:
            hist.append(float(np.abs(predict_fn(Xv) - yval).mean()))
    return predict_fn, hist


def predict(model, X):
    """Raw head output as a numpy array (X: numpy or tensor)."""
    import torch
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        Xt = X if torch.is_tensor(X) else torch.from_numpy(X)
        return model(Xt.to(dev)).squeeze(1).cpu().numpy().astype(float)


def oof_predictions(X, y, folds, grid=None, fixed=None, device="cpu", seed=None, log=print):
    """Out-of-fold predictions for every row, plus a per-fold tuning record.

    X: (n, d) raw embeddings; y: true ages; folds: int array of fold ids (0..K-1).
    """
    grid, fixed = grid or GRID_A, fixed or FIXED_A
    seed = C.SEED if seed is None else seed
    folds = np.asarray(folds)
    K = int(folds.max()) + 1
    pred, record = np.full(len(y), np.nan), []
    for k in range(K):
        t0, v = time.time(), inner_val_fold(k, K)
        tr_in, va, tr_all, te = (folds != k) & (folds != v), folds == v, folds != k, folds == k
        st = Standardiser().fit(X[tr_in])
        trials = []
        for i, s in enumerate(grid_settings(grid)):
            s = {**fixed, **s}
            _, hist = fit_head(st(X[tr_in]), y[tr_in], s, s["max_epochs"], seed + 1000 * k + i,
                               device, st(X[va]), y[va])
            best = int(np.argmin(hist))
            trials.append({"setting": {k2: s[k2] for k2 in grid}, "best_epoch": best + 1,
                           "val_mae": hist[best]})
        win = min(trials, key=lambda t: t["val_mae"])
        s = {**fixed, **win["setting"]}
        st = Standardiser().fit(X[tr_all])
        predict_fn, _ = fit_head(st(X[tr_all]), y[tr_all], s, win["best_epoch"],
                                 seed + 1000 * k, device)
        pred[te] = predict_fn(st(X[te]))
        test_mae = float(np.abs(pred[te] - y[te]).mean())
        record.append({"test_fold": k, "inner_val_fold": v, "chosen": win["setting"],
                       "epochs": win["best_epoch"], "inner_val_mae": win["val_mae"],
                       "test_mae": test_mae, "trials": trials,
                       "seconds": round(time.time() - t0, 1)})
        log(f"fold {k}: chose {win['setting']} x {win['best_epoch']} ep, "
            f"inner-val MAE {win['val_mae']:.2f}, test MAE {test_mae:.2f}")
    assert not np.isnan(pred).any()
    return pred, record
