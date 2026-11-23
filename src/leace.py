"""LEACE: least-squares concept erasure (Belrose et al., NeurIPS 2023), for follow-up 2.

Given inputs X (n, d) and a concept Z (n, k; here one-hot race), LEACE finds the affine map
x -> x - W⁺ P W (x - μ) that makes the cross-covariance between the erased inputs and Z
exactly zero while moving x as little as possible (in the Σ-whitened norm). With zero
cross-covariance, the class means of the erased inputs coincide, so no linear classifier
can predict race better than a constant one ("linear guardedness"). Non-linear read-out can
remain; follow-up 2 measures it with an MLP probe.

  W  = Σ_XX^{-1/2}               whitening (eigenvalues below eps·max are dropped)
  P  = orthogonal projection onto the column space of W Σ_XZ
  W⁺ = Σ_XX^{1/2}                (on the kept eigen-space)

Variant 'band' (conditional): Z is first residualised on the fold age-band dummies, so only
race information beyond what the groups' different age mixes explain is erased (race and
age are confounded in UTKFace; plain LEACE could remove some age signal with them).
"""

import numpy as np

from src.folds import age_band
from src.train import Standardiser


def _onehot(labels, classes):
    labels = np.asarray(labels)
    return (labels[:, None] == np.asarray(classes)[None, :]).astype(np.float64)


class Leace:
    """Fit on training rows only; call to erase."""

    def fit(self, X, groups, ages=None, conditional=False, eps=1e-6):
        X = np.asarray(X, np.float64)
        self.classes = sorted(set(np.asarray(groups).tolist()))
        Z = _onehot(groups, self.classes)
        if conditional:   # race beyond the age-band mix
            bands = np.array([age_band(a) for a in ages])
            B = _onehot(bands, sorted(set(bands.tolist())))
            Z = Z - B @ np.linalg.lstsq(B, Z, rcond=None)[0]
        self.mu = X.mean(0)
        Xc, Zc = X - self.mu, Z - Z.mean(0)
        n = len(X)
        S = Xc.T @ Xc / n
        Sxz = Xc.T @ Zc / n
        ev, U = np.linalg.eigh(S)
        keep = ev > eps * ev.max()
        U, ev = U[:, keep], ev[keep]
        W = (U / np.sqrt(ev)) @ U.T             # Σ^{-1/2}
        W_pinv = (U * np.sqrt(ev)) @ U.T        # Σ^{1/2}
        A = W @ Sxz
        u, sv, _ = np.linalg.svd(A, full_matrices=False)
        u = u[:, sv > 1e-10 * max(sv.max(), 1e-300)]
        P = u @ u.T
        self.E = W_pinv @ P @ W                 # x' = x - E (x - mu)
        self.rank = int(u.shape[1])
        return self

    def __call__(self, X):
        X = np.asarray(X, np.float64)
        return (X - (X - self.mu) @ self.E.T).astype(np.float32)


class LeaceStandardiser:
    """Standardiser (L2 + z-score, training rows) followed by LEACE fitted on the same rows."""

    def __init__(self, conditional=False):
        self.conditional = conditional

    def fit(self, X, groups, ages):
        self.st = Standardiser().fit(X)
        self.leace = Leace().fit(self.st(X), groups, ages, conditional=self.conditional)
        return self

    def __call__(self, X):
        return self.leace(self.st(X))


def prep_factory(conditional=False):
    """For src.train.oof_predictions(prep=...)."""
    return lambda X, groups, ages: LeaceStandardiser(conditional).fit(X, groups, ages)
