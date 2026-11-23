"""Local tests for LEACE and follow-up 2 (synthetic data only).

  python tests/test_leace.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]
from src import leakage as LK  # noqa: E402
from src import train as T  # noqa: E402
from src.leace import Leace, LeaceStandardiser, prep_factory  # noqa: E402
from test_model_e import GROUPS, synth  # noqa: E402


def test_zero_cross_covariance_and_minimal_change():
    rng = np.random.default_rng(0)
    g = rng.choice(GROUPS, 3000)
    X = rng.normal(size=(3000, 20)) + 3 * (g[:, None] == np.array(GROUPS)[None, :]) @ rng.normal(size=(5, 20))
    le = Leace().fit(X, g)
    Xe = le(X)
    Z = (g[:, None] == np.array(le.classes)[None, :]).astype(float)
    xcov = np.abs((Xe - Xe.mean(0)).T @ (Z - Z.mean(0)) / len(X)).max()
    assert xcov < 1e-6, xcov
    assert le.rank == 4                                  # k - 1 directions for 5 classes
    for c in le.classes:                                 # class means coincide
        assert np.abs(Xe[g == c].mean(0) - Xe.mean(0)).max() < 1e-4
    # the change is confined to a (k - 1)-dimensional subspace (an oblique projection)
    D = X - Xe
    assert np.linalg.matrix_rank(D, tol=1e-6 * np.abs(D).max()) == 4
    assert np.abs(le(Xe) - Xe).max() < 1e-4             # erasing twice changes nothing


def test_planted_race_removed_age_kept():
    X, y, g = synth(6000, seed=2, race_strength=3.0)
    tr, te = np.arange(len(y)) < 4500, np.arange(len(y)) >= 4500
    st = T.Standardiser().fit(X[tr])
    le = LeaceStandardiser().fit(X[tr], g[tr], y[tr])
    p_plain = LK.fit_predict_probe(st(X[tr]), g[tr], st(X[te]))
    p_erased = LK.fit_predict_probe(le(X[tr]), g[tr], le(X[te]))
    a_plain, a_erased = LK.balanced_accuracy(p_plain, g[te]), LK.balanced_accuracy(p_erased, g[te])
    base = {**T.FIXED_A, "hidden": 32, "lr": 1e-3, "weight_decay": 1e-4}
    mae = {}
    for name, f in (("plain", st), ("erased", le)):
        pf, _, _ = T.fit_head(f(X[tr]), y[tr], base, 25, 0, "cpu")
        mae[name] = float(np.abs(pf(f(X[te])) - y[te]).mean())
    print(f"  linear probe {a_plain:.3f} -> {a_erased:.3f}; MAE {mae['plain']:.2f} -> {mae['erased']:.2f}")
    assert a_plain > 0.95 and a_erased < 0.30
    assert mae["erased"] < mae["plain"] + 1.0


def test_band_variant_keeps_age_when_race_and_age_confounded():
    """Race groups with different age mixes: plain LEACE also erases age; 'band' less so."""
    rng = np.random.default_rng(3)
    n = 6000
    g = rng.choice(GROUPS, n)
    age = np.where(np.isin(g, ["White", "Asian"]), rng.integers(0, 40, n), rng.integers(20, 80, n)).astype(float)
    X = (np.outer(age - 40, rng.normal(size=16)) / 40 + rng.normal(0, 0.3, (n, 16)) + 5).astype(np.float32)
    tr, te = np.arange(n) < 4500, np.arange(n) >= 4500
    base = {**T.FIXED_A, "hidden": 32, "lr": 1e-3, "weight_decay": 1e-4}
    mae = {}
    for name, cond in (("plain", False), ("band", True)):
        le = LeaceStandardiser(cond).fit(X[tr], g[tr], age[tr])
        pf, _, _ = T.fit_head(le(X[tr]), age[tr], base, 25, 0, "cpu")
        mae[name] = float(np.abs(pf(le(X[te])) - age[te]).mean())
    print(f"  confounded: MAE plain-LEACE {mae['plain']:.2f}, band-LEACE {mae['band']:.2f}")
    assert mae["band"] < mae["plain"]


def test_oof_with_prep_and_default_unchanged():
    X, y, g = synth(3000, seed=4)
    folds = np.arange(len(y)) % 5
    grid = {"hidden": [32], "lr": [3e-3], "weight_decay": [1e-4]}
    fixed = {**T.FIXED_A, "max_epochs": 4}
    p0, _ = T.oof_predictions(X, y, folds, grid, fixed, log=lambda *_: None)
    p1, _ = T.oof_predictions(X, y, folds, grid, fixed, log=lambda *_: None,
                              prep=lambda X_, g_, a_: T.Standardiser().fit(X_))
    assert np.array_equal(p0, p1)                        # hook with Standardiser = default
    p2, _ = T.oof_predictions(X, y, folds, grid, fixed, groups=g, log=lambda *_: None,
                              prep=prep_factory(True))
    assert not np.isnan(p2).any()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
