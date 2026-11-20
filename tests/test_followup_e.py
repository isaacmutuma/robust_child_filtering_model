"""Local tests for the model E follow-up sweep (synthetic data only).

  python tests/test_followup_e.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import followup_e_force_race as FU  # noqa: E402
from src import train as T  # noqa: E402
from test_model_e import synth  # noqa: E402


def test_control_equals_plain_training():
    """The lambda 0 control is plain training with the same budget (no adversary effect)."""
    X, y, g = synth(1500, seed=1)
    folds = np.arange(len(y)) % 5
    saved = FU.EPOCHS
    FU.EPOCHS = 4
    try:
        s = {**T.FIXED_E, **FU.BASE, "hidden": 32, **FU.SETTINGS["control"]}
        pred, probe, per_fold = FU.run_setting(X, y, g, folds, s, "cpu", 0, log=lambda *_: None)
        tr, te = folds != 0, folds == 0
        st = T.Standardiser().fit(X[tr])
        pf, _, _ = T.fit_head(st(X[tr]), y[tr], {**T.FIXED_A, **FU.BASE, "hidden": 32}, 4, 0,
                              "cpu")
        assert np.array_equal(pred[te], pf(st(X[te])))
        assert all(p is not None for p in probe) and len(per_fold) == 5
    finally:
        FU.EPOCHS = saved


def test_sweep_removes_race_with_pressure():
    """Planted race: strong base-rate pressure lowers the probe vs the control."""
    X, y, g = synth(5000, seed=2, race_strength=3.0)
    folds = np.arange(len(y)) % 5
    saved = FU.EPOCHS
    FU.EPOCHS = 30
    try:
        res = {}
        for name in ("control", "prior-0.3"):
            s = {**T.FIXED_E, **FU.BASE, "hidden": 32, "lr": 1e-3, **FU.SETTINGS[name]}
            pred, probe, _ = FU.run_setting(X, y, g, folds, s, "cpu", 0, log=lambda *_: None)
            res[name] = (FU.LK.balanced_accuracy(probe, g), float(np.abs(pred - y).mean()))
        print("  probe/MAE:", {k: (round(a, 3), round(b, 2)) for k, (a, b) in res.items()})
        assert res["prior-0.3"][0] < res["control"][0] - 0.15
    finally:
        FU.EPOCHS = saved


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
