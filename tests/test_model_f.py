"""Local tests for model F (LoRA in CLIP's last blocks) on a small random ViT; no downloads.

  python tests/test_model_f.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import train as T  # noqa: E402
from src.lora_clip import finish, tokens_before  # noqa: E402
from src.train_f import FIXED_F, fit_f, predict_f  # noqa: E402

GROUPS = ["White", "Black", "Asian", "Indian", "Others"]


def setup(n=48, seed=0):
    import open_clip
    torch.manual_seed(seed)
    visual = open_clip.create_model("ViT-B-32", pretrained=None, force_quick_gelu=True).visual.eval()
    rng = np.random.default_rng(seed)
    imgs = torch.randn(n, 3, 224, 224)
    with torch.no_grad():
        emb = visual(imgs).numpy().astype(np.float32)
        cache = tokens_before(visual, imgs, 9).numpy().astype(np.float16)
    ages = rng.integers(0, 80, n).astype(float)
    groups = rng.choice(GROUPS, n)
    return visual, emb, cache, ages, groups


def test_step0_equals_A_and_training_moves_only_lora_and_head():
    visual, emb, cache, ages, groups = setup()
    rows = np.arange(len(ages))
    st = T.Standardiser().fit(emb)
    sA = {**T.FIXED_A, "hidden": 32, "lr": 3e-3, "weight_decay": 1e-4}
    A_fn, _, _ = T.fit_head(st(emb), ages, sA, 5, 0, "cpu")
    predA = A_fn(st(emb))
    cfg = {**FIXED_F, "batch_size": 16, "lr": 1e-3}
    # epochs = 0: the untrained F must give A's predictions
    F0, _, _ = fit_f(cache, rows, ages, groups, A_fn, st.mu, st.sd, visual, 9, [9, 10, 11], cfg, 0, 0, "cpu",
                     log=lambda *_: None)
    d0 = np.abs(F0(rows) - predA).max()
    assert d0 < 0.05, d0          # fp16 token cache: small rounding only
    before = {k: v.clone() for k, v in visual.state_dict().items()}
    F3, vp, info = fit_f(cache, rows, ages, groups, A_fn, st.mu, st.sd, visual, 9, [9, 10, 11], cfg, 3, 0, "cpu",
                         rows_val=rows, log=lambda *_: None)
    mae0, mae3 = np.abs(predA - ages).mean(), np.abs(F3(rows) - ages).mean()
    print(f"  step-0 max |F - A| = {d0:.4f} yrs; train MAE A {mae0:.2f} -> F after 3 epochs {mae3:.2f}; "
          f"LoRA params {info['n_lora_params']}")
    assert mae3 < mae0
    assert vp.shape == (3, len(rows))
    for k, v in visual.state_dict().items():           # the original CLIP is untouched
        assert torch.equal(v, before[k]), k
    m = F3.model
    assert all(not p.requires_grad for n, p in m.visual.named_parameters() if "lora_" not in n)
    assert any(p.abs().sum() > 0 for p in [m.visual.transformer.resblocks[11].attn.lora_B])


def test_f2_adversary_runs():
    visual, emb, cache, ages, groups = setup(seed=1)
    rows = np.arange(len(ages))
    st = T.Standardiser().fit(emb)
    A_fn, _, _ = T.fit_head(st(emb), ages, {**T.FIXED_A, "hidden": 32, "lr": 3e-3, "weight_decay": 1e-4}, 3, 0, "cpu")
    F2, _, info = fit_f(cache, rows, ages, groups, A_fn, st.mu, st.sd, visual, 9, [11],
                        {**FIXED_F, "batch_size": 16}, 2, 0, "cpu", adversary=True, log=lambda *_: None)
    assert len(info["adv_bal_acc_by_epoch"]) == 2 and np.isfinite(F2(rows)).all()


def test_finish_matches_full_forward():
    visual, emb, cache, ages, groups = setup(n=4, seed=2)
    with torch.no_grad():
        e = finish(visual, torch.from_numpy(cache.astype(np.float32)), 9).numpy()
    assert np.abs(e - emb).max() < 0.05     # fp16 storage of the tokens


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
