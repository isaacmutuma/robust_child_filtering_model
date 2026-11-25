"""Run model F (LoRA inside CLIP's last blocks) on UTKFace: F1-1, F1-4 and F2.

  python scripts/run_model_f.py --work /path/with/20GB --configs F1_1,F1_4,F2
  python scripts/run_model_f.py --work ... --configs F1_4 --folds 0      # time one fold first

Self-contained (course GPU server or Colab): downloads the pinned UTKFace archive and CLIP
ViT-L/14 weights (sha256-checked), runs every image once through CLIP's first 20 blocks and
stores those tokens (fp16 memmap, ~12.5 GB) plus the exact frozen embeddings, rebuilds model
A per fold (F starts from A's head), then trains each F condition with the same protocol as
E: for test fold k, epochs are chosen on validation fold (k+1)%5 with tuning rule v3 against
A's reference, then refit on the 4 training folds and predict fold k.
F2 uses the F1 depth with the higher mean validation worst-group TPR (decided from the F1
records, never from test folds) plus E's adversary.
Outputs: WORK/oof__<config>.csv (per image, never in git) and results/models/<config>.json
(aggregate, with config and git commit).
"""

import argparse
import json
import tarfile
import time
from pathlib import Path, PurePosixPath

import numpy as np

from _common import REPO, sha256
from configs import config as C
from src import metrics as M
from src import train as T
from src.folds import make_folds
from src.provenance import write_result
from src.train_f import FIXED_F, fit_f

START = 20                                    # blocks 0..19 frozen and stored
CONFIGS = {"F1_1": {"blocks": [23], "adversary": False},
           "F1_4": {"blocks": [20, 21, 22, 23], "adversary": False},
           "F2": {"blocks": None, "adversary": True}}     # blocks = better F1 depth


def get_utkface(work):
    from huggingface_hub import hf_hub_download
    from src.data.utkface import list_utkface
    S = C.SOURCES["utkface"]
    img_dir = work / "UTKFace"
    if not img_dir.exists():
        tar = hf_hub_download(S["hf_repo"], S["file"], revision=S["revision"], repo_type="dataset")
        assert sha256(tar) == S["sha256"], "UTKFace archive does not match the pinned version"
        with tarfile.open(tar) as t:
            t.extractall(work, members=[m for m in t.getmembers()
                                        if m.isfile() and ".." not in PurePosixPath(m.name).parts])
        cand = [p for p in work.iterdir() if p.is_dir() and p.name.lower().startswith("utkface") and p != img_dir]
        if cand:
            cand[0].rename(img_dir)
    rows, _ = list_utkface(img_dir)
    rows = sorted(rows, key=lambda r: r["id"])
    assert len(rows) == S["n_records"], len(rows)
    return rows


def build_cache(rows, work, device):
    """tokens entering block START (fp16 memmap) + exact frozen embeddings (fp32)."""
    import torch
    from PIL import Image
    from src.clip_model import load_clip
    from src.lora_clip import finish, tokens_before
    tok_f, emb_f = work / f"tokens_b{START}.f16", work / "emb_fp32.npy"
    model, preprocess, _, winfo = load_clip(device)
    visual = model.visual.float()
    n, L, d = len(rows), 257, visual.transformer.width
    if tok_f.exists() and emb_f.exists():
        return np.memmap(tok_f, np.float16, "r", shape=(n, L, d)), np.load(emb_f), visual, winfo
    cache = np.memmap(tok_f, np.float16, "w+", shape=(n, L, d))
    emb = np.empty((n, 768), np.float32)
    t0, bs = time.time(), 64
    with torch.no_grad():
        for b in range(0, n, bs):
            x = torch.stack([preprocess(Image.open(r["path"]).convert("RGB")) for r in rows[b:b + bs]]).to(device)
            tok = tokens_before(visual, x, START)
            cache[b:b + bs] = tok.cpu().numpy().astype(np.float16)
            emb[b:b + bs] = finish(visual, tok, START).float().cpu().numpy()
            print(f"\rcache {min(b + bs, n)}/{n} ({time.time() - t0:.0f} s)", end="", flush=True)
    print()
    cache.flush()
    np.save(emb_f, emb)
    return np.memmap(tok_f, np.float16, "r", shape=(n, L, d)), emb, visual, winfo


def a_heads(X, y, groups, folds, recA, k, device):
    """A's reference (3 training folds, for tuning) and A's refit (4 folds) for test fold k."""
    K, seed = int(folds.max()) + 1, C.SEED
    v = (k + 1) % K
    tr_in, va, tr_all = (folds != k) & (folds != v), folds == v, folds != k
    i = T.grid_settings(T.GRID_A).index({kk: recA["chosen"][kk] for kk in T.GRID_A})
    s = {**T.FIXED_A, **recA["chosen"]}
    st_in, st_all = T.Standardiser().fit(X[tr_in]), T.Standardiser().fit(X[tr_all])
    ref_fn, vp, _ = T.fit_head(st_in(X[tr_in]), y[tr_in], s, recA["epochs"], seed + 1000 * k + i, device, st_in(X[va]))
    ref = T.group_fpr_mae(vp[-1], y[va], groups[va])
    ref["tpr"] = T.group_tpr(vp[-1], y[va], groups[va])
    refit_fn, _, _ = T.fit_head(st_all(X[tr_all]), y[tr_all], s, recA["epochs"], seed + 1000 * k, device)
    return (tr_in, va, tr_all), (st_in, ref_fn, ref), (st_all, refit_fn)


def run_config(name, blocks, adversary, cache, X, y, groups, folds, recA, visual, device, fold_list, log=print):
    pred, record = np.full(len(y), np.nan), []
    for k in fold_list:
        t0 = time.time()
        (tr_in, va, tr_all), (st_in, ref_fn, ref), (st_all, refit_fn) = a_heads(X, y, groups, folds, recA[k], k, device)
        rows_in, rows_va, rows_all, rows_te = (np.flatnonzero(m) for m in (tr_in, va, tr_all, folds == k))
        log(f"{name} fold {k}: tuning on {len(rows_in)} rows, validation {len(rows_va)}")
        _, vp, info_t = fit_f(cache, rows_in, y[rows_in], groups[rows_in], ref_fn, st_in.mu, st_in.sd, visual, START,
                              blocks, FIXED_F, FIXED_F["max_epochs"], C.SEED + 1000 * k, device, rows_val=rows_va,
                              adversary=adversary, log=log)
        stats = [T.selection_stats_v3(p, y[rows_va], groups[rows_va], ref) for p in vp]
        best = min(range(len(stats)), key=lambda e: T._rank_key(stats[e], "worst_group_v3"))
        log(f"{name} fold {k}: chose {best + 1} epochs (validation worst-group TPR "
            f"{stats[best]['worst_group_tpr']:.3f}, allowed {stats[best]['feasible']}); refit on {len(rows_all)} rows")
        fn, _, info_r = fit_f(cache, rows_all, y[rows_all], groups[rows_all], refit_fn, st_all.mu, st_all.sd, visual,
                              START, blocks, FIXED_F, best + 1, C.SEED + 1000 * k, device, adversary=adversary, log=log)
        pred[rows_te] = fn(rows_te)
        record.append({"test_fold": k, "epochs": best + 1, "lora_blocks": blocks,
                       "inner_val_worst_group_tpr": stats[best]["worst_group_tpr"],
                       "inner_val_feasible": stats[best]["feasible"], "inner_val_mae": stats[best]["mae"],
                       "test_mae": float(np.abs(pred[rows_te] - y[rows_te]).mean()),
                       "n_lora_params": info_r["n_lora_params"],
                       "epoch_seconds_tuning": info_t["epoch_seconds"], "epoch_seconds_refit": info_r["epoch_seconds"],
                       **({"adv_bal_acc_by_epoch_refit": info_r["adv_bal_acc_by_epoch"]} if adversary else {}),
                       "seconds": round(time.time() - t0, 1)})
        log(f"{name} fold {k}: test MAE {record[-1]['test_mae']:.2f} ({record[-1]['seconds']:.0f} s)")
    return pred, record


def main():
    import pandas as pd
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--configs", default="F1_1,F1_4,F2")
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold_list = [int(f) for f in a.folds.split(",")]
    rows = get_utkface(work)
    ids = np.array([r["id"] for r in rows])
    y = np.array([r["age"] for r in rows], float)
    groups = np.array([r["group"] for r in rows])
    cache, X, visual, winfo = build_cache(rows, work, device)
    fold_of = make_folds(list(ids), list(groups), list(y))
    folds = np.array([fold_of[i] for i in ids])

    # model A on the exact frozen embeddings (same procedure and seeds as results/models/A.json)
    predA, recA = T.oof_predictions(X, y, folds, T.GRID_A, T.FIXED_A, device, groups=groups, select="mae")
    dA = M.prepare(M.frame(ids, groups, y, predA))
    plan = M.BootPlan(dA)
    repA = M.model_report(dA, plan)
    committedA = json.loads((REPO / "results/models/A.json").read_text())["report"]["tpr_gap_W_minus_B"]["est"]
    print(f"A rebuilt: gap {repA['tpr_gap_W_minus_B']['est']:.3f} (committed {committedA:.3f})")

    done = {}
    for name in a.configs.split(","):
        cfg = CONFIGS[name]
        blocks = cfg["blocks"]
        if blocks is None:   # F2: better F1 depth by mean validation worst-group TPR
            f1 = {n: json.loads((REPO / f"results/models/{n}.json").read_text()) for n in ("F1_1", "F1_4")
                  if (REPO / f"results/models/{n}.json").exists()}
            f1.update(done)
            assert f1, "run F1_1 and F1_4 before F2"
            best = max(f1, key=lambda n: np.mean([r["inner_val_worst_group_tpr"] for r in f1[n]["folds"]]))
            blocks = CONFIGS[best]["blocks"]
            print(f"F2 uses the depth of {best}: blocks {blocks}")
        pred, record = run_config(name, blocks, cfg["adversary"], cache, X, y, groups, folds, recA, visual,
                                  device, fold_list)
        out = {"model": name, "lora_blocks": blocks, "adversary": cfg["adversary"], "fixed": FIXED_F,
               "start_block": START, "folds": record, "device": device, "torch": torch.__version__,
               "clip_weights": winfo, "A_rebuilt_gap": repA["tpr_gap_W_minus_B"], "A_committed_gap": committedA}
        if len(fold_list) == 5:
            pd.DataFrame({"id": ids, "fold": folds, "pred_age": pred}).to_csv(work / f"oof__{name}.csv", index=False)
            d = M.prepare(M.frame(ids, groups, y, pred))
            out["report"] = M.model_report(d, plan)
            out["paired_vs_A"] = M.paired_vs_A(dA, d, plan)
            g, r = out["report"]["tpr_gap_W_minus_B"], out["paired_vs_A"]["gap_reduction_A_minus_M"]
            print(f"{name}: gap {g['est']:.3f} [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}], reduction vs A {r['est']:+.3f} "
                  f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}], MAE {out['report']['mae_overall']['est']:.2f}")
            res = write_result(REPO / "results" / "models" / f"{name}.json", out, C.as_dict(), REPO)
        else:
            res = write_result(REPO / "results" / "timing" / f"{name}_folds{a.folds.replace(',', '')}.json",
                               out, C.as_dict(), REPO)
        done[name] = out
        print("->", res)


if __name__ == "__main__":
    main()
