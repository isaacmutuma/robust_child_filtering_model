"""Extract frozen CLIP ViT-L/14 image embeddings (QuickGELU) for one dataset (Colab GPU).

  CRF_ENV=colab python scripts/extract_embeddings.py --dataset utkface
  CRF_ENV=colab python scripts/extract_embeddings.py --dataset fairface_train
  add --limit 256 for a smoke run (tagged 'smoke', never used by models)

Per-image output (Drive only): EMB_DIR/<dataset>__clip_vitl14[__smoke].npz with
  ids, emb (float32, raw = before L2 normalisation), age, age_bin, gender, group.
Aggregate manifest (committable): results/embeddings/<same stem>.json with the npz sha256,
weights, counts, git commit and config. Images that fail to load are listed by COUNT only
and excluded from the npz.
"""

import argparse
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from _common import REPO, atomic_copy, sha256
from configs import config as C
from src.provenance import write_result


def rows_for(dataset):
    """Rows + an image-opening function for `dataset`."""
    from PIL import Image
    if dataset == "utkface":
        from src.data.utkface import list_utkface
        rows, skipped = list_utkface(Path(C.DATA_DIR) / "UTKFace")
        assert len(rows) == C.SOURCES["utkface"]["n_records"], (len(rows), len(skipped))
        return rows, lambda r: Image.open(r["path"]).convert("RGB"), len(skipped)
    if dataset == "fairface_train":
        from huggingface_hub import hf_hub_download
        from src.data.fairface_train import list_fairface_train, open_bytes
        s = C.SOURCES["fairface_train"]
        shards = []
        for f, want in s["files"].items():
            p = hf_hub_download(s["hf_repo"], f, revision=s["revision"], repo_type=s["repo_type"],
                                cache_dir=str(C.HF_CACHE_DIR))
            got = sha256(p)
            if got != want:
                raise SystemExit(f"FairFace shard sha256 mismatch {f}: {got}")
            shards.append(p)
        return list_fairface_train(shards), lambda r: open_bytes(r["image_bytes"]), 0
    raise SystemExit(f"unknown dataset {dataset}")


class _DS:
    """torch Dataset: (pixel tensor, ok flag, error name) per row; never raises on bad images."""

    def __init__(self, rows, open_fn, preprocess):
        self.rows, self.open_fn, self.pre = rows, open_fn, preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        try:
            return self.pre(self.open_fn(self.rows[i])), True, ""
        except Exception as e:  # counted, excluded
            return torch.zeros(3, 224, 224), False, type(e).__name__


def main():
    import torch
    from src.clip_model import load_clip
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["utkface", "fairface_train"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no GPU: switch the Colab runtime to T4")

    rows, open_fn, n_skipped_names = rows_for(a.dataset)
    if a.limit:
        rows = rows[: a.limit]
    stem = f"{a.dataset}__{C.EMB_TAG}" + ("__smoke" if a.limit else "")
    model, preprocess, _, winfo = load_clip("cuda")

    dl = torch.utils.data.DataLoader(_DS(rows, open_fn, preprocess), batch_size=a.batch_size,
                                     num_workers=a.workers, shuffle=False)
    embs, oks, errs, t0 = [], [], Counter(), time.time()
    with torch.no_grad():
        for x, ok, err in dl:
            embs.append(model.encode_image(x.cuda()).float().cpu().numpy())
            oks.append(ok.numpy())
            errs.update(e for e in err if e)
            print(f"\r{sum(len(o) for o in oks)}/{len(rows)}", end="", flush=True)
    print()
    secs = time.time() - t0
    emb, ok = np.concatenate(embs), np.concatenate(oks).astype(bool)
    keep = [r for r, k in zip(rows, ok) if k]
    emb = emb[ok]
    assert np.isfinite(emb).all(), "non-finite embedding"

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / f"{stem}.npz"
        np.savez(tmp, ids=np.array([r["id"] for r in keep]), emb=emb.astype(np.float32),
                 age=np.array([r["age"] for r in keep], dtype=np.float32),
                 age_bin=np.array([r["age_bin"] for r in keep]),
                 gender=np.array([r["gender"] for r in keep]),
                 group=np.array([r["group"] for r in keep]))
        digest = sha256(tmp)
        dst = atomic_copy(tmp, Path(C.EMB_DIR) / tmp.name)
    assert sha256(dst) == digest, "Drive copy differs from local npz"

    norms = np.linalg.norm(emb, axis=1)
    groups = Counter(r["group"] for r in keep)
    minors = Counter(r["group"] for r in keep if r["age"] <= C.MINOR_MAX_AGE)
    teens = Counter(r["group"] for r in keep if 10 <= r["age"] <= C.MINOR_MAX_AGE)
    res = write_result(REPO / "results" / "embeddings" / f"{stem}.json", {
        "dataset": a.dataset, "smoke": bool(a.limit), "n_rows": len(rows),
        "n_embedded": int(ok.sum()), "n_failed": int((~ok).sum()), "fail_reasons": dict(errs),
        "n_skipped_names": n_skipped_names, "dim": int(emb.shape[1]),
        "npz": {"drive_path": str(dst.relative_to(C.DRIVE_ROOT)), "sha256": digest},
        "norm": {"mean": float(norms.mean()), "min": float(norms.min()), "max": float(norms.max())},
        "n_by_group": dict(groups), "n_minors_by_group": dict(minors),
        "n_teens_10_17_by_group": dict(teens),
        "weights": winfo, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "batch_size": a.batch_size, "precision": "fp32", "seconds": round(secs, 1),
    }, C.as_dict(), REPO)
    print(f"{a.dataset}: {int(ok.sum())}/{len(rows)} embedded, {int((~ok).sum())} failed, "
          f"dim {emb.shape[1]}, {secs:.0f}s\nnpz -> {dst}\nmanifest -> {res}")


if __name__ == "__main__":
    main()
