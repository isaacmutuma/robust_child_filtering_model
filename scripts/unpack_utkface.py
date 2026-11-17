"""Unpack the audit's pinned UTKFace archive from Drive to DATA_DIR/UTKFace (Colab).

  CRF_ENV=colab python scripts/unpack_utkface.py
Checks the sha256 first; path-traversal-safe; idempotent (skips if the file count matches).
"""

import tarfile
from pathlib import Path, PurePosixPath

from _common import sha256  # noqa: F401  (also puts the repo on sys.path)
from configs import config as C

S = C.SOURCES["utkface"]


def main():
    out = Path(C.DATA_DIR) / "UTKFace"
    if out.is_dir() and sum(1 for _ in out.iterdir()) == S["n_files"]:
        print("UTKFace already unpacked:", out); return
    tar = Path(C.RAW_DIR) / S["file"]
    got = sha256(tar)
    if got != S["sha256"]:
        raise SystemExit(f"UTKFace sha256 mismatch for {tar}: {got}")
    Path(C.DATA_DIR).mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as t:
        members = [m for m in t.getmembers() if m.isfile() and not m.name.startswith("/")
                   and ".." not in PurePosixPath(m.name).parts]
        t.extractall(C.DATA_DIR, members=members)
    cands = [p for p in Path(C.DATA_DIR).iterdir()
             if p.is_dir() and p.name.lower().startswith("utkface") and p.name != "UTKFace"]
    if not out.exists() and cands:
        cands[0].rename(out)
    n = sum(1 for _ in out.iterdir())
    assert n == S["n_files"], f"UTKFace: {n} files, expected {S['n_files']}"
    print("UTKFace files:", n, "sha256 OK")


if __name__ == "__main__":
    if C.ENV != "colab":
        raise SystemExit("run on Colab (CRF_ENV=colab)")
    main()
