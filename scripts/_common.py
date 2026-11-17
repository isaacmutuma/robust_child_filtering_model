"""Shared script helpers: repo on sys.path, sha256, atomic copy to Drive."""

import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def sha256(p, bs=1 << 20):
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def atomic_copy(src, dst):
    """Copy src to dst via dst.tmp + rename, so a disconnect never leaves a half file."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy(src, tmp)
    tmp.replace(dst)
    return dst
