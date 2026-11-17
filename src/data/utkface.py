"""UTKFace labels from file names (same parse as the audit's loader).

[age]_[gender]_[race]_[datetime].jpg.chip.jpg ; gender 0 male / 1 female ;
race 0 White, 1 Black, 2 Asian, 3 Indian, 4 Others. Malformed names are skipped and
counted, never guessed (the pinned archive has 3).
"""

import re
from pathlib import Path

from configs import config as C

_NAME = re.compile(r"^(\d+)_([01])_([0-4])_(\d+)")
_GENDER = {0: "male", 1: "female"}


def parse_name(name):
    """(id, age, gender, group) for a UTKFace file name, or None if malformed."""
    m = _NAME.match(name)
    if not m:
        return None
    return name.split(".")[0], int(m[1]), _GENDER[int(m[2])], C.UTKFACE_RACES[int(m[3])]


def list_utkface(root):
    """Return (rows, skipped). rows: dicts id/age/gender/group/path, sorted by file name."""
    rows, skipped = [], []
    for p in sorted(Path(root).iterdir()):
        if not p.name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parsed = parse_name(p.name)
        if parsed is None:
            skipped.append(p.name)
            continue
        i, age, g, grp = parsed
        rows.append({"id": i, "age": float(age), "age_bin": "", "gender": g, "group": grp,
                     "path": str(p)})
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate UTKFace ids")
    return rows, skipped
