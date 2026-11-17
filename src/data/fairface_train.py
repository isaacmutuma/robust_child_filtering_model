"""FairFace train split (0.25 padding) read straight from the pinned parquet shards.

Ages are bins: `age` holds NaN and `age_bin` the bin label; models use an interval loss.
Ids are 'ff_train_<n>' in shard order (n from 1), mirroring the audit's val numbering.
"""

import io
import math

from configs import config as C


def list_fairface_train(shard_paths):
    """Return rows: dicts id/age/age_bin/gender/group/image_bytes, in shard order."""
    import pyarrow.parquet as pq
    rows, n = [], 0
    for sp in shard_paths:
        for r in pq.read_table(sp).to_pylist():
            n += 1
            rows.append({"id": f"ff_train_{n}", "age": math.nan,
                         "age_bin": C.FAIRFACE_AGE_BINS[r["age"]],
                         "gender": C.FAIRFACE_GENDERS[r["gender"]].lower(),
                         "group": C.FAIRFACE_RACES[r["race"]],
                         "image_bytes": r["image"]["bytes"]})
    return rows


def open_bytes(b):
    """PIL RGB image from encoded bytes."""
    from PIL import Image
    return Image.open(io.BytesIO(b)).convert("RGB")
