"""Cross-validation folds shared by every model (docs/SUCCESS_CRITERIA.md §1).

Outer folds: StratifiedKFold over group x age band, seed 0, on ids sorted by name, so
the result does not depend on input order. Cells smaller than N_FOLDS are merged into
'<group>|rare', then into 'rare'; if that pool is still smaller than N_FOLDS, its images
join their group's largest cell. So StratifiedKFold never sees a class it can't split.

Inner validation (hyperparameter tuning, never the test fold): for outer test fold k,
tune on folds != k, (k+1) % N and validate on fold (k+1) % N; then refit on all four
training folds with the chosen settings.
"""

from collections import Counter

import numpy as np

from configs import config as C


def age_band(age, bands=None):
    """Label like '10-17' of the band holding `age` ('60+' for the last band)."""
    for lo, hi in bands or C.FOLD_AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}+" if hi >= 200 else f"{lo}-{hi}"
    raise ValueError(f"age {age} outside all bands")


def strat_keys(groups, ages, k):
    """Stratification key per image, with small cells merged (see module docstring)."""
    keys = [f"{g}|{age_band(a)}" for g, a in zip(groups, ages)]
    cnt = Counter(keys)
    keys = [x if cnt[x] >= k else f"{x.split('|')[0]}|rare" for x in keys]
    cnt = Counter(keys)
    keys = [x if cnt[x] >= k else "rare" for x in keys]
    if 0 < Counter(keys)["rare"] < k:   # pool still too small: join the group's largest cell
        cnt = Counter(keys)
        biggest = {}
        for x in keys:
            g = x.split("|")[0]
            if x != "rare" and cnt[x] > cnt.get(biggest.get(g), 0):
                biggest[g] = x
        top = max(cnt, key=cnt.get)
        keys = [biggest.get(g.split("|")[0], top) if x == "rare" else x
                for x, g in zip(keys, [f"{gg}|" for gg in groups])]
    return keys


def make_folds(ids, groups, ages, k=None, seed=None):
    """dict id -> fold (0..k-1). Deterministic given the same (id, group, age) set."""
    from sklearn.model_selection import StratifiedKFold
    k = k or C.N_FOLDS
    seed = C.SEED if seed is None else seed
    order = np.argsort(np.asarray(ids, dtype=object), kind="stable")
    ids_s = [ids[i] for i in order]
    keys = strat_keys([groups[i] for i in order], [ages[i] for i in order], k)
    if len(set(ids_s)) != len(ids_s):
        raise ValueError("duplicate ids")
    fold = np.empty(len(ids_s), dtype=int)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for f, (_, test) in enumerate(skf.split(np.zeros(len(ids_s)), keys)):
        fold[test] = f
    return dict(zip(ids_s, fold.tolist()))


def inner_val_fold(test_fold, k=None):
    """Inner validation fold used for tuning when `test_fold` is held out."""
    return (test_fold + 1) % (k or C.N_FOLDS)
