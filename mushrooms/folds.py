"""Shared exact research folds, independent of estimator and preprocessing."""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

NUMERIC = ["cap-diameter", "stem-height", "stem-width"]
CATEGORICAL = ["spore-print-color", "gill-color", "habitat", "season", "ring-type", "cap-shape", "stem-surface"]
SEED = 20261010


def informative_groups(X):
    keys = [tuple(None if pd.isna(v) else v for v in row)
            for row in X[NUMERIC + CATEGORICAL].itertuples(index=False, name=None)]
    return pd.factorize(pd.Series(keys, dtype="object"))[0]


def cv_folds(X, y, n_splits=3, cv_seed=SEED):
    if hasattr(y, "index") and not X.index.equals(y.index):
        raise ValueError("X/y index order differs")
    return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
                .split(X, y, informative_groups(X)))


def fold_fingerprint(X, y, n_splits=3, cv_seed=SEED):
    membership = [X.index[valid].tolist() for _, valid in cv_folds(X, y, n_splits, cv_seed)]
    return hashlib.sha256(json.dumps(membership).encode()).hexdigest()
