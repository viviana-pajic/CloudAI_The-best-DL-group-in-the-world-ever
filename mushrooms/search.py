"""Preparation and training for the selected Extra Trees model."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer, RobustScaler, StandardScaler
ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common.protocol import load_mushrooms
SEED = 20261010
THREADS = 4
NUMERIC = ["cap-diameter", "stem-height", "stem-width"]
CATEGORICAL = ["spore-print-color", "gill-color", "habitat", "season", "ring-type", "cap-shape", "stem-surface"]
NOISE = ["jumbled_noise_0", "jumbled_noise_1"]


def read_partition(which="search"):
    """Never allow accidental confirmation access from a search call."""
    if which not in {"search", "calibration"}:
        raise ValueError("Confirmation is guarded and requires a separate authorized script.")
    selected = load_mushrooms(which).sort_index()
    if not selected.index.is_unique:
        raise AssertionError("Partition membership differs from manifest.")
    X = selected.drop(columns="class")
    y = selected["class"].map({"e": 0, "p": 1}).astype(int)
    return X, y

class FeatureDesign(BaseEstimator, TransformerMixin):
    """Stateless feature hypotheses; no label-derived information."""
    def __init__(self, include_noise=False, engineered=False, missing_counts=False, drop_spore=False):
        self.include_noise = include_noise
        self.engineered = engineered
        self.missing_counts = missing_counts
        self.drop_spore = drop_spore

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        columns = NUMERIC + CATEGORICAL + (NOISE if self.include_noise else [])
        if self.drop_spore:
            columns = [c for c in columns if c != "spore-print-color"]
        out = X[columns].copy()
        if self.engineered:
            for name in NUMERIC:
                out[name + "_log1p"] = np.log1p(out[name].clip(lower=0))
            for first, second in [("cap-diameter", "stem-height"), ("stem-width", "stem-height"), ("cap-diameter", "stem-width")]:
                out[first + "_over_" + second] = (out[first] / out[second].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        if self.missing_counts:
            out["missing_numeric_count"] = X[NUMERIC].isna().sum(axis=1)
            out["missing_categorical_count"] = X[CATEGORICAL].isna().sum(axis=1)
        return out

def preprocessing(config, X):
    designed = FeatureDesign(**config["features"]).fit_transform(X)
    numeric = list(designed.select_dtypes(include="number").columns)
    categorical = [c for c in designed if c not in numeric]
    prep = config["preprocessing"]
    numeric_steps = []
    if prep["numeric_missing"] == "knn":
        numeric_steps.extend([("impute_scale", StandardScaler()),
                              ("impute", KNNImputer(n_neighbors=7, weights="distance", add_indicator=prep["missing_flags"]))])
    elif prep["numeric_missing"] == "iterative":
        numeric_steps.append(("impute", IterativeImputer(max_iter=10, random_state=SEED,
                                                         add_indicator=prep["missing_flags"], skip_complete=True)))
    elif prep["numeric_missing"] != "native":
        numeric_steps.append(("impute", SimpleImputer(strategy=prep["numeric_missing"], add_indicator=prep["missing_flags"])))
    if prep.get("scale") == "standard":
        numeric_steps.append(("scale", StandardScaler()))
    elif prep.get("scale") == "robust":
        numeric_steps.append(("scale", RobustScaler()))
    elif prep.get("scale") == "quantile":
        numeric_steps.append(("scale", QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=SEED)))
    numeric_transform = Pipeline(numeric_steps) if numeric_steps else "passthrough"
    cat_impute = SimpleImputer(strategy=prep["categorical_missing"], fill_value="__MISSING__")
    if prep["encoding"] == "ordinal":
        cat_encode = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        cat_encode = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=prep.get("min_frequency"))
    return ColumnTransformer([
        ("num", numeric_transform, numeric),
        ("cat", Pipeline([("impute", cat_impute), ("encode", cat_encode)]), categorical),
    ], sparse_threshold=0.0), len(numeric), len(categorical)

def threshold_for_recall(y, probabilities, target):
    fpr, tpr, thresholds = roc_curve(y, probabilities, drop_intermediate=False)
    valid = np.where(tpr >= target)[0]
    index = valid[np.argmin(fpr[valid])]
    return float(thresholds[index]), float(1 - fpr[index]), float(tpr[index])

def metrics(y, probabilities, threshold=.5):
    predicted = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "ap": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "log_loss": float(log_loss(y, np.clip(probabilities, 1e-7, 1-1e-7))),
        "accuracy": float((tn + tp) / len(y)),
        "poison_recall": float(tp / (tp + fn)),
        "edible_specificity": float(tn / (tn + fp)),
        "poison_precision": float(tp / (tp + fp)) if tp + fp else 0.,
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def build_model(config, X):
    if config['family'] != 'extra_trees':
        raise ValueError('This helper trains the selected Extra Trees family.')
    features = FeatureDesign(**config['features'])
    prep, _, _ = preprocessing(config, X)
    model = ExtraTreesClassifier(**config['params'], random_state=SEED, n_jobs=THREADS)
    return Pipeline([('features', features), ('prepare', prep), ('model', model)])
