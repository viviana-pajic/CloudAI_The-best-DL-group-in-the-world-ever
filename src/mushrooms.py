"""Reproduce the mushroom audit, development baselines and saved demo model."""

import hashlib
import json
from pathlib import Path
import time
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate,
    train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "98938790e7dadb00c64eadda1ee9a05961286888"
SOURCE = (
    f"https://raw.githubusercontent.com/mjochen/CloudAI/{COMMIT}/"
    "Discussion%20topics/mushroom_project_dataset.csv"
)
EXPECTED_SHA256 = "1f1a25f2f330458ed95ce9f6ffe3241582312a42cb21797a2b7bd5e5ad281114"
# Task 8:shuffled copies of cap-shape with no relation to the label
NOISE_COLUMNS = ["jumbled_noise_0","jumbled_noise_1"]
# Task 11: missing a poisonous mushroom is the costly error, so every model is
# operated at a threshold that catches at least 90% of poisonous mushrooms.
TARGET_POISON_RECALL = 0.90

def load_data():
    """Download once and reject a cached file with a different fingerprint."""
    path = ROOT / "data/raw/mushroom_project_dataset.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            path.write_bytes(response.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError(
            "Dataset checksum mismatch. Use the replacement course CSV. "
            "If the teacher changes it, verify the new source and fingerprint together."
        )
    frame = pd.read_csv(path)
    if "class" not in frame or frame["class"].isna().any():
        raise ValueError("Expected a complete class column.")
    if set(frame["class"].unique()) != {"e", "p"}:
        raise ValueError("Expected edible/poisonous labels e and p.")
    return frame


def audit(frame):
    """Record the dataset shape, missingness and duplicate checks."""
    path = ROOT / "data/raw/mushroom_project_dataset.csv"
    result = {
        "source": SOURCE,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(frame),
        "columns": len(frame.columns),
        "class_counts": {str(k): int(v) for k, v in frame["class"].value_counts().items()},
        "missing_percent": (frame.isna().mean() * 100).round(2).to_dict(),
        "exact_duplicates": int(frame.duplicated().sum()),
        "feature_duplicates": int(frame.drop(columns="class").duplicated().sum()),
        "numeric_summary": frame.select_dtypes(include="number").describe().round(3).to_dict(),
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/data_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def make_split(frame):
    """Reserve fixed stratified training, validation and final-test partitions."""
    if frame.drop(columns="class").duplicated().any():
        raise ValueError("Duplicate features found. Use a group-aware split before training.")
    train, remainder = train_test_split(
        frame.index, test_size=0.4, random_state=42, stratify=frame["class"]
    )
    validation, test = train_test_split(
        remainder, test_size=0.5, random_state=42,
        stratify=frame.loc[remainder, "class"],
    )
    splits = {
        "train": sorted(map(int, train)),
        "validation": sorted(map(int, validation)),
        "test": sorted(map(int, test)),
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/split_indices.json").write_text(
        json.dumps(splits, indent=2), encoding="utf-8"
    )
    return splits

def clean(frame):
    """Apply the cleaning chosen in task 8: drop the two noise columns, keep all rows."""
    return frame.drop(columns=[c for c in NOISE_COLUMNS if c in frame])

def development_data(frame):
    """Cleaned development rows (train + validation) and the reserved test rows.

    The 1,000 test rows are the same as in make_split, so the final test stays
    untouched. The 4,000 development rows are used with cross-validation.
    """
    splits = make_split(frame)
    cleaned = clean(frame)
    features = cleaned.drop(columns="class")
    labels = cleaned["class"].map({"e": 0, "p": 1})
    development = splits["train"] + splits["validation"]
    return (features.loc[development], labels.loc[development],
            features.loc[splits["test"]], labels.loc[splits["test"]])

def cv_splitter(repeats=2):
    """The same stratified 5-fold folds for every model, repeated for stability."""
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=42)

def threshold_for_recall(labels, probabilities, target=TARGET_POISON_RECALL):
    """Highest-precision threshold that still reaches the target poison recall."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    reaches_target = recall[:-1] >= target
    best = int(np.argmax(np.where(reaches_target, precision[:-1], -1.0)))
    return float(thresholds[best])


def cross_validated_scores(name, pipeline, features, labels):
    """Task 11: score one model the agreed way, so every model is comparable."""
    folds = cross_validate(
        pipeline, features, labels, cv=cv_splitter(),
        scoring=["average_precision", "roc_auc"], return_train_score=True, n_jobs=1,
    )
    out_of_fold = cross_val_predict(
        pipeline, features, labels, method="predict_proba",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    )[:, 1]
    threshold = threshold_for_recall(labels, out_of_fold)
    predictions = (out_of_fold >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "model": name,
        "cv_average_precision": folds["test_average_precision"].mean(),
        "cv_average_precision_std": folds["test_average_precision"].std(),
        "train_average_precision": folds["train_average_precision"].mean(),
        "cv_roc_auc": folds["test_roc_auc"].mean(),
        "cv_roc_auc_std": folds["test_roc_auc"].std(),
        "threshold": threshold,
        "poison_recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "poison_precision": tp / (tp + fp) if tp + fp else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }, out_of_fold

def preprocessor(features, scale=False,missing_flags=False):
    """Describe transformations; fitting happens inside the training pipeline."""
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=missing_flags))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def scores(labels, predictions, probabilities):
    """Use poisonous/unsafe as positive class 1 for all classification metrics."""
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "poison_precision": float(precision_score(labels, predictions, zero_division=0)),
        "poison_recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def input_schema(training_features):
    """Describe form inputs using training data only."""
    example = training_features.iloc[0]
    schema = {}
    for column in training_features.columns:
        values = training_features[column]
        if pd.api.types.is_numeric_dtype(values):
            schema[column] = {
                "type": "number",
                "example": None if pd.isna(example[column]) else float(example[column]),
                "observed_min": float(values.min()),
                "observed_max": float(values.max()),
            }
        else:
            schema[column] = {
                "type": "category",
                "example": None if pd.isna(example[column]) else str(example[column]),
                "values": sorted(map(str, values.dropna().unique())),
            }
    return schema


def experiment(frame):
    """Fit three baselines and compare validation results; leave the test unused."""
    audit(frame)
    splits = make_split(frame)
    features = frame.drop(columns="class")
    labels = frame["class"].map({"e": 0, "p": 1})
    train, validation = splits["train"], splits["validation"]
    candidates = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=2
        ),
    }
    rows, fitted = [], {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("prepare", preprocessor(features.loc[train], scale=name == "logistic_regression", missing_flags=True)),
            ("model", estimator),
        ])
        start = time.perf_counter()
        pipeline.fit(features.loc[train], labels.loc[train])
        probabilities = pipeline.predict_proba(features.loc[validation])[
            :, list(pipeline.classes_).index(1)
        ]
        rows.append({
            "model": name, "split": "validation", "threshold": 0.5,
            **scores(labels.loc[validation], pipeline.predict(features.loc[validation]), probabilities),
            "fit_seconds": round(time.perf_counter() - start, 3),
            "train_accuracy": float(accuracy_score(labels.loc[train], pipeline.predict(features.loc[train]))),
        })
        fitted[name] = pipeline

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "reports/mushroom_validation_metrics.csv", index=False)
    model = fitted["random_forest"]
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/mushroom_random_forest_pilot.joblib")
    (ROOT / "models/input_schema.json").write_text(
        json.dumps(input_schema(features.loc[train]), indent=2), encoding="utf-8"
    )
    model_info = {
        "name": "Random forest baseline",
        "threshold": 0.5,
        "dataset_rows": len(frame),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "reserved_test_rows": len(splits["test"]),
        "final_test_evaluated": False,
        "metrics": results.set_index("model").loc["random_forest"].to_dict(),
    }
    (ROOT / "models/model_info.json").write_text(
        json.dumps(model_info, indent=2), encoding="utf-8"
    )
    probabilities = model.predict_proba(features.loc[validation])[
        :, list(model.classes_).index(1)
    ]
    errors = frame.loc[validation].copy()
    errors["poison_probability"] = probabilities
    errors["predicted_class"] = np.where(probabilities >= 0.5, "p", "e")
    errors.loc[errors["predicted_class"] != errors["class"]].to_csv(
        ROOT / "reports/mushroom_validation_errors.csv", index=False
    )
    print(results.round(4).to_string(index=False))
    print(f"\nRows: training={len(train)}, validation={len(validation)}, reserved test={len(splits['test'])}")
    print("Positive class: p (poisonous/unsafe). Results are validation metrics.")
    return results


if __name__ == "__main__":
    experiment(load_data())
