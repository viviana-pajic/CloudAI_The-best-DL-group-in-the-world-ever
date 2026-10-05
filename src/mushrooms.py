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
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "98938790e7dadb00c64eadda1ee9a05961286888"
SOURCE = (
    f"https://raw.githubusercontent.com/mjochen/CloudAI/{COMMIT}/"
    "Discussion%20topics/mushroom_project_dataset.csv"
)
EXPECTED_SHA256 = "1f1a25f2f330458ed95ce9f6ffe3241582312a42cb21797a2b7bd5e5ad281114"


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


def preprocessor(features, scale=False):
    """Describe transformations; fitting happens inside the training pipeline."""
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
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
            ("prepare", preprocessor(features.loc[train], scale=name == "logistic_regression")),
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
