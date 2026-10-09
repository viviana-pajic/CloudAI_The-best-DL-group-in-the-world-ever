"""Current mushroom preparation and development evaluation; never score the test.

Run with the Python 3.11 requirements-pycaret.txt environment. Historical results
remain in notebook 02 and reports/mushroom_validation_metrics.csv.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from mushrooms import (
    ROOT, EXPECTED_SHA256, FEATURE_COLUMNS, clean, cv_splitter, input_schema,
    load_data, make_split, preprocessor, scores,
)

REPORTS = ROOT / "reports/mushrooms"
PROCESSED = ROOT / "data/processed/mushrooms"
CONTRACT_VERSION = "mushroom-development-v1"
SEED = 42


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def environment():
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "pycaret"]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}


def feature_groups(features):
    """Exact equality of retained predictors, including their missingness pattern.

    Tuple keys preserve equality without treating a compressed hash as identity.
    Equal predictors do not establish that two rows are the same source mushroom.
    """
    groups, codes = {}, []
    for row in features.itertuples(index=False, name=None):
        key = tuple(None if pd.isna(value) else value for value in row)
        if key not in groups:
            groups[key] = len(groups)
        codes.append(groups[key])
    return np.asarray(codes)


def load_development(frame=None):
    """Return only development X/y. Test features/labels are not returned."""
    frame = load_data() if frame is None else frame
    splits = make_split(frame)
    development = splits["train"] + splits["validation"]
    prepared = clean(frame.loc[development]).copy()
    if set(prepared.columns) != set(FEATURE_COLUMNS + ["class"]):
        raise ValueError("Unexpected mushroom feature contract.")
    X = prepared[FEATURE_COLUMNS]
    y = prepared["class"].map({"e": 0, "p": 1}).astype(int)
    if not X.index.is_unique or set(X.index) & set(splits["test"]):
        raise ValueError("Development indices must be unique and disjoint from test.")
    return X, y, splits


def fold_fingerprint(X, y):
    ids = [X.index[val].tolist() for _, val in cv_splitter().split(X, y)]
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def contract(X, y):
    return {
        "version": CONTRACT_VERSION, "dataset_sha256": EXPECTED_SHA256,
        "features": FEATURE_COLUMNS, "development_rows": len(X),
        "positive_class": "p", "target_mapping": {"e": 0, "p": 1},
        "cleaning": "Drop two jumbled_noise columns; retain every row and observed value.",
        "preprocessing": "Fold-fitted median numeric imputation; no numeric missing flags; "
                         "categorical __MISSING__ and one-hot encoding; numeric scaling only for LR.",
        "cv": "RepeatedStratifiedKFold: 5 folds, 2 repeats, random_state=42",
        "fold_fingerprint": fold_fingerprint(X, y),
        "ranking_metric": "mean fold average precision (not trapezoidal PR-AUC)",
        "secondary_metric": "mean fold ROC-AUC",
        "operating_metrics": "Development OOF scores averaged across 2 repeats; fixed threshold 0.5.",
        "threshold_status": "No operating threshold selected in this checkpoint.",
        "test_model_predictions_computed": False,
        "environment": environment(),
    }


def collision_audit(frame, splits):
    features = clean(frame).drop(columns="class")[FEATURE_COLUMNS]
    codes = feature_groups(features)
    partitions = {i: name for name, ids in splits.items() for i in ids}
    findings = []
    for group in pd.Series(codes).value_counts().loc[lambda counts: counts > 1].index:
        indices = features.index[codes == group].tolist()
        first = features.loc[indices[0]]
        findings.append({
            "rows": [{"source_row": int(i), "partition": partitions[i]} for i in indices],
            "labels_agree": bool(frame.loc[indices, "class"].nunique() == 1),
            "label_counts": {str(k): int(v) for k, v in frame.loc[indices, "class"].value_counts().items()},
            "missingness_identical": bool(features.loc[indices].isna().drop_duplicates().shape[0] == 1),
            "missing_features": first.index[first.isna()].tolist(),
            "retained_values": {key: (None if pd.isna(value) else value) for key, value in first.items()},
        })
    return {
        "raw_feature_duplicate_extra_rows": int(frame.drop(columns="class").duplicated().sum()),
        "retained_feature_duplicate_extra_rows": int(features.duplicated().sum()),
        "groups": findings,
        "interpretation": "Identical retained predictors, not proven duplicate source observations. "
                          "There is no source observation ID; removed noise values distinguish the raw rows.",
        "policy": "Keep all rows and the original test membership. Measure development sensitivity "
                  "by excluding matching training predictors within each CV fold only.",
        "test_access": "Label agreement inspected only for collision integrity, as requested. "
                       "No test model predictions or performance metrics computed.",
    }


def prepare():
    """Freeze existing row membership and write compact preparation evidence."""
    frame = load_data()
    X, y, splits = load_development(frame)
    manifest_path = ROOT / "mushrooms/split_manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, {"dataset_sha256": EXPECTED_SHA256,
                                  "original_seed": SEED, "splits": splits})
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS / "contract.json", contract(X, y))
    write_json(REPORTS / "collisions.json", collision_audit(frame, splits))
    # This is model-ready raw input, not a globally fitted numeric design matrix.
    X.assign(class_label=y).to_csv(PROCESSED / "development.csv", index_label="source_row")
    return X, y, splits


def candidates(X):
    """A bounded hypothesis-led set: reference, linear, bagged and boosted trees."""
    estimators = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=3000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=3,
                                                random_state=SEED, n_jobs=2),
        "extra_trees": ExtraTreesClassifier(n_estimators=150, min_samples_leaf=3,
                                            random_state=SEED, n_jobs=2),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED),
    }
    return {name: Pipeline([
        ("prepare", preprocessor(X, scale=name == "logistic_regression", dense=True)),
        ("model", model),
    ]) for name, model in estimators.items()}


def evaluate_model(name, pipeline, X, y, exclude_collisions=False):
    """Same repeated folds for every candidate; all preprocessing stays inside fit.

    The collision sensitivity keeps validation rows/folds unchanged and excludes
    only training rows whose retained predictors equal a validation row.
    """
    frozen = json.loads((ROOT / "mushrooms/split_manifest.json").read_text(encoding="utf-8"))
    allowed = set(frozen["splits"]["train"] + frozen["splits"]["validation"])
    if not X.index.is_unique or not X.index.equals(y.index) or not set(X.index) <= allowed:
        raise ValueError("Evaluation accepts development rows only.")
    groups = feature_groups(X)
    oof_sum, appearances = np.zeros(len(X)), np.zeros(len(X), dtype=int)
    rows = []
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        for fold, (train, validation) in enumerate(cv_splitter().split(X, y)):
            original_size = len(train)
            if exclude_collisions:
                train = train[~np.isin(groups[train], groups[validation])]
            model = clone(pipeline)
            fit_start = time.perf_counter()
            model.fit(X.iloc[train], y.iloc[train])
            fit_seconds = time.perf_counter() - fit_start
            probabilities = model.predict_proba(X.iloc[validation])[:, list(model.classes_).index(1)]
            train_probabilities = model.predict_proba(X.iloc[train])[:, list(model.classes_).index(1)]
            measured = scores(y.iloc[validation], (probabilities >= .5).astype(int), probabilities)
            train_measured = scores(y.iloc[train], (train_probabilities >= .5).astype(int), train_probabilities)
            rows.append({"model": name, "fold": fold, "repeat": fold // 5,
                         "training_rows": len(train), "validation_rows": len(validation),
                         "collision_training_exclusions": original_size - len(train),
                         "fit_seconds": fit_seconds, "train_average_precision": train_measured["average_precision"],
                         **measured})
            oof_sum[validation] += probabilities
            appearances[validation] += 1
    if not np.all(appearances == 2):
        raise AssertionError("Every development row must have exactly two held-out predictions.")
    probabilities = oof_sum / appearances
    folds = pd.DataFrame(rows)
    operating = scores(y, (probabilities >= .5).astype(int), probabilities)
    summary = {
        "model": name, "contract_version": CONTRACT_VERSION,
        "fold_fingerprint": fold_fingerprint(X, y),
        "cv_average_precision": folds.average_precision.mean(),
        "cv_average_precision_std": folds.average_precision.std(ddof=0),
        "cv_roc_auc": folds.roc_auc.mean(), "cv_roc_auc_std": folds.roc_auc.std(ddof=0),
        "train_average_precision": folds.train_average_precision.mean(),
        "mean_fit_seconds": folds.fit_seconds.mean(),
        "evaluation_seconds": time.perf_counter() - start,
        "threshold": .5, "oof_averaging": "mean of two held-out probabilities per row",
        "collision_training_exclusions": int(folds.collision_training_exclusions.sum()),
        **{"oof_" + key: value for key, value in operating.items()},
    }
    predictions = pd.DataFrame({"source_row": X.index, "label": y.to_numpy(),
                                "poison_probability": probabilities,
                                "missing_feature_count": X.isna().sum(axis=1).to_numpy()})
    return summary, folds, predictions


def record_evaluation(names, prefix, X, y):
    pipelines = candidates(X)
    summaries, folds = [], []
    for name in names:
        print(f"Evaluating {name}: 5 folds x 2 repeats...", flush=True)
        row, fold_rows, oof = evaluate_model(name, pipelines[name], X, y)
        summaries.append(row)
        folds.append(fold_rows)
        oof.to_csv(PROCESSED / f"{name}_oof.csv", index=False)
        print(f"  AP={row['cv_average_precision']:.6f}, ROC-AUC={row['cv_roc_auc']:.6f}", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(REPORTS / f"{prefix}_cv.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(REPORTS / f"{prefix}_folds.csv", index=False)
    write_json(REPORTS / f"{prefix}_run.json", {
        "contract": contract(X, y),
        "model_parameters": {name: pipelines[name].named_steps["model"].get_params() for name in names},
        "source_sha256": {filename: hashlib.sha256((ROOT / "src" / filename).read_bytes()).hexdigest()
                          for filename in ["mushrooms.py", "mushroom_workflow.py"]},
    })
    return summary


def export_demo(X, y, summary):
    """One provisional artifact bundle; not a frozen final model or test result."""
    pipeline = candidates(X)["random_forest"]
    with threadpool_limits(limits=2):
        pipeline.fit(X, y)
    row = summary.set_index("model").loc["random_forest"]
    metadata = {
        "name": "Integrated random forest reference (provisional)",
        "status": "development reference; final model and operating threshold not selected",
        "contract_version": CONTRACT_VERSION, "dataset_sha256": EXPECTED_SHA256,
        "threshold": .5, "training_rows": len(X), "reserved_test_rows": 1000,
        "final_test_evaluated": False, "metric_scope": "development OOF (two repeats averaged)",
        "metrics": {"accuracy": float(row.oof_accuracy), "poison_recall": float(row.oof_poison_recall),
                    "average_precision": float(row.cv_average_precision)},
        "environment": environment(), "feature_columns": FEATURE_COLUMNS,
    }
    bundle = {"pipeline": pipeline, "schema": input_schema(X), "metadata": metadata}
    path = ROOT / "models/mushroom_candidate.joblib"
    path.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, path)
    reloaded = joblib.load(path)
    np.testing.assert_allclose(pipeline.predict_proba(X.iloc[:5]),
                               reloaded["pipeline"].predict_proba(X.iloc[:5]))


def run_baselines():
    X, y, _ = prepare()
    summary = record_evaluation(["majority_baseline", "logistic_regression", "random_forest"],
                                "baseline", X, y)
    print("Collision sensitivity: same RF, same validation rows; exclude fold-training matches.", flush=True)
    sensitivity, fold_rows, _ = evaluate_model("random_forest", candidates(X)["random_forest"],
                                              X, y, exclude_collisions=True)
    pd.DataFrame([sensitivity]).to_csv(REPORTS / "collision_sensitivity.csv", index=False)
    fold_rows.to_csv(REPORTS / "collision_sensitivity_folds.csv", index=False)
    export_demo(X, y, summary)
    print(summary[["model", "cv_average_precision", "cv_roc_auc", "oof_poison_recall"]].to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "baseline"], default="baseline", nargs="?")
    args = parser.parse_args()
    if args.action == "prepare":
        X, y, _ = prepare()
        print(f"Prepared {len(X)} development rows x {X.shape[1]} raw features. Test remains reserved.")
    else:
        run_baselines()
