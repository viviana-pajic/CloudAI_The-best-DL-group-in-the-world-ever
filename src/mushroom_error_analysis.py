"""Development-only error analysis and separately evaluated threshold selection.

The RF configuration is fixed. Inner CV chooses only a decision threshold;
outer development folds evaluate that policy. This is not model tuning.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from mushrooms import ROOT, scores
from mushroom_workflow import (
    PROCESSED, REPORTS, SEED, candidates, contract, load_development, write_json,
)

TARGET_RECALL = 0.90


def threshold_table(labels, probabilities):
    """All distinct operating points, with ties classified together using >=."""
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Provide aligned one-dimensional labels containing both classes.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    order = np.argsort(-p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(p) - 1]
    tp = np.r_[0, np.cumsum(sorted_y)[ends]].astype(int)
    fp = np.r_[0, (ends + 1) - np.cumsum(sorted_y)[ends]].astype(int)
    # The first point classifies nothing as poisonous, even if a score equals 1.
    thresholds = np.r_[np.nextafter(1., 2.), sorted_p[ends]]
    positive, negative = int(y.sum()), int((y == 0).sum())
    return pd.DataFrame({
        "threshold": thresholds, "tn": negative - fp, "fp": fp,
        "fn": positive - tp, "tp": tp,
        "poison_precision": np.divide(tp, tp + fp, out=np.zeros(len(tp)), where=(tp + fp) != 0),
        "poison_recall": tp / positive, "specificity": (negative - fp) / negative,
        "f1": 2 * tp / (positive + tp + fp),
    })


def select_threshold(labels, probabilities, target=TARGET_RECALL):
    """Maximize specificity subject to empirical recall >= target.

    Ties use the largest observed threshold. Calibration recall is a constraint
    on these predictions, not a guarantee on new data.
    """
    if not 0 < target <= 1:
        raise ValueError("Recall target must be in (0, 1].")
    table = threshold_table(labels, probabilities)
    feasible = table.loc[table.poison_recall >= target]
    return float(feasible.sort_values(["specificity", "threshold"], ascending=False).iloc[0].threshold)


def operating_metrics(labels, probabilities, threshold=None, predictions=None):
    if predictions is None:
        predictions = (np.asarray(probabilities) >= threshold).astype(int)
    result = scores(labels, predictions, probabilities)
    result["specificity"] = result["tn"] / (result["tn"] + result["fp"])
    result["false_negative_rate"] = result["fn"] / (result["fn"] + result["tp"])
    return result


def evaluate_threshold_policy(X, y, pipeline=None):
    """Nested 5 outer / 3 inner folds; all fitting uses development rows only."""
    manifest = json.loads((ROOT / "mushrooms/split_manifest.json").read_text(encoding="utf-8"))
    allowed = set(manifest["splits"]["train"] + manifest["splits"]["validation"])
    if not X.index.is_unique or not X.index.equals(y.index) or not set(X.index) <= allowed:
        raise ValueError("Threshold evaluation accepts development rows only.")
    pipeline = candidates(X)["random_forest"] if pipeline is None else pipeline
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probabilities, thresholds = np.zeros(len(X)), np.zeros(len(X))
    fold_ids, appearances = np.full(len(X), -1), np.zeros(len(X), dtype=int)
    rows, audit = [], []
    with threadpool_limits(limits=2):
        for fold, (train, validation) in enumerate(outer.split(X, y)):
            X_train, y_train = X.iloc[train], y.iloc[train]
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            calibration = np.zeros(len(train))
            calibration_counts = np.zeros(len(train), dtype=int)
            inner_ids = []
            for inner_train, inner_validation in inner.split(X_train, y_train):
                model = clone(pipeline).fit(X_train.iloc[inner_train], y_train.iloc[inner_train])
                calibration[inner_validation] = model.predict_proba(X_train.iloc[inner_validation])[:, list(model.classes_).index(1)]
                calibration_counts[inner_validation] += 1
                inner_ids.append(X_train.index[inner_validation].tolist())
            assert np.all(calibration_counts == 1)
            threshold = select_threshold(y_train, calibration)
            # No outer validation label has been passed to threshold selection.
            model = clone(pipeline).fit(X_train, y_train)
            p = model.predict_proba(X.iloc[validation])[:, list(model.classes_).index(1)]
            probabilities[validation], thresholds[validation] = p, threshold
            fold_ids[validation], appearances[validation] = fold, appearances[validation] + 1
            rows.append({"fold": fold, "training_rows": len(train), "validation_rows": len(validation),
                         "threshold": threshold,
                         "inner_calibration_recall": float((calibration[y_train.to_numpy() == 1] >= threshold).mean()),
                         **operating_metrics(y.iloc[validation], p, threshold)})
            audit.append({"fold": fold, "outer_validation_source_rows": X.index[validation].tolist(),
                          "inner_validation_ids_sha256": hashlib.sha256(json.dumps(inner_ids).encode()).hexdigest()})
    assert np.all(appearances == 1)
    predictions = pd.DataFrame({"source_row": X.index, "label": y.to_numpy(),
                                "poison_probability": probabilities, "outer_fold": fold_ids,
                                "selected_threshold": thresholds,
                                "policy_prediction": (probabilities >= thresholds).astype(int),
                                "missing_feature_count": X.isna().sum(axis=1).to_numpy()})
    return pd.DataFrame(rows), predictions, audit


def error_slices(predictions, predicted, model, policy):
    data = predictions.copy()
    data["predicted"] = np.asarray(predicted)
    data["missing_group"] = pd.cut(data.missing_feature_count, [-1, 2, 4, 10], labels=["0-2", "3-4", "5-10"])
    rows = []
    for group, part in data.groupby("missing_group", observed=False):
        positive, negative = part.label.eq(1), part.label.eq(0)
        fn = int((positive & part.predicted.eq(0)).sum())
        fp = int((negative & part.predicted.eq(1)).sum())
        rows.append({"model": model, "policy": policy, "missing_features": str(group), "n": len(part),
                     "poisonous_n": int(positive.sum()), "edible_n": int(negative.sum()),
                     "false_negatives": fn, "false_positives": fp,
                     "false_negative_rate": fn / positive.sum() if positive.any() else None,
                     "false_positive_rate": fp / negative.sum() if negative.any() else None})
    return pd.DataFrame(rows)


def run_error_analysis():
    X, y, _ = load_development()
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print("RF threshold policy: 5 outer development folds, 3 inner calibration folds; fixed model.", flush=True)
    folds, predictions, audit = evaluate_threshold_policy(X, y)
    p = predictions.poison_probability.to_numpy()
    # A single provisional threshold for later use: selection evidence only.
    selected = select_threshold(y, p)
    curve = threshold_table(y, p)
    curve.to_csv(REPORTS / "threshold_curve.csv", index=False)
    rows = []
    for threshold in [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
        rows.append({"selection": "fixed grid", "target_recall": None, "threshold": threshold,
                     **operating_metrics(y, p, threshold)})
    for target in [.80, .90, .95]:
        threshold = select_threshold(y, p, target)
        rows.append({"selection": "selected on these OOF scores (descriptive)", "target_recall": target,
                     "threshold": threshold, **operating_metrics(y, p, threshold)})
    pd.DataFrame(rows).to_csv(REPORTS / "threshold_tradeoff.csv", index=False)
    folds.to_csv(REPORTS / "threshold_folds.csv", index=False)
    predictions.to_csv(PROCESSED / "threshold_policy_oof.csv", index=False)
    error_slices(predictions, predictions.policy_prediction, "random_forest", "nested recall policy").to_csv(
        REPORTS / "threshold_error_slices.csv", index=False)
    # Preserve identifiable development failures under the evaluated policy too.
    failures = predictions.loc[predictions.label.eq(1) & predictions.policy_prediction.eq(0)]
    examples = failures.nsmallest(10, "poison_probability").set_index("source_row").join(X)
    examples.to_csv(REPORTS / "threshold_false_negatives.csv", index_label="source_row")
    result = {
        "contract": contract(X, y), "model": "random_forest", "target_poison_recall": TARGET_RECALL,
        "model_parameters": candidates(X)["random_forest"].named_steps["model"].get_params(),
        "selection_rule": "Maximum specificity with calibration recall >= 0.90; ties use largest threshold; predict p when score >= threshold.",
        "protocol": "StratifiedKFold 5 outer folds; 3 inner folds per outer training set; shuffle=True and seed=42 throughout. One outer held-out score per row.",
        "selected_development_threshold": selected,
        "selected_threshold_metrics_apparent": operating_metrics(y, p, selected),
        "nested_policy_metrics": operating_metrics(y, p, predictions=predictions.policy_prediction),
        "fixed_0_5_same_outer_scores": operating_metrics(y, p, .5),
        "outer_folds": audit,
        "test_model_predictions_computed": False,
        "limitations": [
            "The 90% recall target is a coursework preference inherited from task 11, not a lecturer mandate or real-world safety standard.",
            "Outer folds evaluate the threshold-selection procedure, not the single threshold selected afterwards using all development OOF scores.",
            "RF family and preparation were selected using earlier development results; this is not nested evaluation of all model-selection decisions.",
            "Inner models train on fewer rows; their score distributions and fitted thresholds may shift. Scores are not calibrated risk estimates.",
            "No independent final-test or real-world performance claim; the existing demo remains at its provisional 0.5 threshold.",
        ],
        "source_sha256": {name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
                          for name in ["mushrooms.py", "mushroom_workflow.py", "mushroom_error_analysis.py"]},
    }
    write_json(REPORTS / "threshold_run.json", result)
    print(f"Selected development threshold={selected:.8f}; nested policy recall={result['nested_policy_metrics']['poison_recall']:.6f}", flush=True)
    return result, folds, pd.DataFrame(rows)


if __name__ == "__main__":
    run_error_analysis()
