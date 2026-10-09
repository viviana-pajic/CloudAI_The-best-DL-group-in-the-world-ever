"""Bounded PyCaret screening, then comparison on the shared development folds.

The external PyCaret holdout is the ORIGINAL development validation partition,
never the final test. No holdout prediction is requested. Screening uses 3,000
rows; its scores are kept separate from the common 4,000-row evaluation.
"""
import os
from pathlib import Path
import time

# Keep optional-library caches/logs out of users' global configuration directories.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".tools/matplotlib"))

import pandas as pd
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from mushrooms import ROOT, cv_splitter
from mushroom_workflow import (
    REPORTS, candidates, contract, prepare, record_evaluation, write_json,
)

SCREEN_MODELS = ["logistic_regression", "random_forest", "extra_trees", "hist_gradient_boosting"]


def run_automl():
    X, y, splits = prepare()
    training = X.loc[splits["train"]].assign(class_label=y.loc[splits["train"]])
    holdout = X.loc[splits["validation"]].assign(class_label=y.loc[splits["validation"]])
    pipelines = candidates(X)
    experiment = ClassificationExperiment()
    experiment.setup(
        data=training, test_data=holdout, target="class_label", index=False,
        preprocess=False, fold_strategy=cv_splitter(), session_id=42,
        n_jobs=1, use_gpu=False, html=False, verbose=False, system_log=False,
        log_experiment=False, memory=False,
    )
    experiment.add_metric("ap", "AP", average_precision_score,
                          target="pred_proba", greater_is_better=True)
    # A complete sklearn Pipeline is an estimator supported by compare_models.
    # Thus imputers/encoders/scalers are fitted separately inside every CV fit.
    # Native PyCaret preprocessing is disabled to avoid double preprocessing.
    print("PyCaret screening: four complete pipelines, 3,000 rows, 5 folds x 2 repeats.", flush=True)
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        experiment.compare_models(
            include=[pipelines[name] for name in SCREEN_MODELS], sort="AP", n_select=2,
            fold=cv_splitter(), round=8, errors="raise", verbose=False,
        )
    screening = experiment.pull().copy()
    # With estimator objects PyCaret assigns table IDs by include-list position,
    # before sorting. Map those IDs, never the leaderboard row order.
    screening.insert(0, "model", [SCREEN_MODELS[int(i)] for i in screening.index])
    screening.to_csv(REPORTS / "automl_screening.csv", index=False)
    if len(screening) != len(SCREEN_MODELS) or screening.AP.isna().any():
        raise RuntimeError("A bounded screening candidate is missing or has no AP score.")
    write_json(REPORTS / "automl_run.json", {
        "contract": contract(X, y), "tool": "PyCaret ClassificationExperiment.compare_models",
        "screening_rows": len(training), "administrative_holdout_rows": len(holdout),
        "administrative_holdout": "Original development validation partition; not scored or used for ranking.",
        "preprocessing": "Native PyCaret preprocessing disabled; complete shared sklearn pipelines passed as estimators.",
        "models": SCREEN_MODELS, "ranking": "mean fold AP", "repeats": 2, "folds": 5,
        "screening_seconds": time.perf_counter() - start,
        "limitations": "PyCaret may downcast numeric dtypes during setup. Screening uses 3,000 rows; "
                       "use the separate shared 4,000-row comparison for like-for-like results. "
                       "This is model screening, not hyperparameter tuning or final selection.",
        "model_parameters": {name: pipelines[name].named_steps["model"].get_params() for name in SCREEN_MODELS},
    })
    print(screening[["model", "AP", "AUC", "Recall", "Prec.", "F1"]].to_string(index=False), flush=True)
    print("Now evaluate the two additional families on the same 4,000-row folds as the baselines.", flush=True)
    baseline_path = REPORTS / "baseline_cv.csv"
    if not baseline_path.exists():
        raise RuntimeError("Run mushroom_workflow.py baseline before the AutoML comparison.")
    baseline_run = pd.read_json(REPORTS / "baseline_run.json", typ="series")["contract"]
    if baseline_run != contract(X, y):
        raise RuntimeError("Baseline contract/environment differs. Reproduce baselines before combining scores.")
    additional = record_evaluation(["extra_trees", "hist_gradient_boosting"], "candidates", X, y)
    comparison = pd.concat([pd.read_csv(baseline_path), additional], ignore_index=True)
    comparison = comparison.sort_values("cv_average_precision", ascending=False)
    comparison.to_csv(REPORTS / "comparison.csv", index=False)
    print(comparison[["model", "cv_average_precision", "cv_roc_auc", "oof_poison_recall"]].to_string(index=False))
    return screening, comparison


if __name__ == "__main__":
    run_automl()
