"""Run an additional CPU XGBoost search inside an approved SageMaker notebook."""
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor
from citibike_forecast import (REPORTS, ForecastRegressor, chronological_cv,
                               development_data, record_model)


def train_on_aws():
    metadata_file = Path('/opt/ml/metadata/resource-metadata.json')
    if not metadata_file.exists():
        raise RuntimeError('This prepared task must run in a SageMaker notebook, not on BESTIJA.')
    resource = json.loads(metadata_file.read_text())
    arn = resource.get('ResourceArn', '')
    if not re.match(r'^arn:[^:]+:sagemaker:[^:]+:\d{12}:', arn):
        raise RuntimeError('SageMaker resource metadata could not be verified.')
    provenance = {
        'service': 'Amazon SageMaker notebook',
        'resource_arn_redacted': re.sub(r':\d{12}:', ':<account>:', arn),
        'proof': 'Metadata read from the executing SageMaker notebook; retain the console record separately.',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    X, y, train, _, _ = development_data()
    estimator = ForecastRegressor(XGBRegressor(tree_method='hist', device='cpu',
        objective='reg:squarederror', random_state=42, n_jobs=2, colsample_bytree=.9))
    grid = {'model__max_depth': [2, 3, 5], 'model__n_estimators': [100, 300],
            'model__learning_rate': [.03, .1], 'model__min_child_weight': [3, 10],
            'model__reg_lambda': [1., 10.], 'model__subsample': [.8, 1.],
            'seasonal_residual': [False, True]}
    search = GridSearchCV(estimator, grid, cv=chronological_cv(), scoring='neg_mean_absolute_error',
                          n_jobs=1, error_score='raise', return_train_score=True)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        search.fit(X.loc[train], y.loc[train])
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search.cv_results_).drop(columns='params').to_csv(REPORTS / 'aws_xgboost_grid.csv', index=False)
    details = {'method': 'GridSearchCV on AWS CPU', 'configurations': len(list(ParameterGrid(grid))),
               'cv_fits': len(list(ParameterGrid(grid))) * 3, 'best_parameters': search.best_params_,
               'best_cv_mae': -float(search.best_score_), 'elapsed_seconds': time.perf_counter() - started,
               'validation_used_for_search': False}
    return record_model('aws_xgboost_tuned', clone(search.best_estimator_), details, provenance)


if __name__ == '__main__':
    train_on_aws()
