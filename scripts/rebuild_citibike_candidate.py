"""Rebuild the selected configuration on CPU for a clean clone or CI."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import json
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from citibike_additional import SeasonalAverage
from citibike_forecast import (FULL, ForecastRegressor, SeasonalNaive, development_data,
                               environment, metrics, prediction_row, read_plan, write_json)


def configured_estimator(config):
    kind = config['model_type']
    if kind == 'xgboost':
        from xgboost import XGBRegressor
        parameters = dict(config['model_parameters'], device='cpu')
        model = XGBRegressor(**parameters)
    elif kind == 'random_forest':
        model = RandomForestRegressor(**config['model_parameters'])
    elif kind == 'ridge':
        model = make_pipeline(StandardScaler(), Ridge(alpha=config['alpha']))
    elif kind == 'seasonal_average':
        return SeasonalAverage(**config['model_parameters'])
    elif kind == 'naive':
        return SeasonalNaive(**config['model_parameters'])
    else:
        raise ValueError('Unsupported selected model type.')
    return ForecastRegressor(model, config['feature_set'], config['seasonal_residual'])


def rebuild():
    config = json.loads((ROOT / 'citibike/selected_configuration.json').read_text())
    if config['contract_version'] != read_plan()['version']:
        raise ValueError('Selected configuration and evaluation plan differ.')
    X, y, train, validation, digest = development_data()
    estimator = configured_estimator(config)
    with threadpool_limits(limits=4):
        checked = clone(estimator).fit(X.loc[train], y.loc[train])
        measured = metrics(y.loc[validation], checked.predict(X.loc[validation]))
        baseline = metrics(y.loc[validation], X.loc[validation, 'lag_7'])
        if measured['mae'] >= baseline['mae']:
            raise ValueError('The rebuilt candidate no longer beats the agreed weekly baseline; review it before promotion.')
        fitted = clone(estimator).fit(X, y)
    metadata = {
        'name': config['name'], 'status': 'CPU rebuild of the local development candidate; AWS/final test pending',
        'contract_version': config['contract_version'], 'daily_sha256': digest,
        'training_start': str(X.index.min().date()), 'training_end': str(X.index.max().date()),
        'training_rows': len(X), 'history_days': 28, 'forecast_horizon_days': 1,
        'feature_columns': FULL, 'environment': environment(),
        'metric_scope': 'Fresh July-September validation of the fixed configuration, before all-development CPU refit',
        'metrics': {**measured, 'mae_improvement_over_lag_7': 1 - measured['mae'] / baseline['mae']},
        'final_test_evaluated': False, 'test_dates_reserved': 92,
        'limitations': 'Recorded 2023 rides; no live feed, future weather or final-test performance.'}
    path = ROOT / 'models/citibike_candidate.joblib'
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': fitted, 'metadata': metadata}, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(fitted.predict(X.iloc[:3]), loaded['pipeline'].predict(X.iloc[:3]))
    write_json(ROOT / 'models/citibike_candidate.metadata.json', metadata)
    print('Rebuilt on CPU:', metadata['name'], 'validation MAE:', measured['mae'], flush=True)
    return metadata


if __name__ == '__main__':
    rebuild()
