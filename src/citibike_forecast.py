"""Next-day ride-count forecasting with chronological development evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'citibike/evaluation_plan.json'
REPORTS = ROOT / 'reports/citibike'
PROCESSED = ROOT / 'data/processed/citibike'
MODELS = ROOT / 'models/citibike'
SEED = 42
CALENDAR = ['weekday_sin', 'weekday_cos', 'year_sin', 'year_cos', 'weekend']
LAGS = [f'lag_{n}' for n in [1, 2, 3, 7, 14, 21, 28]]
COMPACT = CALENDAR + ['lag_1', 'lag_7', 'lag_14', 'mean_7', 'mean_28']
FULL = CALENDAR + LAGS + ['mean_3', 'mean_7', 'mean_14', 'mean_28', 'std_7',
                        'weekly_change', 'recent_change', 'holiday', 'before_holiday', 'after_holiday']
FEATURE_SETS = {'compact': COMPACT, 'full': FULL}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def environment():
    import platform
    packages = ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'xgboost', 'flaml', 'lightgbm']
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'packages': versions}


def read_plan():
    return json.loads(PLAN.read_text(encoding='utf-8'))


def load_daily():
    path = ROOT / 'data/processed/citibike_daily.csv'
    manifest = json.loads((ROOT / 'citibike/archive_manifest.json').read_text(encoding='utf-8'))
    snapshot = ROOT / 'citibike/development_daily.csv'
    version = json.loads((ROOT / 'citibike/development_daily_version.json').read_text(encoding='utf-8'))
    if (hashlib.sha256(snapshot.read_bytes()).hexdigest() != version['sha256']
            or version['original_annual_daily_sha256'] != manifest['daily_sha256']):
        raise ValueError('The development snapshot provenance changed.')
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest['daily_sha256']:
            raise ValueError('Daily counts differ from the audited acquisition. Rerun notebook 00.')
        daily = pd.read_csv(path, index_col='date', parse_dates=['date'])
        expected = pd.read_csv(snapshot, index_col='date', parse_dates=['date'])
        pd.testing.assert_frame_equal(daily.loc[:'2023-09-30'], expected)
    else:
        # A clean clone can model the audited development aggregate without a 2 GB download.
        daily = pd.read_csv(snapshot, index_col='date', parse_dates=['date'])
        digest = manifest['daily_sha256']
    validate_daily(daily)
    return daily, digest


def validate_daily(daily):
    if not isinstance(daily.index, pd.DatetimeIndex) or daily.index.tz is not None or daily.empty:
        raise ValueError('Use nonempty local calendar dates without a timezone.')
    if daily.index.has_duplicates or not daily.index.equals(pd.date_range(daily.index.min(), daily.index.max())):
        raise ValueError('Daily dates must be sorted, unique and consecutive.')
    if not daily.index.equals(daily.index.normalize()):
        raise ValueError('Use calendar dates, not times within a day.')
    if list(daily.columns) != ['trips']:
        raise ValueError('Expected the audited date/trips table.')
    values = daily.trips.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError('Daily ride counts must be finite nonnegative integers.')


def calendar_features(index):
    frame = pd.DataFrame(index=index)
    day = index.dayofweek
    year_fraction = (index.dayofyear - 1) / np.where(index.is_leap_year, 366, 365)
    frame['weekday_sin'] = np.sin(2 * np.pi * day / 7)
    frame['weekday_cos'] = np.cos(2 * np.pi * day / 7)
    frame['year_sin'] = np.sin(2 * np.pi * year_fraction)
    frame['year_cos'] = np.cos(2 * np.pi * year_fraction)
    frame['weekend'] = (day >= 5).astype(int)
    holidays = USFederalHolidayCalendar().holidays(index.min() - pd.Timedelta(days=1), index.max() + pd.Timedelta(days=1))
    frame['holiday'] = index.isin(holidays).astype(int)
    frame['before_holiday'] = (index + pd.Timedelta(days=1)).isin(holidays).astype(int)
    frame['after_holiday'] = (index - pd.Timedelta(days=1)).isin(holidays).astype(int)
    return frame


def feature_table(daily):
    """Each row's ride information ends at t-1. The target is never a feature."""
    validate_daily(daily)
    features = calendar_features(daily.index)
    prior = daily.trips.shift(1)
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        features[f'lag_{lag}'] = daily.trips.shift(lag)
    for window in [3, 7, 14, 28]:
        features[f'mean_{window}'] = prior.rolling(window, min_periods=window).mean()
    features['std_7'] = prior.rolling(7, min_periods=7).std(ddof=0)
    features['weekly_change'] = features.lag_7 - features.lag_14
    features['recent_change'] = features.lag_1 - features.lag_7
    return features[FULL]


def development_data():
    daily, digest = load_daily()
    plan = read_plan()
    # Discard reserved targets before forming any development feature or model input.
    development = daily.loc[:plan['validation_end']].copy()
    X = feature_table(development).loc[plan['training_start']:]
    y = development.trips.loc[X.index]
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError('Not enough history for the documented feature window.')
    training = X.loc[:plan['training_end']]
    validation = X.loc[plan['validation_start']:plan['validation_end']]
    if len(training) != 153 or len(validation) != 92 or len(X) != 245:
        raise ValueError('The documented 2023 development membership changed.')
    return X, y, training.index, validation.index, digest


def chronological_cv():
    settings = read_plan()['search_cv']
    return TimeSeriesSplit(n_splits=settings['n_splits'], test_size=settings['test_size'], gap=settings['gap'])


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not len(actual) or not np.isfinite(predicted).all():
        raise ValueError('Expected finite predictions aligned to the actual counts.')
    return {'mae': float(mean_absolute_error(actual, predicted)),
            'rmse': float(root_mean_squared_error(actual, predicted)),
            'wape': float(np.abs(actual - predicted).sum() / actual.sum()),
            'bias': float((predicted - actual).mean()), 'days': len(actual)}


class SeasonalNaive(RegressorMixin, BaseEstimator):
    def __init__(self, kind='lag_7'):
        self.kind = kind

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X):
        if self.kind == 'training_mean':
            return np.full(len(X), self.mean_)
        return X[self.kind].to_numpy(dtype=float)


class ForecastRegressor(RegressorMixin, BaseEstimator):
    """Select known features; optionally learn the change from last week's count."""
    def __init__(self, model=None, feature_set='full', seasonal_residual=False):
        self.model = model
        self.feature_set = feature_set
        self.seasonal_residual = seasonal_residual

    def fit(self, X, y):
        self.columns_ = FEATURE_SETS[self.feature_set]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.model_ = clone(self.model)
        target = np.asarray(y) - X.lag_7.to_numpy() if self.seasonal_residual else y
        self.model_.fit(X[self.columns_], target)
        return self

    def predict(self, X):
        prediction = self.model_.predict(X[self.columns_])
        if self.seasonal_residual:
            prediction = prediction + X.lag_7.to_numpy()
        return np.maximum(0., prediction)


def contract(digest):
    plan = read_plan()
    X, y, train, validation, _ = development_data()
    folds = []
    for fit, check in chronological_cv().split(X.loc[train]):
        dates = train
        folds.append({'training_start': str(dates[fit[0]].date()), 'training_end': str(dates[fit[-1]].date()),
                      'validation_start': str(dates[check[0]].date()), 'validation_end': str(dates[check[-1]].date()),
                      'training_rows': len(fit), 'validation_rows': len(check)})
    snapshot_version = json.loads((ROOT / 'citibike/development_daily_version.json').read_text(encoding='utf-8'))
    return {'version': plan['version'], 'daily_sha256': digest,
            'development_snapshot_sha256': snapshot_version['sha256'], 'plan': plan, 'feature_columns': FULL,
            'folds': folds, 'environment': environment(), 'test_model_predictions_computed': False}


def source_fingerprints():
    names = ['citibike_forecast.py']
    return {name: hashlib.sha256((ROOT / 'src' / name).read_bytes()).hexdigest() for name in names}


def prepare():
    X, y, training, validation, digest = development_data()
    write_json(REPORTS / 'contract.json', contract(digest))
    write_json(ROOT / 'citibike/forecast_split_manifest.json', {
        'daily_sha256': digest, 'training_dates': training.strftime('%Y-%m-%d').tolist(),
        'validation_dates': validation.strftime('%Y-%m-%d').tolist(),
        'reserved_test_dates': pd.date_range('2023-10-01', '2023-12-31').strftime('%Y-%m-%d').tolist(),
        'test_model_predictions_computed': False,
    })
    PROCESSED.mkdir(parents=True, exist_ok=True)
    X.assign(target_rides=y).to_csv(PROCESSED / 'development.csv', index_label='date')
    print(f'Prepared {len(training)} training and {len(validation)} validation dates; 92 final-test dates reserved.')
    return X, y, training, validation, digest


def record_model(name, estimator, search=None, extra=None):
    X, y, train, validation, digest = development_data()
    cv_rows = []
    with threadpool_limits(limits=4):
        for fold, (fit, check) in enumerate(chronological_cv().split(X.loc[train])):
            model = clone(estimator).fit(X.loc[train].iloc[fit], y.loc[train].iloc[fit])
            predicted = model.predict(X.loc[train].iloc[check])
            cv_rows.append({'fold': fold, **metrics(y.loc[train].iloc[check], predicted)})
        fitted = clone(estimator).fit(X.loc[train], y.loc[train])
        prediction = fitted.predict(X.loc[validation])
    result = {'model': name, 'validation': metrics(y.loc[validation], prediction),
              'cv_mean_mae': float(np.mean([r['mae'] for r in cv_rows])), 'cv_folds': cv_rows,
              'parameters': estimator.get_params(deep=False), 'search': search,
              'contract': contract(digest), 'source_sha256': source_fingerprints(), 'extra': extra or {}}
    # Estimators are recorded separately as configuration text, not JSON objects.
    result['parameters'] = {key: (str(value) if isinstance(value, BaseEstimator) else value)
                            for key, value in result['parameters'].items()}
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS / f'{name}_run.json', result)
    pd.DataFrame({'date': validation, 'actual': y.loc[validation].to_numpy(), 'predicted': prediction,
                  'error': prediction - y.loc[validation].to_numpy()}).to_csv(REPORTS / f'{name}_validation.csv', index=False)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, MODELS / f'{name}_configuration.joblib')
    print(name, json.dumps(result['validation']), flush=True)
    return result


def run_baselines():
    records = []
    for kind in ['training_mean', 'lag_1', 'lag_7', 'mean_7']:
        records.append(record_model(kind, SeasonalNaive(kind)))
    records.append(record_model('ridge_reference', ForecastRegressor(make_pipeline(StandardScaler(), Ridge(alpha=10.)), 'compact')))
    return pd.DataFrame([{'model': r['model'], **r['validation'], 'cv_mean_mae': r['cv_mean_mae']} for r in records])


def gpu_estimator():
    from xgboost import XGBRegressor
    return XGBRegressor(tree_method='hist', device='cuda', objective='reg:squarederror',
                        random_state=SEED, n_jobs=2, colsample_bytree=.9, verbosity=0)


def run_tuning(family):
    X, y, train, _, _ = development_data()
    if family == 'ridge':
        estimator = ForecastRegressor(make_pipeline(StandardScaler(), Ridge()), 'full')
        grid = {'model__ridge__alpha': [.01, .1, 1., 10., 100., 1000., 10000.],
                'feature_set': ['compact', 'full'], 'seasonal_residual': [False, True]}
        workers = 4
    elif family == 'xgboost':
        estimator = ForecastRegressor(gpu_estimator(), 'full')
        grid = {'model__max_depth': [2, 3, 5], 'model__n_estimators': [100, 300],
                'model__learning_rate': [.03, .1], 'model__min_child_weight': [3, 10],
                'model__reg_lambda': [1., 10.], 'model__subsample': [.8, 1.],
                'seasonal_residual': [False, True]}
        workers = 1  # One concurrent GPU fit prevents competing CUDA jobs.
    else:
        raise ValueError('Unknown tuning family.')
    from sklearn.model_selection import ParameterGrid
    count = len(list(ParameterGrid(grid)))
    print(f'{family}: {count} configurations, {count * 3} chronological CV fits.', flush=True)
    started = time.perf_counter()
    search = GridSearchCV(estimator, grid, scoring='neg_mean_absolute_error', cv=chronological_cv(),
                          n_jobs=workers, refit=True, error_score='raise', return_train_score=True, verbose=1)
    with threadpool_limits(limits=4):
        search.fit(X.loc[train], y.loc[train])
    pd.DataFrame(search.cv_results_).drop(columns='params').to_csv(REPORTS / f'{family}_grid.csv', index=False)
    actual_device = None
    if family == 'xgboost':
        actual_device = json.loads(search.best_estimator_.model_.get_booster().save_config())['learner']['generic_param']['device']
        if not actual_device.startswith('cuda'):
            raise RuntimeError('The requested GPU search silently fell back to CPU.')
    details = {'method': 'GridSearchCV', 'configurations': count, 'cv_fits': count * 3,
               'best_parameters': search.best_params_, 'best_cv_mae': -float(search.best_score_),
               'elapsed_seconds': time.perf_counter() - started, 'actual_fitted_device': actual_device,
               'validation_used_for_search': False}
    return record_model(f'{family}_tuned', clone(search.best_estimator_), details)


def run_automl(seconds=180):
    from flaml import AutoML
    X, y, train, _, _ = development_data()
    automl = AutoML()
    print(f'FLAML: chronological CV, four model families, up to {seconds}s / 80 trials.', flush=True)
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        automl.fit(X_train=X.loc[train], y_train=y.loc[train], task='regression', metric='mae',
                   estimator_list=['lgbm', 'rf', 'extra_tree', 'xgboost'], eval_method='cv',
                   split_type=chronological_cv(), time_budget=seconds, max_iter=80, n_jobs=4,
                   seed=SEED, sample=False, ensemble=False, retrain_full=True, skip_transform=True,
                   log_file_name=str(REPORTS / 'automl_search.jsonl'), log_type='all', verbose=2,
                   mlflow_logging=False)
    details = {'tool': 'FLAML AutoML', 'estimator_families': ['lgbm', 'rf', 'extra_tree', 'xgboost'],
               'best_family': automl.best_estimator, 'best_parameters': automl.best_config,
               'best_cv_mae': float(automl.best_loss), 'elapsed_seconds': time.perf_counter() - started,
               'budget_seconds': seconds, 'max_trials': 80, 'validation_used_for_search': False,
               'scope': 'Automatic configuration search on training dates with chronological folds; not an exhaustive search.'}
    return record_model('automl', ForecastRegressor(clone(automl.model.estimator), 'full'), details)


def comparison():
    rows = []
    for path in REPORTS.glob('*_run.json'):
        run = json.loads(path.read_text(encoding='utf-8'))
        if 'validation' in run:
            if run['contract'] != contract(run['contract']['daily_sha256']):
                raise ValueError(f'Experiment contract changed: {path.name}')
            for name, digest in run['source_sha256'].items():
                if hashlib.sha256((ROOT / 'src' / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Experiment source changed: {name}; rerun the experiment.')
            rows.append({'model': run['model'], **run['validation'], 'cv_mean_mae': run['cv_mean_mae']})
    table = pd.DataFrame(rows).sort_values(['mae', 'model']).reset_index(drop=True)
    naive = table.set_index('model').loc['lag_7', 'mae']
    table['mae_improvement_over_lag_7'] = 1 - table.mae / naive
    table.to_csv(REPORTS / 'comparison.csv', index=False)
    return table


def run_errors():
    table = comparison()
    winner = table.iloc[0]['model']
    predictions = pd.read_csv(REPORTS / f'{winner}_validation.csv', parse_dates=['date']).set_index('date')
    predictions['absolute_error'] = predictions.error.abs()
    predictions['weekday'] = predictions.index.day_name()
    predictions['month'] = predictions.index.month
    predictions.nlargest(10, 'absolute_error').to_csv(REPORTS / 'largest_errors.csv', index_label='date')
    groups = []
    for kind in ['weekday', 'month']:
        for label, part in predictions.groupby(kind):
            groups.append({'group': kind, 'value': str(label), **metrics(part.actual, part.predicted)})
    pd.DataFrame(groups).to_csv(REPORTS / 'error_groups.csv', index=False)
    # Planned feature ablation: same selected model configuration, same train/validation dates.
    estimator = joblib.load(MODELS / f'{winner}_configuration.joblib')
    ablation = []
    if isinstance(estimator, ForecastRegressor):
        for feature_set in ['compact', 'full']:
            record = record_model(f'{winner}_{feature_set}_ablation', clone(estimator).set_params(feature_set=feature_set))
            ablation.append({'feature_set': feature_set, **record['validation'], 'cv_mean_mae': record['cv_mean_mae']})
    pd.DataFrame(ablation).to_csv(REPORTS / 'feature_ablation.csv', index=False)
    write_json(REPORTS / 'error_analysis.json', {
        'initial_selected_model': winner, 'scope': 'Development validation; diagnostics and ablation may influence development selection.',
        'test_model_predictions_computed': False,
        'limitations': 'No observed same-day weather or target is added after inspecting errors. Errors are not proof of weather causation. Final-test performance remains unknown.',
    })
    return predictions, pd.DataFrame(groups), pd.DataFrame(ablation)


def select_candidate():
    table = comparison()
    selected = table.iloc[0]['model']
    estimator = joblib.load(MODELS / f'{selected}_configuration.joblib')
    X, y, train, validation, digest = development_data()
    with threadpool_limits(limits=4):
        fitted = clone(estimator).fit(X, y)
    # A GPU-trained XGBoost model can use CPU inference without needing a GPU host.
    if isinstance(fitted, ForecastRegressor) and hasattr(fitted.model_, 'get_booster'):
        fitted.model_.set_params(device='cpu')
    metadata = {'name': selected, 'status': 'Local development candidate; AWS comparison and final-test evaluation pending',
                'contract_version': read_plan()['version'], 'daily_sha256': digest,
                'training_start': str(X.index.min().date()), 'training_end': str(X.index.max().date()),
                'training_rows': len(X), 'forecast_horizon_days': 1, 'history_days': 28,
                'feature_columns': FULL, 'environment': environment(),
                'metric_scope': 'July-September rolling one-day development validation before all-development refit',
                'metrics': {key: float(table.iloc[0][key]) for key in ['mae', 'rmse', 'wape', 'bias', 'mae_improvement_over_lag_7']},
                'final_test_evaluated': False, 'test_dates_reserved': 92,
                'limitations': 'Historical 2023 citywide recorded trips; no live feed or future weather. Development selection is optimistic. This is not a final release.'}
    bundle = {'pipeline': fitted, 'metadata': metadata}
    joblib.dump(bundle, ROOT / 'models/citibike_candidate.joblib')
    reloaded = joblib.load(ROOT / 'models/citibike_candidate.joblib')
    np.testing.assert_allclose(fitted.predict(X.iloc[:3]), reloaded['pipeline'].predict(X.iloc[:3]))
    write_json(REPORTS / 'candidate.json', metadata)
    print('Selected:', selected, 'validation MAE:', metadata['metrics']['mae'], flush=True)
    return table, metadata


def prediction_row(history, forecast_date):
    """Use exactly 28 consecutive observed dates ending the day before forecast."""
    date = pd.Timestamp(forecast_date)
    if date.tz is not None or date != date.normalize():
        raise ValueError('Use an ISO calendar forecast date.')
    validate_daily(history)
    expected = pd.date_range(date - pd.Timedelta(days=28), date - pd.Timedelta(days=1))
    if not history.index.equals(expected):
        raise ValueError('Provide exactly 28 consecutive daily totals ending the day before the forecast.')
    # The placeholder only creates the forecast-date row; shift(1) excludes its value.
    extended = pd.concat([history, pd.DataFrame({'trips': [0]}, index=pd.DatetimeIndex([date]))])
    return feature_table(extended).iloc[[-1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'baseline', 'automl', 'ridge', 'xgboost', 'errors', 'select'])
    parser.add_argument('--automl-seconds', type=int, default=180)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'baseline':
        print(run_baselines().to_string(index=False))
    elif args.command == 'automl':
        run_automl(args.automl_seconds)
    elif args.command in ['ridge', 'xgboost']:
        run_tuning(args.command)
    elif args.command == 'errors':
        run_errors()
    elif args.command == 'select':
        print(select_candidate()[0].to_string(index=False))


if __name__ == '__main__':
    # Import the stable module before creating objects that will be serialized.
    import citibike_forecast as workflow
    workflow.main()
