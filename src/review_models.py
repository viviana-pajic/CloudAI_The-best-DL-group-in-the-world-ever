"""Validated inputs for the frozen review models."""
from pathlib import Path
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import sys
from threading import RLock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'mushrooms'), str(ROOT / 'citibike'), str(ROOT)]

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.review_features import build_feature_frame
from weather_search import add_weather_definitions, BASE_WEATHER

add_weather_definitions()
NUMERIC = ['cap-diameter', 'stem-height', 'stem-width']
CATEGORICAL = ['spore-print-color', 'gill-color', 'habitat', 'season', 'ring-type', 'cap-shape', 'stem-surface']
THRESHOLD = 0.0398015873015873
PREDICTION_LOCK = RLock()
MODEL_LOAD_LOCK = RLock()
BIKES = {
    'counts': {'file': 'citibike_counts_xgboost.joblib', 'name': 'Counts-only XGBoost', 'weather_days': 0, 'mae': 14761.370650249979},
    'weather_cpu': {'file': 'citibike_weather_cpu.joblib', 'name': '75% SVR / 25% histogram boosting', 'weather_days': 8, 'mae': 13494.129871783016},
    'weather_gru': {'file': 'citibike_weather_gru.joblib', 'name': '75% GRU / 25% CatBoost', 'weather_days': 1, 'mae': 13263.41703622252},
}

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def load_model(filename):
    with MODEL_LOAD_LOCK:
        return cached_model(filename)

@lru_cache(maxsize=4)
def cached_model(filename):
    for package, wanted in {'numpy': '2.5.3', 'pandas': '3.0.6', 'scikit-learn': '1.9.1', 'joblib': '1.6.0'}.items():
        if importlib.metadata.version(package) != wanted:
            raise RuntimeError(f'Install requirements-review.txt: {package} must be {wanted}.')
    path = ROOT / 'models' / filename
    manifest = ROOT / 'models/model_manifest.json'
    if manifest.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != read_json(manifest)['sha256'][filename]:
        raise RuntimeError(f'Model file changed: {filename}')
    return joblib.load(path)

def finite_number(value, name, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f'{name} must be a number.')
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f'{name} must be finite' + (' and nonnegative.' if nonnegative else '.'))
    return value

def mushroom_frame(values):
    if not isinstance(values, dict) or set(values) != set(NUMERIC + CATEGORICAL):
        raise ValueError('Supply all ten mushroom fields. Use null for a missing value.')
    cleaned = {}
    for name in NUMERIC:
        cleaned[name] = np.nan if values[name] is None else finite_number(values[name], name, True)
    for name in CATEGORICAL:
        value = values[name]
        if value is not None and (not isinstance(value, str) or len(value) > 32):
            raise ValueError(f'{name} must be a short category code or null.')
        cleaned[name] = np.nan if value is None or value == '' else value
    return pd.DataFrame([cleaned], columns=NUMERIC + CATEGORICAL)

def predict_mushroom(payload):
    if not isinstance(payload, dict) or set(payload) != {'features', 'policy'}:
        raise ValueError('Supply features and policy.')
    policy = payload['policy']
    if policy not in ('cautious', 'default'):
        raise ValueError('Choose cautious or default.')
    model = load_model('mushroom_extra_trees.joblib')
    X = mushroom_frame(payload['features'])
    with PREDICTION_LOCK, threadpool_limits(limits=1):
        score = float(model.predict_proba(X)[0, list(model.classes_).index(1)])
    manifest = read_json(ROOT / 'models/model_manifest.json')
    threshold = manifest['mushroom_cautious_threshold'] if policy == 'cautious' else .5
    flag = score >= threshold
    return {'model': 'Extra Trees', 'research_score': score, 'threshold': threshold,
            'flag_as_potentially_poisonous': flag,
            'decision': 'Flagged as potentially poisonous' if flag else 'Not flagged by this rule',
            'note': 'Coursework prediction. This does not establish whether a mushroom is safe to eat.'}

def prediction_date(value):
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError('Use a prediction date in YYYY-MM-DD format.')
    try:
        date = pd.Timestamp(value)
    except (ValueError, TypeError):
        raise ValueError('Invalid prediction date.') from None
    if pd.isna(date) or date.strftime('%Y-%m-%d') != value or date.tzinfo is not None:
        raise ValueError('Use a prediction date in YYYY-MM-DD format.')
    return date

def bike_frame(payload):
    if not isinstance(payload, dict):
        raise ValueError('Supply a JSON object.')
    if set(payload) != {'model', 'prediction_date', 'prior_counts', 'forecasts'}:
        raise ValueError('Supply model, prediction_date, prior_counts and forecasts.')
    key = payload['model']
    if key not in BIKES:
        raise ValueError('Unknown bike model.')
    date = prediction_date(payload['prediction_date'])
    counts = payload['prior_counts']
    if not isinstance(counts, list) or len(counts) != 28:
        raise ValueError('Supply 28 completed daily totals, oldest first and newest last.')
    counts = [finite_number(v, 'Daily total', True) for v in counts]
    if any(v != int(v) for v in counts) or sum(counts) == 0:
        raise ValueError('Daily totals must be whole numbers, with some rides in the 28-day history.')
    index = pd.date_range(date - pd.Timedelta(days=28), date)
    daily = pd.DataFrame({'trips': counts + [0.]}, index=index)
    daily.index.name = 'date'
    X = build_feature_frame(daily).iloc[[-1]].copy()
    required = BIKES[key]['weather_days']
    forecasts = payload['forecasts']
    if not isinstance(forecasts, list) or len(forecasts) != required:
        raise ValueError(f'This model needs {required} dated forecast records.')
    if required:
        expected = pd.date_range(date-pd.Timedelta(days=required-1), date)
        rows = []
        for record, expected_date in zip(forecasts, expected):
            if not isinstance(record, dict) or set(record) != set(BASE_WEATHER + ['date']):
                raise ValueError('Every forecast needs a date and all six documented weather fields.')
            if prediction_date(record['date']) != expected_date:
                raise ValueError('Forecast dates must be consecutive and end on the prediction date.')
            row = {n: finite_number(record[n], n, n in BASE_WEATHER[3:]) for n in BASE_WEATHER}
            if not row[BASE_WEATHER[1]] <= row[BASE_WEATHER[0]] <= row[BASE_WEATHER[2]]:
                raise ValueError('Forecast temperatures must follow min <= mean <= max.')
            if row[BASE_WEATHER[4]] > row[BASE_WEATHER[5]]:
                raise ValueError('Mean forecast wind cannot exceed maximum forecast wind.')
            rows.append(row)
        f = pd.DataFrame(rows, index=expected)
        for name in BASE_WEATHER:
            X[name] = f.iloc[-1][name]
        X['forecast_temperature_range'] = f.iloc[-1][BASE_WEATHER[2]] - f.iloc[-1][BASE_WEATHER[1]]
        X['forecast_temperature_weekend'] = f.iloc[-1][BASE_WEATHER[0]] * int(date.dayofweek >= 5)
        X['forecast_warm_degree'] = max(0., f.iloc[-1][BASE_WEATHER[0]] - 18.)
        X['forecast_cold_degree'] = max(0., 10. - f.iloc[-1][BASE_WEATHER[0]])
        X['forecast_rain_log'] = np.log1p(f.iloc[-1][BASE_WEATHER[3]])
        if required == 8:
            X['forecast_temperature_change_1'] = f.iloc[-1][BASE_WEATHER[0]] - f.iloc[-2][BASE_WEATHER[0]]
            X['forecast_temperature_change_7'] = f.iloc[-1][BASE_WEATHER[0]] - f.iloc[0][BASE_WEATHER[0]]
            X['forecast_rain_change_7'] = f.iloc[-1][BASE_WEATHER[3]] - f.iloc[0][BASE_WEATHER[3]]
            X['forecast_wind_change_7'] = f.iloc[-1][BASE_WEATHER[4]] - f.iloc[0][BASE_WEATHER[4]]
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError('This history cannot produce finite features. Check the daily totals.')
    return key, X

def predict_bike(payload):
    key, X = bike_frame(payload)
    try:
        model = load_model(BIKES[key]['file'])
    except ModuleNotFoundError as error:
        raise RuntimeError('Install the optional PyTorch environment to use the GRU model.') from error
    with PREDICTION_LOCK, threadpool_limits(limits=1):
        count = float(model.predict(X)[0])
    if not math.isfinite(count) or count < 0:
        raise RuntimeError('The model produced an invalid forecast.')
    return {'model': BIKES[key]['name'], 'prediction_date': payload['prediction_date'],
            'predicted_rides': round(count), 'unrounded_prediction': count,
            'note': 'One-day forecast using completed prior counts. Historical example; no live feed is connected.'}

def bike_example(key):
    if key not in BIKES:
        raise ValueError('Unknown bike model.')
    return read_json(ROOT / 'examples' / f'bike_{key}.json')

def mushroom_example():
    frame = pd.read_csv(ROOT / 'examples/mushroom_inputs.csv')
    row = frame.iloc[0]
    return {'policy': 'cautious', 'features': {n: None if pd.isna(row[n]) else (float(row[n]) if n in NUMERIC else str(row[n])) for n in NUMERIC + CATEGORICAL}}
