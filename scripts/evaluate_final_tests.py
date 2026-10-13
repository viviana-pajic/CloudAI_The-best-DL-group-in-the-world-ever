"""Once-only final evaluation of the predeclared frozen models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'mushrooms'), str(ROOT/'citibike'), str(ROOT)]
import numpy as np
import pandas as pd
from src.review_models import BIKES, NUMERIC, CATEGORICAL, load_model, predict_bike, bike_frame, bike_example
from src.release_integrity import digest, verify_release
from search import metrics
from research_models import metric
from weather_search import BASE_WEATHER
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mushroom-csv', type=Path, required=True)
    parser.add_argument('--bike-daily', type=Path, required=True)
    parser.add_argument('--forecasts', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--acknowledge-final-test', action='store_true')
    args = parser.parse_args()
    if not args.preflight_only and not args.acknowledge_final_test:
        parser.error('Acknowledge the once-only test, or request preflight only.')
    decision = verify_release(required=True)
    split = json.loads((ROOT/'mushrooms/split_manifest.json').read_text())
    annual = json.loads((ROOT/'citibike/archive_manifest.json').read_text())
    if digest(args.mushroom_csv) != split['dataset_sha256']:
        raise ValueError('Wrong teacher mushroom file.')
    if digest(args.bike_daily) != annual['daily_sha256']:
        raise ValueError('Wrong audited annual daily file.')
    if len(split['splits']['test']) != 1000 or len(set(split['splits']['test'])) != 1000:
        raise ValueError('Invalid reserved mushroom split.')
    key = decision['bike_model']
    days = BIKES[key]['weather_days']
    if days and (not args.forecasts or not args.forecasts.is_file()):
        raise ValueError('Download genuine final-period forecasts first.')
    forecasts = pd.read_csv(args.forecasts, parse_dates=['date']).set_index('date') if days else None
    dates = pd.date_range('2023-10-01', '2023-12-31')
    # Preflight does not read either reserved target table. Validate forecast availability
    # and the serving environment with synthetic histories and training-side examples.
    if days:
        needed = pd.date_range(dates.min()-pd.Timedelta(days=days-1), dates.max())
        if not forecasts.index.is_unique or not needed.isin(forecasts.index).all():
            raise ValueError('Forecast dates are missing or duplicated.')
        if np.isinf(forecasts.loc[needed, BASE_WEATHER].to_numpy()).any():
            raise ValueError('Infinite final weather inputs.')
    payloads = []
    for date in dates:
        weather = []
        if days:
            required = pd.date_range(date-pd.Timedelta(days=days-1), date)
            weather = [{'date': str(d.date()), **{n: float(row[n]) for n in BASE_WEATHER}}
                       for d, row in forecasts.loc[required, BASE_WEATHER].iterrows()]
        payload = {'model': key, 'prediction_date': str(date.date()),
                   'prior_counts': [10000]*28, 'forecasts': weather}
        if weather and any(not np.isfinite(row[n]) for row in weather for n in BASE_WEATHER):
            if decision.get('bike_missing_forecast_fallback') != 'counts':
                raise ValueError('Missing forecast without a frozen fallback.')
            payload['forecasts'] = []
        check = {**payload, 'model': 'counts'} if days and not payload['forecasts'] else payload
        bike_frame(check)
        payloads.append(payload)
    load_model('mushroom_extra_trees.joblib')
    predict_bike(bike_example(key))
    if args.preflight_only:
        print('Preflight passed: artifact hashes, 92 dated input paths and inference environment. Reserved targets not read.')
        print('Counts-only fallback dates:', [p['prediction_date'] for p in payloads if days and not p['forecasts']])
        return
    marker = ROOT/'reports/final_test_started.json'
    with marker.open('x', encoding='utf-8') as f:
        json.dump({'utc': datetime.now(timezone.utc).isoformat(),
                   'decision_sha256': digest(ROOT/'release_decision.json', text=True)}, f, indent=2)
    mushroom = pd.read_csv(args.mushroom_csv)
    test = mushroom.iloc[split['splits']['test']]
    X = test[NUMERIC+CATEGORICAL]
    y = test['class'].map({'e': 0, 'p': 1})
    if len(test) != 1000 or y.isna().any():
        raise ValueError('Invalid reserved mushroom rows.')
    model = load_model('mushroom_extra_trees.joblib')
    with threadpool_limits(limits=1):
        scores = model.predict_proba(X)[:, list(model.classes_).index(1)]
    cautious = metrics(y, scores, decision['mushroom_threshold'])
    default = metrics(y, scores, decision['secondary_mushroom_threshold'])
    # Persist the first dataset immediately; never erase it if a later step fails.
    pd.DataFrame({'source_row_id': test.index, 'actual': y, 'score': scores,
                  'cautious_flag': scores >= decision['mushroom_threshold'],
                  'default_flag': scores >= decision['secondary_mushroom_threshold']}).to_csv(
                      ROOT/'reports/final_mushroom_predictions.csv', index=False)
    daily = pd.read_csv(args.bike_daily, parse_dates=['date']).set_index('date')
    if not daily.index.equals(pd.date_range('2023-01-01', '2023-12-31')):
        raise ValueError('Annual daily dates changed.')
    if daily.trips.isna().any() or (daily.trips < 0).any() or (daily.trips % 1 != 0).any():
        raise ValueError('Invalid audited daily counts.')
    rows = []
    for date, payload in zip(dates, payloads):
        payload['prior_counts'] = daily.loc[date-pd.Timedelta(days=28):date-pd.Timedelta(days=1), 'trips'].astype(int).tolist()
        result = predict_bike(payload)
        predicted = result['unrounded_prediction']
        rows.append({'date': str(date.date()), 'actual': int(daily.loc[date, 'trips']),
                     'prediction': predicted, 'used_model': 'counts' if result['missing_forecast_fallback'] else key, 'lag_7': int(daily.loc[date-pd.Timedelta(days=7), 'trips'])})
        pd.DataFrame(rows).to_csv(ROOT/'reports/final_bike_predictions.csv', index=False)
    frame = pd.DataFrame(rows)
    record = {'status': 'complete', 'utc': datetime.now(timezone.utc).isoformat(),
              'operator': decision['operator'], 'assistance': 'Codex',
              'decision_sha256': digest(ROOT/'release_decision.json', text=True), 'decision': decision,
              'mushrooms': cautious, 'mushrooms_default': default, 'mushroom_test_rows': len(test),
              'citibike': metric(frame.actual, frame.prediction),
              'citibike_lag_7': metric(frame.actual, frame.lag_7), 'bike_test_days': len(frame),
              'bike_fallback_dates': frame.loc[frame.used_model == 'counts', 'date'].tolist() if days else [],
              'citibike_weather_complete': metric(frame.loc[frame.used_model == key, 'actual'], frame.loc[frame.used_model == key, 'prediction']),
              'citibike_lag_7_weather_complete': metric(frame.loc[frame.used_model == key, 'actual'], frame.loc[frame.used_model == key, 'lag_7']),
              'model_sha256': json.loads((ROOT/'models/model_manifest.json').read_text())['sha256'],
              'input_hashes': {'mushroom_csv': digest(args.mushroom_csv), 'bike_daily': digest(args.bike_daily),
                               'forecasts': digest(args.forecasts) if days else None},
              'prediction_hashes': {n: digest(ROOT/'reports'/n, text=True) for n in
                  ['final_mushroom_predictions.csv', 'final_bike_predictions.csv']},
              'note': 'Reserved final test. Settings, both mushroom rules and lag-7 baseline frozen first. No tuning on these results.',
              'weather_limit': 'Archived forecast initialization is an availability proxy; exact historical publication times are unverified.'}
    (ROOT/'reports/final_test_results.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in record.items() if k != 'decision'}, indent=2))


if __name__ == '__main__':
    main()