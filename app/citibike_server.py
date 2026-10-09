"""Local next-day forecasting API, using the saved complete Citi Bike predictor."""
import argparse
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from citibike_forecast import FULL, prediction_row


def load_bike_bundle(path):
    bundle = joblib.load(path)
    info = bundle['metadata']
    if info['environment']['packages']['scikit-learn'] != importlib.metadata.version('scikit-learn'):
        raise ValueError('Use the documented Citi Bike environment to load this predictor.')
    if (info['feature_columns'] != FULL or list(bundle['pipeline'].feature_names_in_) != FULL
            or info['forecast_horizon_days'] != 1 or info['history_days'] != 28):
        raise ValueError('Citi Bike model/input contract mismatch.')
    return bundle['pipeline'], info


def parse_request(payload):
    if not isinstance(payload, dict) or set(payload) != {'forecast_date', 'history'}:
        raise ValueError('Provide forecast_date and history only.')
    forecast_date = payload['forecast_date']
    if not isinstance(forecast_date, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', forecast_date):
        raise ValueError('Use an ISO forecast date: YYYY-MM-DD.')
    date.fromisoformat(forecast_date)
    rows = payload['history']
    if not isinstance(rows, list) or len(rows) != 28:
        raise ValueError('Provide exactly 28 daily history rows.')
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'date', 'trips'}:
            raise ValueError('Each history row needs date and trips only.')
        if not isinstance(row['date'], str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', row['date']):
            raise ValueError('History dates must use YYYY-MM-DD.')
        date.fromisoformat(row['date'])
        if isinstance(row['trips'], bool) or not isinstance(row['trips'], (int, float)):
            raise ValueError('Ride counts must be numbers, not text or booleans.')
    history = pd.DataFrame(rows).set_index('date')
    history.index = pd.to_datetime(history.index)
    return prediction_row(history, forecast_date), forecast_date


def handler_for_bike(model, info):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, value, kind='application/json'):
            body = json.dumps(value, allow_nan=False).encode() if kind == 'application/json' else value
            self.send_response(status)
            self.send_header('Content-Type', kind + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                return self.send(200, (ROOT / 'app/citibike.html').read_bytes(), 'text/html')
            if self.path == '/bike/model-info':
                return self.send(200, info)
            if self.path == '/bike/example':
                dates = pd.date_range('2023-12-04', '2023-12-31')
                history = [{'date': str(day.date()), 'trips': 80000 if day.dayofweek >= 5 else 100000} for day in dates]
                return self.send(200, {'forecast_date': '2024-01-01', 'history': history,
                                      'note': 'Made-up counts for an input demonstration; no reserved-test observations are used.'})
            return self.send(404, {'error': 'Route not found'})

        def do_POST(self):
            if self.path != '/bike/predict':
                return self.send(404, {'error': 'Route not found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16000:
                    raise ValueError('Expected a small JSON request.')
                row, forecast_date = parse_request(json.loads(self.rfile.read(length)))
                predicted = float(model.predict(row)[0])
                self.send(200, {'forecast_date': forecast_date, 'predicted_rides': round(predicted),
                                'same_as_last_week': int(row.iloc[0].lag_7), 'model': info['name'],
                                'note': 'One-day historical-data forecast using information through the previous day; not a live forecast.'})
            except (ValueError, TypeError, UnicodeDecodeError, OverflowError) as error:
                self.send(400, {'error': str(error)})
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    path = ROOT / 'models/citibike_candidate.joblib'
    if not path.exists():
        raise SystemExit('Rebuild the Citi Bike candidate first; see docs/CITIBIKE_WORKFLOW.md.')
    model, info = load_bike_bundle(path)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for_bike(model, info))
    print(f'Citi Bike API: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
