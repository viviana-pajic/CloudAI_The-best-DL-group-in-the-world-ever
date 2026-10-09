"""Saved-pipeline/API agreement and malformed forecast requests."""
import io
import json
from pathlib import Path
import sys
import unittest

if sys.version_info[:2] != (3, 12):
    raise unittest.SkipTest('Run Citi Bike API checks in its Python 3.12 environment.')

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'app')]
import pandas as pd
from citibike_server import handler_for_bike, load_bike_bundle, parse_request


class BikeAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.info = load_bike_bundle(ROOT / 'models/citibike_candidate.joblib')
        dates = pd.date_range('2023-12-04', '2023-12-31')
        history = [{'date': str(day.date()), 'trips': 80000 if day.dayofweek >= 5 else 100000} for day in dates]
        cls.payload = {'forecast_date': '2024-01-01', 'history': history}

    def request(self, payload):
        body = json.dumps(payload).encode()
        handler = object.__new__(handler_for_bike(self.model, self.info))
        handler.path = '/bike/predict'
        handler.headers = {'Content-Length': str(len(body))}
        handler.rfile = io.BytesIO(body)
        responses = []
        handler.send = lambda status, data: responses.append((status, data))
        handler.do_POST()
        return responses[0]

    def test_actual_saved_predictor_agrees_with_api(self):
        row, _ = parse_request(self.payload)
        status, response = self.request(self.payload)
        self.assertEqual(status, 200)
        self.assertEqual(response['predicted_rides'], round(float(self.model.predict(row)[0])))
        self.assertEqual(response['same_as_last_week'], int(row.iloc[0].lag_7))
        self.assertFalse(self.info['final_test_evaluated'])

    def test_bad_input_returns_400_instead_of_a_prediction(self):
        for key, value in [('trips', True), ('trips', -1), ('trips', float('nan')), ('date', '2023-02-30')]:
            payload = json.loads(json.dumps(self.payload))
            payload['history'][0][key] = value
            self.assertEqual(self.request(payload)[0], 400)
        payload = dict(self.payload, forecast_date='2024-01-02')
        self.assertEqual(self.request(payload)[0], 400)
        self.assertEqual(self.request(dict(self.payload, future_answer=999))[0], 400)


if __name__ == '__main__':
    unittest.main()
