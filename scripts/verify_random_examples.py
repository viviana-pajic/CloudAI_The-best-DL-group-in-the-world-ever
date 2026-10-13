"""Check real random demo inputs without fitting or scoring reserved targets."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.demo_examples import pool, random_example
from src.review_models import bike_frame, mushroom_frame, predict_bike, predict_mushroom
from src.release_integrity import digest, verify_release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify_release(required=True)
    mushrooms = pool('mushrooms')
    bikes = pool('bikes')
    assert len(mushrooms) == 120 and len(bikes) == 153
    assert len({json.dumps(r['input']['features'], sort_keys=True) for r in mushrooms}) == 120
    for row in mushrooms:
        mushroom_frame(row['input']['features'])
    for row in bikes:
        assert '2023-01-29' <= row['example_id'] <= '2023-06-30'
        for key, days in [('counts', 0), ('weather_cpu', 8), ('weather_gru', 1)]:
            payload = {**row['input'], 'model': key,
                       'forecasts': row['input']['forecasts'][-days:] if days else []}
            bike_frame(payload)
    for kind in ('mushrooms', 'bikes'):
        previous = None
        for _ in range(30):
            row = random_example(kind, exclude=previous)
            assert row['example_id'] != previous
            previous = row['example_id']
    checks = {'all_example_contracts_valid': True, 'adjacent_repeats_excluded': True}
    if args.url:
        base = args.url.rstrip('/')
        parsed = urlparse(base)
        if parsed.scheme not in ('http', 'https') or parsed.path or parsed.query or parsed.username:
            raise ValueError('Use a plain origin.')
        if parsed.scheme == 'http' and parsed.hostname != '127.0.0.1':
            raise ValueError('HTTP is allowed for localhost only.')

        def request(path, data=None):
            req = Request(base+path, data=None if data is None else json.dumps(data).encode(),
                          headers={'Content-Type': 'application/json'})
            with urlopen(req, timeout=60) as response:
                assert response.headers['Cache-Control'] == 'no-store'
                return json.loads(response.read())

        health = request('/health')
        assert health['interface_update_sha256'] == digest(ROOT/'release_ui_update.json', text=True)
        for kind, key in [('mushrooms', None), ('bikes', 'counts'),
                          ('bikes', 'weather_cpu'), ('bikes', 'weather_gru')]:
            route = '/api/mushroom/example/random' if kind == 'mushrooms' else '/api/bike/example/random'
            previous = None
            seen = []
            for _ in range(3):
                query = {'exclude': previous or ''}
                if key:
                    query['model'] = key
                row = request(route+'?'+urlencode(query))
                assert row['example_id'] != previous
                previous = row['example_id']
                seen.append(previous)
                payload = row['input']
                predict = '/api/mushroom/predict' if kind == 'mushrooms' else '/api/bike/predict'
                actual = request(predict, payload)
                local = predict_mushroom(payload) if kind == 'mushrooms' else predict_bike(payload)
                field = 'research_score' if kind == 'mushrooms' else 'unrounded_prediction'
                assert abs(actual[field]-local[field]) < (1e-10 if kind == 'mushrooms' else .05)
            checks[key or kind] = {'different_on_every_click': True, 'example_ids': seen,
                                       'predictions_match_frozen_local_model': True}
        try:
            request('/api/bike/example/random?model=invalid')
            raise AssertionError('Invalid model accepted.')
        except HTTPError as error:
            assert error.code == 400
        checks['invalid_model_rejected'] = True
    record = {'utc': datetime.now(timezone.utc).isoformat(), 'url': args.url,
              'pool_sizes': {'mushrooms': 120, 'bikes': 153}, 'checks': checks,
              'original_decision_sha256': digest(ROOT/'release_decision.json', text=True),
              'interface_update_sha256': digest(ROOT/'release_ui_update.json', text=True),
              'reserved_test_targets_read_or_rescored': False, 'passed': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
