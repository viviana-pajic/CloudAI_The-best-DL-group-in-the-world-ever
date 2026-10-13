"""Check the public release using existing training-side example inputs."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.release_integrity import release_summary, verify_release
from src.review_models import predict_bike, predict_mushroom

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.url.rstrip('/')
    parsed = urlparse(base)
    if parsed.scheme != 'https' or parsed.path or parsed.query or parsed.username:
        raise ValueError('Use the public HTTPS origin without credentials or a path.')
    if not re.fullmatch('[0-9a-f]{40}', args.expected_commit):
        raise ValueError('Supply the full release commit.')

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        req = Request(base + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urlopen(req, timeout=60) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    status, body = request('/health')
    assert status == 200, (status, body[:200])
    health = json.loads(body)
    verify_release(required=True)
    assert health['frozen_release'] == release_summary(), health
    assert health['model_sha256'] == json.loads((ROOT/'models/model_manifest.json').read_text())['sha256']
    assert health['final_tests_complete']
    assert health['ready'] and health['release'] == args.expected_commit, health
    assert set(health['models']) == {'counts', 'weather_cpu', 'weather_gru'}
    assert len(health['model_sha256']) == 4
    assert all(re.fullmatch('[0-9a-f]{64}', digest) for digest in health['model_sha256'].values())
    status, body = request('/')
    assert status == 200 and b'Mushroom alarm' in body
    checks = {'page': status, 'health': 200}
    status, body = request('/api/release')
    public_results = json.loads(body)
    saved_results = json.loads((ROOT/'reports/final_test_results.json').read_text())
    assert status == 200 and public_results['status'] == 'complete'
    for name, value in public_results.items():
        assert saved_results[name] == value, name
    checks['saved_final_results_match'] = True
    predictions = {}
    for model in health['models']:
        payload = json.loads((ROOT/'examples'/f'bike_{model}.json').read_text())
        status, body = request('/api/bike/predict', payload)
        assert status == 200, (model, status, body[:200])
        result = json.loads(body)
        local = predict_bike(payload)
        assert math.isclose(result['unrounded_prediction'], local['unrounded_prediction'], rel_tol=1e-7, abs_tol=.05), (model, result, local)
        assert math.isfinite(result['unrounded_prediction']) and result['predicted_rides'] >= 0
        predictions[model] = result
        checks[model] = status
    status, body = request('/api/mushroom/example')
    assert status == 200
    payload = json.loads(body)
    for policy in ('default', 'cautious'):
        payload['policy'] = policy
        status, body = request('/api/mushroom/predict', payload)
        assert status == 200, (policy, status, body[:200])
        result = json.loads(body)
        local = predict_mushroom(payload)
        assert math.isclose(result['research_score'], local['research_score'], rel_tol=1e-10, abs_tol=1e-12)
        assert 0 <= result['research_score'] <= 1
        expected = .5 if policy == 'default' else health['mushroom_cautious_threshold']
        assert result['threshold'] == expected
        assert result['flag_as_potentially_poisonous'] == (result['research_score'] >= expected)
        predictions['mushroom_' + policy] = result
        checks['mushroom_' + policy] = status
    fallback_payload = json.loads((ROOT/'examples/bike_weather_gru.json').read_text())
    fallback_payload['forecasts'] = []
    status, body = request('/api/bike/predict', fallback_payload)
    fallback = json.loads(body)
    assert status == 200 and fallback['missing_forecast_fallback']
    local_fallback = predict_bike(fallback_payload)
    assert math.isclose(fallback['unrounded_prediction'], local_fallback['unrounded_prediction'], rel_tol=1e-7, abs_tol=.05)
    checks['missing_weather_fallback'] = status
    predictions['missing_weather_fallback'] = fallback
    invalid = json.loads((ROOT/'examples/bike_counts.json').read_text())
    invalid['prior_counts'] = []
    status, _ = request('/api/bike/predict', invalid)
    assert status == 400
    checks['invalid_input'] = status
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'url': base,
              'health': health, 'checks': checks, 'predictions': predictions,
              'reserved_targets_scored_by_this_check': False, 'passed': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
