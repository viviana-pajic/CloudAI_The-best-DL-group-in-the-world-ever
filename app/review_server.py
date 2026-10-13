"""Local review webpage and prediction API."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.review_models import BIKES, load_model, predict_mushroom, predict_bike, bike_example, mushroom_example
from src.release_integrity import release_summary, verify_release

class Handler(BaseHTTPRequestHandler):
    def respond(self, status, content, kind='application/json'):
        body = json.dumps(content, allow_nan=False).encode() if kind == 'application/json' else content
        self.send_response(status)
        self.send_header('Content-Type', kind + '; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path)
        if path.path == '/':
            return self.respond(200, (ROOT/'app/review.html').read_bytes(), 'text/html')
        if path.path == '/health':
            manifest = json.loads((ROOT/'models/model_manifest.json').read_text())
            return self.respond(200, {'ready': True, 'models': list(BIKES),
                'release': os.environ.get('RENDER_GIT_COMMIT', 'local-review'),
                'model_origin': manifest['origin'], 'model_sha256': manifest['sha256'],
                'mushroom_cautious_threshold': manifest['mushroom_cautious_threshold'],
                'frozen_release': release_summary(),
                'final_tests_complete': (ROOT/'reports/final_test_results.json').is_file()})
        if path.path == '/api/release':
            file = ROOT/'reports/final_test_results.json'
            if not file.is_file():
                return self.respond(200, {'status': 'pending', 'release': release_summary()})
            record = json.loads(file.read_text(encoding='utf-8'))
            return self.respond(200, {n: record[n] for n in
                ['status', 'decision_sha256', 'mushrooms', 'mushrooms_default', 'mushroom_test_rows',
                 'citibike', 'citibike_lag_7', 'bike_test_days', 'bike_fallback_dates', 'citibike_weather_complete', 'citibike_lag_7_weather_complete', 'note', 'weather_limit']})
        if path.path == '/api/mushroom/example':
            return self.respond(200, mushroom_example())
        if path.path == '/api/bike/example':
            try:
                return self.respond(200, bike_example(parse_qs(path.query).get('model', ['weather_gru'])[0]))
            except ValueError as error:
                return self.respond(400, {'error': str(error)})
        if path.path == '/api/models':
            return self.respond(200, BIKES)
        if path.path.startswith('/docs/') or path.path.startswith('/reports/research/'):
            file = (ROOT/path.path.lstrip('/')).resolve()
            allowed = ((ROOT/'docs').resolve(), (ROOT/'reports/research').resolve())
            if file.is_relative_to(allowed[0]) or file.is_relative_to(allowed[1]):
                if file.is_file() and file.suffix in ('.md', '.pdf', '.png', '.html'):
                    kind = {'.md':'text/plain','.pdf':'application/pdf','.png':'image/png','.html':'text/html'}[file.suffix]
                    return self.respond(200, file.read_bytes(), kind)
        return self.respond(404, {'error': 'Page not found.'})

    def do_POST(self):
        if self.path not in ('/api/mushroom/predict', '/api/bike/predict'):
            return self.respond(404, {'error': 'API route not found.'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 64000:
                return self.respond(413, {'error': 'Request must contain at most 64 KB.'})
            payload = json.loads(self.rfile.read(size))
            result = predict_mushroom(payload) if self.path == '/api/mushroom/predict' else predict_bike(payload)
            return self.respond(200, result)
        except (ValueError, TypeError, KeyError) as error:
            return self.respond(400, {'error': str(error)})
        except RuntimeError as error:
            return self.respond(503, {'error': str(error)})
        except Exception:
            return self.respond(500, {'error': 'Prediction failed. Check the server environment and model files.'})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8780')))
    args = parser.parse_args()
    verify_release()
    for filename in ('mushroom_extra_trees.joblib', 'citibike_counts_xgboost.joblib', 'citibike_weather_cpu.joblib'):
        load_model(filename)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'CloudAI review: http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
