"""One webpage origin, two isolated model runtimes behind local APIs."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def gateway_handler(mushroom_port=8781, bike_port=8782):
    class Gateway(BaseHTTPRequestHandler):
        def respond(self, status, body, kind='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def proxy(self, port, path):
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                return self.respond(400, b'{"error":"Invalid request length"}')
            if length < 0 or length > 16000:
                return self.respond(413, b'{"error":"Request too large"}')
            data = self.rfile.read(length) if self.command == 'POST' else None
            request = Request(f'http://127.0.0.1:{port}{path}', data=data, method=self.command,
                              headers={'Content-Type': 'application/json'})
            try:
                with urlopen(request, timeout=20) as response:
                    return self.respond(response.status, response.read(), response.headers.get('Content-Type', 'application/json'))
            except HTTPError as error:
                return self.respond(error.code, error.read(), error.headers.get('Content-Type', 'application/json'))
            except (URLError, TimeoutError):
                return self.respond(503, b'{"error":"The model API is unavailable. Check the local app logs."}')

        def do_GET(self):
            if self.path == '/':
                return self.respond(200, (ROOT / 'app/home.html').read_bytes(), 'text/html; charset=utf-8')
            if self.path == '/citibike':
                return self.respond(200, (ROOT / 'app/citibike.html').read_bytes(), 'text/html; charset=utf-8')
            if self.path == '/mushrooms':
                return self.proxy(mushroom_port, '/')
            if self.path in ['/schema', '/model-info']:
                return self.proxy(mushroom_port, self.path)
            if self.path in ['/bike/model-info', '/bike/example']:
                return self.proxy(bike_port, self.path)
            if self.path == '/health':
                try:
                    for port, path in [(mushroom_port, '/model-info'), (bike_port, '/bike/model-info')]:
                        with urlopen(f'http://127.0.0.1:{port}{path}', timeout=2) as response:
                            json.load(response)
                    return self.respond(200, b'{"mushrooms":"ready","citibike":"ready"}')
                except (URLError, TimeoutError, ValueError):
                    return self.respond(503, b'{"error":"A model API is unavailable"}')
            return self.respond(404, b'{"error":"Route not found"}')

        def do_POST(self):
            if self.path == '/predict':
                return self.proxy(mushroom_port, self.path)
            if self.path == '/bike/predict':
                return self.proxy(bike_port, self.path)
            return self.respond(404, b'{"error":"Route not found"}')
    return Gateway


def serve(host='127.0.0.1', port=8780, mushroom_port=8781, bike_port=8782):
    server = ThreadingHTTPServer((host, port), gateway_handler(mushroom_port, bike_port))
    print(f'CloudAI shared app: http://{host}:{port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
