"""Serve the local browser interface and saved mushroom inference pipeline."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def handler_for(model, schema, model_info):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, data, kind="application/json"):
            body = json.dumps(data).encode() if kind == "application/json" else data
            self.send_response(status)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(200, (ROOT / "app/index.html").read_bytes(), "text/html")
            elif self.path == "/schema":
                self.send(200, schema)
            elif self.path == "/model-info":
                self.send(200, model_info)
            else:
                self.send(404, {"error": "Route not found"})

        def do_POST(self):
            if self.path != "/predict":
                return self.send(404, {"error": "Route not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16000:
                    raise ValueError("Expected a small JSON object.")
                values = json.loads(self.rfile.read(length))
                if not isinstance(values, dict) or set(values) - set(schema):
                    raise ValueError("Unknown input fields.")
                row = {}
                for name, specification in schema.items():
                    value = values.get(name)
                    if value in (None, ""):
                        row[name] = float("nan")
                    elif specification["type"] == "number":
                        number = float(value)
                        if isinstance(value, bool) or not math.isfinite(number) or number < 0:
                            raise ValueError(f"{name}: use a finite nonnegative number or leave blank.")
                        row[name] = number
                    elif str(value) not in specification["values"]:
                        raise ValueError(f"{name}: choose a category from the form.")
                    else:
                        row[name] = str(value)
                frame = pd.DataFrame([row], columns=list(schema))
                probability = float(model.predict_proba(frame)[0, list(model.classes_).index(1)])
                threshold = model_info["threshold"]
                self.send(200, {
                    "poison_probability": probability,
                    "prediction": "p" if probability >= threshold else "e",
                    "threshold": threshold,
                    "note": "Hypothetical coursework data. Never use this for real edibility decisions.",
                })
            except (ValueError, TypeError, UnicodeDecodeError) as error:
                self.send(400, {"error": str(error)})
    return Handler


if __name__ == "__main__":
    model_paths = [
        ROOT / "models/mushroom_random_forest_pilot.joblib",
        ROOT / "models/input_schema.json",
        ROOT / "models/model_info.json",
    ]
    if not all(path.exists() for path in model_paths):
        sys.exit("Run python src/mushrooms.py before starting the local interface.")
    model = joblib.load(model_paths[0])
    schema = json.loads(model_paths[1].read_text(encoding="utf-8"))
    model_info = json.loads(model_paths[2].read_text(encoding="utf-8"))
    server = ThreadingHTTPServer(("127.0.0.1", 8765), handler_for(model, schema, model_info))
    print("CloudAI local demo: http://127.0.0.1:8765 (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
