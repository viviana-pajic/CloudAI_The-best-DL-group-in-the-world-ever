"""Serve the local browser interface and saved mushroom inference pipeline."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import importlib.metadata
import math
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_bundle(path):
    """Load the current model, schema and threshold together, or fail clearly."""
    bundle = joblib.load(path)
    model, schema, info = bundle["pipeline"], bundle["schema"], bundle["metadata"]
    if info["environment"]["packages"]["scikit-learn"] != importlib.metadata.version("scikit-learn"):
        raise ValueError("Use the same scikit-learn version as the recorded training environment.")
    if list(schema) != list(model.feature_names_in_) or list(schema) != info["feature_columns"]:
        raise ValueError("Model/schema feature contract mismatch.")
    if list(model.classes_) != [0, 1] or not 0 <= info["threshold"] <= 1:
        raise ValueError("Invalid positive-class interpretation or threshold.")
    return model, schema, info


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
            except (ValueError, TypeError, UnicodeDecodeError, OverflowError) as error:
                self.send(400, {"error": str(error)})
    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    bundle_path = ROOT / "models/mushroom_candidate.joblib"
    if not bundle_path.exists():
        sys.exit("Run src/mushroom_workflow.py baseline in the documented mushroom environment first.")
    model, schema, model_info = load_bundle(bundle_path)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(model, schema, model_info))
    print(f"CloudAI local demo: http://127.0.0.1:{args.port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
