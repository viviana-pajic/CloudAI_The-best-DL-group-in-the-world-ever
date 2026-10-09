"""Recreate the existing mushroom app bundle without changing its experiments."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from mushroom_workflow import export_demo, load_development

if __name__ == '__main__':
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('Use the separate Python 3.11 mushroom environment.')
    record = json.loads((ROOT / 'reports/mushrooms/baseline_run.json').read_text())
    for name in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib']:
        if importlib.metadata.version(name) != record['contract']['environment']['packages'][name]:
            raise SystemExit(f'Mushroom package mismatch: {name}.')
    for name, expected in record['source_sha256'].items():
        if hashlib.sha256((ROOT / 'src' / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f'The recorded mushroom source changed: {name}.')
    X, y, _ = load_development()
    export_demo(X, y, pd.read_csv(ROOT / 'reports/mushrooms/baseline_cv.csv'))
    print('Rebuilt the unchanged provisional mushroom reference, threshold 0.5.')
