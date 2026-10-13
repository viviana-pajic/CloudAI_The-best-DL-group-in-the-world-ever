"""Check the frozen release on Windows and Linux."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path, text=False):
    content = Path(path).read_bytes()
    if text:
        content = content.replace(b'\r\n', b'\n')
    return hashlib.sha256(content).hexdigest()


def verify_release(required=False):
    path = ROOT / 'release_decision.json'
    if not path.exists():
        if required:
            raise FileNotFoundError('Freeze the release before evaluation.')
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    if record['status'] != 'frozen' or record['hash_scheme'] != 'binary-raw/text-LF-v1':
        raise ValueError('Unsupported release decision.')
    for name, wanted in record['files'].items():
        if digest(ROOT / name, text=not name.endswith('.joblib')) != wanted:
            raise RuntimeError('Release changed after freezing: ' + name)
    return record


def release_summary():
    path = ROOT / 'release_decision.json'
    if not path.exists():
        return {'frozen': False}
    record = json.loads(path.read_text(encoding='utf-8'))
    return {'frozen': True, 'decision_sha256': digest(path, text=True),
            'bike_model': record['bike_model'], 'mushroom_policy': record['mushroom_policy'],
            'mushroom_threshold': record['mushroom_threshold']}
