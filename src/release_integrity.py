"""Check the frozen release on Windows and Linux."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI_FILES = {'app/review.html', 'app/review_server.py', 'src/release_integrity.py'}
DEMO_FILES = {'src/demo_examples.py', 'examples/random_mushrooms.json',
              'examples/random_bikes.json', 'examples/random_example_sources.json',
              'scripts/verify_random_examples.py'}


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
    overrides = {}
    update = ROOT / 'release_ui_update.json'
    if update.exists():
        patch = json.loads(update.read_text(encoding='utf-8'))
        if patch['original_decision_sha256'] != digest(path, text=True):
            raise RuntimeError('Interface update belongs to a different frozen release.')
        if set(patch['updated_files']) != UI_FILES or set(patch['added_files']) != DEMO_FILES:
            raise RuntimeError('Only the declared interface and example files may change.')
        for name, hashes in patch['updated_files'].items():
            if hashes['original_sha256'] != record['files'][name]:
                raise RuntimeError('Interface update has an incorrect original fingerprint.')
            overrides[name] = hashes['updated_sha256']
        for name, wanted in patch['added_files'].items():
            if digest(ROOT / name, text=True) != wanted:
                raise RuntimeError('Example update changed: ' + name)
    for name, wanted in record['files'].items():
        wanted = overrides.get(name, wanted)
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
