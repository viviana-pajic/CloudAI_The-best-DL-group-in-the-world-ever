"""Deploy frozen artifacts; keep later training builds as separate candidates."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.release_integrity import release_summary, verify_release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-root', type=Path)
    args = parser.parse_args()
    frozen = verify_release()
    if not frozen:
        if not args.candidate_root:
            raise ValueError('Supply a candidate build or freeze a release.')
        manifest = json.loads((args.candidate_root / 'models/model_manifest.json').read_text())
        for name in [*manifest['sha256'], 'model_manifest.json']:
            shutil.copy2(args.candidate_root / 'models' / name, ROOT / 'models' / name)
    record = {'utc': datetime.now(timezone.utc).isoformat(),
              'serving': 'exact frozen artifacts' if frozen else 'CPU candidate build',
              'release': release_summary(),
              'candidate_build_replaces_frozen_models': False}
    (ROOT / 'reports/hosting_build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
