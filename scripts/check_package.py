"""Check the source ZIP before environment installation or cloud training."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    package=ROOT/'PACKAGE_MANIFEST.json'
    if not package.exists():
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        models=json.loads((ROOT/'models/model_manifest.json').read_text())['sha256']
        for name,wanted in models.items():
            if hashlib.sha256((ROOT/'models'/name).read_bytes()).hexdigest()!=wanted:
                raise RuntimeError('Model mismatch: '+name)
        print('Git checkout:',commit)
        print('Saved model hashes match. Run verify_review.py after installing the environment.')
        return
    manifest=json.loads(package.read_text())
    wrong=[]
    for name,wanted in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=wanted:
            wrong.append(name)
    if wrong:
        raise RuntimeError('Package mismatch: '+', '.join(wrong))
    print(f"Package checked: {len(manifest['files'])} files. Final tests have not been scored.")

if __name__=='__main__':
    main()
