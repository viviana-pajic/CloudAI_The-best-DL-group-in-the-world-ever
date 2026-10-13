"""Rebuild CPU release candidates without reading final test targets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'citibike'),str(ROOT)]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_root.resolve()
    if out.exists():
        raise FileExistsError('Use a new build folder.')
    out.mkdir(parents=True)
    models=out/'models'
    models.mkdir()
    manifest=json.loads((ROOT/'models/model_manifest.json').read_text())
    for name in manifest['sha256']:
        shutil.copy2(ROOT/'models'/name,models/name)
    records={}
    for dataset,name in [('mushrooms','mushroom_extra_trees.joblib'),('citibike','citibike_weather_cpu.joblib')]:
        run=out/dataset
        subprocess.run([sys.executable,str(ROOT/'scripts/train_review_aws.py'),'--dataset',dataset,
                        '--operator','Automated CPU build','--local-check','--output',str(run)],check=True)
        record=json.loads((run/'run_record.json').read_text())
        records[dataset]=record
        shutil.copy2(run/'trained_model.joblib',models/name)
    import joblib
    from research_models import load_features,make_estimator
    config=json.loads((ROOT/'reports/research/citibike_counts_xgboost_metadata.json').read_text())['configuration']
    X,y,_=load_features()
    counts=make_estimator('xgboost',config).fit(X,y)
    joblib.dump(counts,models/'citibike_counts_xgboost.joblib',compress=3)
    manifest.update(origin='Automated CPU rebuild. GRU artifact remains the frozen GPU-trained candidate.',
        rebuilt_utc=datetime.now(timezone.utc).isoformat(),
        sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in models.glob('*.joblib')},
        mushroom_cautious_threshold=records['mushrooms']['cautious_threshold'],
        aws_training=False,final_tests_scored=False)
    (models/'model_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'build_record.json').write_text(json.dumps({'runs':records,'manifest':manifest},indent=2)+'\n')
    print(out)

if __name__=='__main__':
    main()
