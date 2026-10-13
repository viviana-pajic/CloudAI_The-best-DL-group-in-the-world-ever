"""Record the team's release choice before final-test evaluation."""
import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.release_integrity import digest

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bike-model',choices=['counts','weather_cpu','weather_gru'],required=True)
    parser.add_argument('--operator',required=True)
    parser.add_argument('--reason',required=True)
    args=parser.parse_args()
    if args.operator.strip().upper() in ('YOUR NAME', ''):
        raise ValueError('Enter the actual operator name.')
    target=ROOT/'release_decision.json'
    if target.exists() or (ROOT/'reports/final_test_started.json').exists():
        raise FileExistsError('A release decision or final-test attempt already exists. Review it as a team.')
    manifest=json.loads((ROOT/'models/model_manifest.json').read_text())
    for name,wanted in manifest['sha256'].items():
        if digest(ROOT/'models'/name)!=wanted:raise RuntimeError('Model differs from development artifact: '+name)
    names=['src/review_models.py','src/review_features.py','models/model_manifest.json',
           'mushrooms/split_manifest.json','citibike/archive_manifest.json']
    names += ['models/'+n for n in manifest['sha256']]
    names += ['mushrooms/search.py','mushrooms/folds.py','citibike/research_models.py',
              'citibike/selection.py','citibike/weather_search.py','citibike/expanded_search.py',
              'citibike/sequence_search.py',
              'common/hardware.py','common/protocol.py','scripts/evaluate_final_tests.py',
              'scripts/download_final_test_forecasts.py','scripts/download_forecast_temperature.py',
              'scripts/download_forecast_jma.py']
    names += ['src/release_integrity.py','scripts/freeze_release.py','scripts/assemble_hosting.py',
              'scripts/verify_hosted.py','scripts/verify_review.py','scripts/rebuild_review_models.py',
              'scripts/train_review_aws.py','app/review_server.py','app/review.html','Dockerfile',
              '.github/workflows/review-release.yml','render.yaml',
              'requirements-review.txt','requirements-review-linux.lock.txt','reports/release_preflight_decision.json']
    record={'status':'frozen','utc':datetime.now(timezone.utc).isoformat(),'operator':args.operator,
            'reason':args.reason,'assistance':'Codex','bike_model':args.bike_model,
            'bike_missing_forecast_fallback':'counts','mushroom_policy':'cautious','secondary_mushroom_threshold':0.5,'bike_baseline':'lag_7',
            'test_protocol':{'mushroom_rows':1000,'bike_start':'2023-10-01','bike_end':'2023-12-31',
                'bike_horizon_days':1,'bike_history':'Rolling evaluation: earlier completed test days are known history.'},
            'hash_scheme':'binary-raw/text-LF-v1',
            'preflight_revision':{'previous_decision_sha256':digest(ROOT/'reports/release_preflight_decision.json',text=True),
                'reason':'GFS archive has 53 missing required hours on 2023-12-29 through 2023-12-31. Use counts-only fallback, not observations or invented forecasts. Reserved targets remain unread.'},
            'mushroom_threshold':json.loads((ROOT/'models/model_manifest.json').read_text())['mushroom_cautious_threshold'],
            'files':{n:digest(ROOT/n,text=not n.endswith('.joblib')) for n in names},
            'final_test_scored_before_this_decision':False,
            'rule':'Report the final results. Do not change models or thresholds to improve those results.'}
    with target.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(record,f,indent=2)
        f.write('\n')
    print(target)

if __name__=='__main__':
    main()
