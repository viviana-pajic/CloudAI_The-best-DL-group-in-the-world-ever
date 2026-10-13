"""Record the team's release choice before final-test evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

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
    names=['src/review_models.py','src/review_features.py','models/model_manifest.json',
           'mushrooms/split_manifest.json','citibike/archive_manifest.json']
    names += [str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'models').glob('*.joblib')]
    names += ['mushrooms/search.py','mushrooms/folds.py','citibike/research_models.py',
              'citibike/selection.py','citibike/weather_search.py','citibike/expanded_search.py',
              'citibike/sequence_search.py',
              'common/hardware.py','common/protocol.py','scripts/evaluate_final_tests.py',
              'scripts/download_final_test_forecasts.py','scripts/download_forecast_temperature.py',
              'scripts/download_forecast_jma.py']
    record={'status':'frozen','utc':datetime.now(timezone.utc).isoformat(),'operator':args.operator,
            'reason':args.reason,'bike_model':args.bike_model,
            'mushroom_threshold':json.loads((ROOT/'models/model_manifest.json').read_text())['mushroom_cautious_threshold'],
            'files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
            'final_test_scored_before_this_decision':False,
            'rule':'Report the final results. Do not change models or thresholds to improve those results.'}
    with target.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    print(target)

if __name__=='__main__':
    main()
