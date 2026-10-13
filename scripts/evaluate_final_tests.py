"""Once-only final test after the team freezes its release choice."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'mushrooms'),str(ROOT/'citibike'),str(ROOT)]
import numpy as np
import pandas as pd
from src.review_models import BIKES, NUMERIC, CATEGORICAL, load_model, predict_bike
from search import metrics
from research_models import metric
from weather_search import BASE_WEATHER
from threadpoolctl import threadpool_limits

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mushroom-csv',type=Path,required=True)
    parser.add_argument('--bike-daily',type=Path,required=True)
    parser.add_argument('--forecasts',type=Path)
    parser.add_argument('--acknowledge-final-test',action='store_true',required=True)
    args=parser.parse_args()
    decision=json.loads((ROOT/'release_decision.json').read_text())
    if decision['status']!='frozen':raise ValueError('Freeze the release first.')
    for name,wanted in decision['files'].items():
        if sha(ROOT/name)!=wanted:raise RuntimeError('Release changed after freezing: '+name)
    mushroom_split=json.loads((ROOT/'mushrooms/split_manifest.json').read_text())
    annual=json.loads((ROOT/'citibike/archive_manifest.json').read_text())
    if sha(args.mushroom_csv)!=mushroom_split['dataset_sha256']:raise ValueError('Wrong teacher mushroom file.')
    if sha(args.bike_daily)!=annual['daily_sha256']:raise ValueError('Wrong audited annual daily file.')
    key=decision['bike_model']
    days=BIKES[key]['weather_days']
    if days and (not args.forecasts or not args.forecasts.is_file()):raise ValueError('The selected model requires final-period forecasts. Do not invent them.')
    # This marker is deliberately exclusive. Never rerun to optimise a test score.
    marker=ROOT/'reports/final_test_started.json'
    with marker.open('x',encoding='utf-8') as f:
        json.dump({'utc':datetime.now(timezone.utc).isoformat(),'decision_sha256':sha(ROOT/'release_decision.json')},f,indent=2)
    mushroom=pd.read_csv(args.mushroom_csv)
    test=mushroom.iloc[mushroom_split['splits']['test']]
    if len(test)!=1000:raise ValueError('Final mushroom split must contain 1,000 rows.')
    X=test[NUMERIC+CATEGORICAL]
    y=test['class'].map({'e':0,'p':1})
    with threadpool_limits(limits=1):
        scores=load_model('mushroom_extra_trees.joblib').predict_proba(X)[:,1]
    mushroom_metrics=metrics(y,scores,decision['mushroom_threshold'])
    daily=pd.read_csv(args.bike_daily,parse_dates=['date']).set_index('date')
    if not daily.index.equals(pd.date_range('2023-01-01','2023-12-31')):raise ValueError('Annual daily dates changed.')
    forecasts=pd.read_csv(args.forecasts,parse_dates=['date']).set_index('date') if days else None
    rows=[]
    for date in pd.date_range('2023-10-01','2023-12-31'):
        weather=[]
        if days:
            required=pd.date_range(date-pd.Timedelta(days=days-1),date)
            for d,row in forecasts.loc[required,BASE_WEATHER].iterrows():
                weather.append({'date':str(d.date()),**{n:float(row[n]) for n in BASE_WEATHER}})
        payload={'model':key,'prediction_date':str(date.date()),
            'prior_counts':daily.loc[date-pd.Timedelta(days=28):date-pd.Timedelta(days=1),'trips'].astype(int).tolist(),
            'forecasts':weather}
        predicted=predict_bike(payload)['unrounded_prediction']
        rows.append({'date':str(date.date()),'actual':int(daily.loc[date,'trips']),'prediction':predicted})
    frame=pd.DataFrame(rows)
    record={'status':'complete','utc':datetime.now(timezone.utc).isoformat(),'decision':decision,
        'mushrooms':mushroom_metrics,'citibike':metric(frame.actual,frame.prediction),
        'input_hashes':{'mushroom_csv':sha(args.mushroom_csv),'bike_daily':sha(args.bike_daily),
                       'forecasts':sha(args.forecasts) if days else None},
        'note':'Reserved final test. Settings and threshold were frozen first. No tuning on this result.'}
    frame.to_csv(ROOT/'reports/final_bike_predictions.csv',index=False)
    pd.DataFrame({'source_row_id':test.index,'actual':y,'score':scores}).to_csv(ROOT/'reports/final_mushroom_predictions.csv',index=False)
    (ROOT/'reports/final_test_results.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':
    main()
