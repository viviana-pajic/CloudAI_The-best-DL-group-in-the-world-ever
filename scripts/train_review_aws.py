"""Small training-only searches. AWS runs require notebook-instance metadata."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import re
import sys
import subprocess
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'mushrooms'), str(ROOT/'citibike'), str(ROOT)]

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def provenance(local_check, operator):
    started = datetime.now(timezone.utc).isoformat()
    resource = Path('/opt/ml/metadata/resource-metadata.json')
    arn = None
    if not local_check:
        if not resource.is_file():
            raise RuntimeError('Use a SageMaker notebook instance. Local tests must use --local-check.')
        arn = json.loads(resource.read_text()).get('ResourceArn', '')
        if not re.fullmatch(r'arn:[^:]+:sagemaker:[^:]+:\d{12}:notebook-instance/.+', arn):
            raise RuntimeError('Notebook-instance metadata is missing or invalid.')
        arn = re.sub(r':\d{12}:', ':<account>:', arn)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source_files = ['mushrooms/search.py', 'mushrooms/folds.py', 'citibike/research_models.py',
                    'citibike/weather_search.py', 'citibike/expanded_search.py',
                    'citibike/selection.py', 'citibike/sequence_search.py',
                    'common/protocol.py', 'common/hardware.py']
    return {'git_commit': commit, 'source_sha256': {n: digest(ROOT / n) for n in source_files}, 'operator': operator, 'started_utc': started, 'execution': 'local verification' if local_check else 'AWS notebook instance',
            'resource_arn_redacted': arn, 'python': platform.python_version(),
            'packages': {n: importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib','xgboost','catboost','optuna']},
            'runner_sha256': digest(__file__), 'reserved_final_test_scored': False,
            'development_confirmation_rescored': False, 'summer_validation_rescored': False}

def run_mushroom(out, record):
    from search import build_model, read_partition, threshold_for_recall, metrics
    from folds import cv_folds, fold_fingerprint
    frozen = json.loads((ROOT/'mushrooms/frozen_candidates.json').read_text())
    configuration = frozen['precalibration_selection']['candidate_configurations']['v3_extra_trees_0094']['config']
    ranges = json.loads((ROOT/'mushrooms/AWS_RANGES.json').read_text())['suggestions'][0]['optional_parameter_grid']
    X,y = read_partition('search')
    folds = cv_folds(X,y,n_splits=5)
    trials=[]
    configurations=[]
    for trees,features in itertools.product(ranges['n_estimators'],ranges['max_features']):
        config=copy.deepcopy(configuration)
        config['params'].update(n_estimators=trees,max_features=features)
        configurations.append(config)
    for i,config in enumerate(configurations):
        scores=[]
        for train,check in folds:
            model=build_model(config,X.iloc[train])
            with threadpool_limits(limits=2):
                model.fit(X.iloc[train],y.iloc[train])
                probabilities=model.predict_proba(X.iloc[check])[:,1]
            scores.append(float(average_precision_score(y.iloc[check],probabilities)))
        trials.append({'candidate':i,'n_estimators':config['params']['n_estimators'],'max_features':config['params']['max_features'],
                       'mean_fold_ap':float(np.mean(scores)),'fold_ap':scores})
        print(f'Mushroom {i+1}/4: AP={np.mean(scores):.6f}',flush=True)
    winner=max(trials,key=lambda t:t['mean_fold_ap'])['candidate']
    chosen=configurations[winner]
    model=build_model(chosen,X)
    with threadpool_limits(limits=2):
        model.fit(X,y)
    model.named_steps['model'].set_params(n_jobs=1)
    calibration_X,calibration_y=read_partition('calibration')
    with threadpool_limits(limits=1):
        scores=model.predict_proba(calibration_X)[:,1]
    threshold,_,_=threshold_for_recall(calibration_y,scores,1.)
    joblib.dump(model,out/'trained_model.joblib',compress=3)
    pd.DataFrame([{k:v for k,v in t.items() if k!='fold_ap'} for t in trials]).to_csv(out/'tuning_results.csv',index=False)
    pd.DataFrame({'source_row_id':calibration_X.index,'actual':calibration_y,'score':scores}).to_csv(out/'calibration_predictions.csv',index=False)
    record.update(dataset='mushrooms',training_rows=len(X),calibration_rows=len(calibration_X),chosen_configuration=chosen,
        chosen_trial=winner,trials=trials,search_data_sha256=digest(ROOT/'data/mushroom_development.csv'),
        partition_sha256=digest(ROOT/'common/mushroom_partitions.json'),fold_fingerprint=fold_fingerprint(X,y,5),
        cautious_threshold=threshold,calibration_metrics=metrics(calibration_y,scores,threshold),
        metric_note='Calibration chooses the threshold. Its performance is not an independent test.',
        inference_threads=1,artifact_sha256=digest(out/'trained_model.joblib'))

def bike_cv(model,X,y):
    from research_models import metric
    expected=[('2023-04-29','2023-05-19'),('2023-05-20','2023-06-09'),('2023-06-10','2023-06-30')]
    scores=[]
    for number,(fit,check) in enumerate(TimeSeriesSplit(n_splits=3,test_size=21).split(X)):
        dates=(str(X.index[check[0]].date()),str(X.index[check[-1]].date()))
        if dates!=expected[number]:
            raise RuntimeError('Chronological folds changed.')
        with threadpool_limits(limits=2):
            trained=clone(model).fit(X.iloc[fit],y.iloc[fit])
            predicted=trained.predict(X.iloc[check])
        scores.append({'fold':number,'start':dates[0],'end':dates[1],**metric(y.iloc[check],predicted)})
    return float(np.mean([s['mae'] for s in scores])),scores

def run_bike(out,record,neural=False):
    from selection import estimator_for_record
    from weather_search import load_weather_features
    name='blend_expanded_history_and_forecasts' if neural else 'blend_2023_history_and_forecasts'
    frozen=json.loads((ROOT/'reports/research/frozen_finalists.json').read_text())
    entry=next(r for r in frozen['finalists'] if r['run_id']==name)
    config=copy.deepcopy(entry)
    if neural:
        import os
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('This exact GRU training path needs CUDA. Use the CPU notebook instead.')
        os.environ['CLOUDAI_RESEARCH_DEADLINE_UTC']=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat()
        from expanded_search import load_extended_weather_features
        X,y,hashes=load_extended_weather_features()
        record.update(torch=importlib.metadata.version('torch'),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
        configurations=[config]
        nearby=copy.deepcopy(config)
        nearby['configuration']['members'][0]['configuration']['hidden']=16
        configurations.append(nearby)
    else:
        X,y,data_hash=load_weather_features()
        hashes={'counts_sha256':data_hash,'temperature_sha256':digest(ROOT/'audit/weather/forecast_daily_temperature.csv'),
                'rain_wind_sha256':digest(ROOT/'audit/weather/forecast_daily_jma_precip_wind.csv')}
        configurations=[]
        exact_C=config['configuration']['members'][0]['configuration']['C']
        for C,iterations in itertools.product([exact_C,exact_C*.8],[250,200]):
            value=copy.deepcopy(config)
            value['configuration']['members'][0]['configuration']['C']=C
            value['configuration']['members'][1]['configuration']['max_iter']=iterations
            configurations.append(value)
    if str(X.index.max().date())!='2023-06-30':
        raise RuntimeError('Do not train or tune on summer or final-test targets.')
    trials=[]
    for i,candidate in enumerate(configurations):
        score,folds=bike_cv(estimator_for_record(candidate),X,y)
        trials.append({'candidate':i,'mean_fold_mae':score,'folds':folds})
        print(f'Bike {i+1}/{len(configurations)}: MAE={score:.2f}',flush=True)
    winner=min(trials,key=lambda t:t['mean_fold_mae'])['candidate']
    model=estimator_for_record(configurations[winner])
    with threadpool_limits(limits=2):
        model.fit(X,y)
    joblib.dump(model,out/'trained_model.joblib',compress=3)
    pd.DataFrame([{'candidate':t['candidate'],'mean_fold_mae':t['mean_fold_mae']} for t in trials]).to_csv(out/'tuning_results.csv',index=False)
    record.update(dataset='citibike',candidate=name,training_rows=len(X),training_end=str(X.index.max().date()),
        input_contract=entry['input_contract'],chosen_trial=winner,chosen_configuration=configurations[winner]['configuration'],
        trials=trials,data_hashes=hashes,artifact_sha256=digest(out/'trained_model.joblib'),
        metric_note='Training chronological CV. No new July-September score or final-test score.')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['mushrooms','citibike','citibike-gru'],required=True)
    parser.add_argument('--operator',required=True)
    parser.add_argument('--local-check',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.operator.strip().upper() in ('YOUR NAME','REPLACE ME',''):
        raise ValueError('Enter the name of the actual operator.')
    record=provenance(args.local_check,args.operator)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=args.output or ROOT/'reports'/('local_check' if args.local_check else 'aws')/(args.dataset+'-'+stamp)
    if out.exists():
        raise FileExistsError('Choose a new output folder so earlier evidence is preserved.')
    out.mkdir(parents=True)
    write_json(out/'run_started.json',record)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if args.dataset=='mushrooms':
                run_mushroom(out,record)
            else:
                run_bike(out,record,args.dataset=='citibike-gru')
        record.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),
                      warnings=sorted(set(type(w.message).__name__+': '+str(w.message) for w in caught)))
        write_json(out/'run_record.json',record)
        print(f'Saved actual run: {out}',flush=True)
    except Exception as error:
        record.update(status='failed',error=str(error))
        write_json(out/'run_record.json',record)
        raise

if __name__=='__main__':
    main()
