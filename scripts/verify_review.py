"""Check serving parity, causal inputs and both HTTP prediction paths."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'mushrooms'),str(ROOT/'citibike'),str(ROOT)]
import numpy as np
import pandas as pd
from src.review_models import bike_example, bike_frame, load_model, mushroom_frame, mushroom_example, predict_mushroom, predict_bike

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['frozen','rebuilt'],default='frozen')
    args=parser.parse_args()
    results={'utc':datetime.now(timezone.utc).isoformat(),'mode':args.mode,
        'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','joblib','torch']},
        'final_tests_scored':False,'confirmation_rescored':False,'summer_rescored':False}
    from weather_search import load_weather_features
    X,y,_=load_weather_features()
    date='2023-06-30'
    predictions={}
    for key,file in [('counts','citibike_counts_xgboost.joblib'),('weather_cpu','citibike_weather_cpu.joblib'),('weather_gru','citibike_weather_gru.joblib')]:
        payload=bike_example(key)
        _,served=bike_frame(payload)
        model=load_model(file)
        members=getattr(model,'models_',[model])
        for member in members:
            columns=getattr(member,'columns_',getattr(member,'context_columns_',[]))
            if hasattr(member,'_sequence'):
                np.testing.assert_allclose(member._sequence(served),member._sequence(X.loc[[date]]),rtol=0,atol=0)
            pd.testing.assert_frame_equal(served[columns],X.loc[[date],columns],check_dtype=False,check_freq=False)
        expected=float(model.predict(X.loc[[date]])[0])
        actual=predict_bike(payload)['unrounded_prediction']
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-6)
        predictions[key]={'prediction':actual,'feature_parity':True,'prediction_difference':actual-expected}
    results['bike_parity']=predictions
    if args.mode=='frozen':
        inputs=pd.read_csv(ROOT/'examples/mushroom_inputs.csv')
        expected=pd.read_csv(ROOT/'examples/mushroom_expected_predictions.csv')
        actual=load_model('mushroom_extra_trees.joblib').predict_proba(inputs)[:,1]
        np.testing.assert_allclose(actual,expected.research_risk_score.to_numpy(),rtol=0,atol=1e-14)
        manifest=json.loads((ROOT/'models/model_manifest.json').read_text())
        assert np.array_equal(actual>=manifest['mushroom_cautious_threshold'],expected.flag_as_potentially_poisonous)
        results['mushroom_parity']={'training_side_unlabelled_examples':5,'max_score_difference':float(np.max(np.abs(actual-expected.research_risk_score.to_numpy()))),'flags_match':True}
    else:
        results['mushroom_rebuilt_prediction']=predict_mushroom(mushroom_example())
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    server=subprocess.Popen([sys.executable,str(ROOT/'app/review_server.py'),'--port',str(port)],cwd=ROOT,
        stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
    def request(path,payload=None):
        req=Request(base+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        try:
            with urlopen(req,timeout=20) as response:
                return response.status,response.read()
        except HTTPError as error:
            return error.code,error.read()
    try:
        for _ in range(120):
            if server.poll() is not None:
                raise RuntimeError(server.stderr.read().decode())
            try:
                status,_=request('/health')
                if status==200:break
            except OSError:time.sleep(.25)
        else:raise TimeoutError('Server did not start.')
        checks={}
        for path in ['/','/api/models','/docs/TEAM_GUIDE.html']:
            status,_=request(path);assert status==200;checks[path]=status
        for key in predictions:
            status,body=request('/api/bike/predict',bike_example(key));assert status==200,(key,body)
            np.testing.assert_allclose(json.loads(body)['unrounded_prediction'],predictions[key]['prediction'],rtol=1e-10,atol=1e-6)
            checks['bike_'+key]=status
        for policy in ['cautious','default']:
            payload=mushroom_example();payload['policy']=policy
            status,body=request('/api/mushroom/predict',payload);assert status==200,body
            checks['mushroom_'+policy]=status
        invalid=[]
        sample=bike_example('weather_gru');sample['forecasts']=[];invalid.append(('/api/bike/predict',sample))
        sample=bike_example('counts');sample['prior_counts']=sample['prior_counts'][:-1];invalid.append(('/api/bike/predict',sample))
        sample=bike_example('counts');sample['prior_counts'][0]=-1;invalid.append(('/api/bike/predict',sample))
        sample=bike_example('counts');sample['prediction_date']='not-a-date';invalid.append(('/api/bike/predict',sample))
        sample=mushroom_example();sample['features']['class']='p';invalid.append(('/api/mushroom/predict',sample))
        for path,payload in invalid:
            status,body=request(path,payload);assert status==400,(status,body)
        results['http_checks']=checks
        results['invalid_inputs_rejected']=len(invalid)
    finally:
        server.terminate();server.wait(timeout=20)
    results['passed']=True
    (ROOT/'reports/review_verification.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
