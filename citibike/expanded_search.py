"""Add the audited 2022 history while keeping the 2023 checking dates."""
import hashlib
import json
import numpy as np
import pandas as pd
from research_models import DATA, ROOT, TRAIN_END, build_feature_frame, load_features

HISTORY = ROOT.parent / "data/citibike_2022_daily.csv"
HISTORY_AUDIT = ROOT.parent / "audit/citibike2022/audit.json"
EXPECTED_CHECKS = [("2023-04-29","2023-05-19"),("2023-05-20","2023-06-09"),("2023-06-10","2023-06-30")]

def load_extended_features(allow_validation=False):
    audit=json.loads(HISTORY_AUDIT.read_text(encoding="utf-8"))
    digest=hashlib.sha256(HISTORY.read_bytes()).hexdigest()
    assert digest==audit["daily_sha256"]
    prior=pd.read_csv(HISTORY,parse_dates=["date"]).set_index("date")
    assert prior.index.equals(pd.date_range("2022-01-01","2022-12-31")) and len(prior)==365
    current=pd.read_csv(DATA,parse_dates=["date"]).set_index("date")
    assert current.index.equals(pd.date_range("2023-01-01","2023-09-30"))
    if not allow_validation:
        current=current.loc[:TRAIN_END].copy()
    daily=pd.concat([prior,current])
    assert daily.index.equals(pd.date_range("2022-01-01",daily.index[-1]))
    X=build_feature_frame(daily).iloc[28:].copy()
    y=daily.trips.loc[X.index]
    assert len(X)==(610 if allow_validation else 518) and np.isfinite(X.to_numpy()).all()
    ordinary_X, ordinary_y,_=load_features()
    pd.testing.assert_frame_equal(X.loc[ordinary_X.index],ordinary_X)
    pd.testing.assert_series_equal(y.loc[ordinary_y.index],ordinary_y)
    return X,y,{"2022_daily_sha256":digest,"2023_development_sha256":hashlib.sha256(DATA.read_bytes()).hexdigest()}

def load_extended_weather_features(allow_validation=False):
    from weather_search import add_weather_definitions,build_weather_frame,DERIVED_WEATHER
    X,y,digests=load_extended_features(allow_validation=allow_validation)
    temperature=ROOT.parent/"audit/weather/forecast_daily_temperature_expanded.csv"
    rain_wind=ROOT.parent/"audit/weather/forecast_daily_jma_precip_wind_expanded.csv"
    forecast=build_weather_frame(temperature,rain_wind)
    assert forecast.index.equals(pd.date_range("2022-01-01","2023-09-30"))
    X=X.join(forecast)
    assert X[DERIVED_WEATHER].notna().all().all()
    add_weather_definitions()
    digests.update(temperature_sha256=hashlib.sha256(temperature.read_bytes()).hexdigest(),rain_wind_sha256=hashlib.sha256(rain_wind.read_bytes()).hexdigest())
    return X,y,digests
