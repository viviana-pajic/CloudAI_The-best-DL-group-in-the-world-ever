"""Daily archived forecast inputs. No observed target-day weather."""
import hashlib
import numpy as np
import pandas as pd
from research_models import FEATURE_SETS, ROOT, load_features

WEATHER = ROOT.parent / "audit/weather/forecast_daily_temperature.csv"
RAIN_WIND = ROOT.parent / "audit/weather/forecast_daily_jma_precip_wind.csv"
ORIGINAL_SETS = {k:list(v) for k,v in FEATURE_SETS.items()}
TEMPERATURE_COLUMNS = ["forecast_temp_day2_mean", "forecast_temp_day2_min", "forecast_temp_day2_max"]
RAIN_WIND_COLUMNS = ["forecast_jma_day2_precip_sum", "forecast_jma_day2_wind_mean", "forecast_jma_day2_wind_max"]
BASE_WEATHER = TEMPERATURE_COLUMNS+RAIN_WIND_COLUMNS
DERIVED_WEATHER = BASE_WEATHER+["forecast_temperature_range", "forecast_temperature_change_1", "forecast_temperature_change_7", "forecast_temperature_weekend", "forecast_warm_degree", "forecast_cold_degree", "forecast_rain_log", "forecast_rain_change_7", "forecast_wind_change_7"]

def add_weather_definitions():
    for base in ["compact","full","weekly","linear"]:
        FEATURE_SETS["weather_"+base] = ORIGINAL_SETS[base]+BASE_WEATHER
        FEATURE_SETS["weather_derived_"+base] = ORIGINAL_SETS[base]+DERIVED_WEATHER

def build_weather_frame(temperature_path=WEATHER,rain_wind_path=RAIN_WIND):
    forecast = pd.read_csv(temperature_path,parse_dates=["date"]).set_index("date")
    assert forecast.index.equals(pd.date_range(forecast.index[0],forecast.index[-1]))
    assert forecast.index[-1] <= pd.Timestamp("2023-09-30")
    assert forecast[TEMPERATURE_COLUMNS].notna().all().all()
    forecast = forecast[TEMPERATURE_COLUMNS].copy()
    rain_wind = pd.read_csv(rain_wind_path,parse_dates=["date"]).set_index("date")
    assert rain_wind.index.equals(forecast.index)
    assert (rain_wind[RAIN_WIND_COLUMNS]>=0).all().all()
    forecast = forecast.join(rain_wind[RAIN_WIND_COLUMNS])
    forecast["forecast_temperature_range"] = forecast.forecast_temp_day2_max-forecast.forecast_temp_day2_min
    forecast["forecast_temperature_change_1"] = forecast.forecast_temp_day2_mean.diff(1)
    forecast["forecast_temperature_change_7"] = forecast.forecast_temp_day2_mean.diff(7)
    forecast["forecast_temperature_weekend"] = forecast.forecast_temp_day2_mean*(forecast.index.dayofweek>=5).astype(int)
    forecast["forecast_warm_degree"] = np.maximum(0,forecast.forecast_temp_day2_mean-18.)
    forecast["forecast_cold_degree"] = np.maximum(0,10.-forecast.forecast_temp_day2_mean)
    forecast["forecast_rain_log"] = np.log1p(forecast.forecast_jma_day2_precip_sum)
    forecast["forecast_rain_change_7"] = forecast.forecast_jma_day2_precip_sum.diff(7)
    forecast["forecast_wind_change_7"] = forecast.forecast_jma_day2_wind_mean.diff(7)
    return forecast

def load_weather_features(allow_validation=False):
    X,y,digest = load_features(allow_validation=allow_validation)
    forecast = build_weather_frame()
    X = X.join(forecast)
    assert X[DERIVED_WEATHER].notna().all().all()
    add_weather_definitions()
    return X,y,digest
