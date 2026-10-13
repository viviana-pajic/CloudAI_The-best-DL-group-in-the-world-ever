"""Same count features as research, with future dates allowed for inference."""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

def build_feature_frame(daily):
    assert daily.index.equals(pd.date_range(daily.index.min(), daily.index.max()))
    assert daily.columns.tolist() == ["trips"] and daily.trips.notna().all()
    index = daily.index
    X = pd.DataFrame(index=index)
    day = index.dayofweek
    fraction = (index.dayofyear - 1) / np.where(index.is_leap_year,366.,365.)
    X["weekday_sin"] = np.sin(2*np.pi*day/7)
    X["weekday_cos"] = np.cos(2*np.pi*day/7)
    X["year_sin"] = np.sin(2*np.pi*fraction)
    X["year_cos"] = np.cos(2*np.pi*fraction)
    X["weekend"] = (day >= 5).astype(int)
    holidays = USFederalHolidayCalendar().holidays(index.min()-pd.Timedelta(days=1), index.max()+pd.Timedelta(days=1))
    X["holiday"] = index.isin(holidays).astype(int)
    X["before_holiday"] = (index+pd.Timedelta(days=1)).isin(holidays).astype(int)
    X["after_holiday"] = (index-pd.Timedelta(days=1)).isin(holidays).astype(int)
    prior = daily.trips.shift(1)
    for n in range(1, 29):
        X[f"lag_{n}"] = daily.trips.shift(n)
    for n in (2, 3, 5, 7, 10, 14, 21, 28):
        X[f"mean_{n}"] = prior.rolling(n, min_periods=n).mean()
        X[f"median_{n}"] = prior.rolling(n, min_periods=n).median()
        X[f"std_{n}"] = prior.rolling(n, min_periods=n).std(ddof=0)
    X["weekly_change"] = X.lag_7-X.lag_14
    X["recent_change"] = X.lag_1-X.lag_7
    X["weekly_mean_2"] = X[["lag_7", "lag_14"]].mean(axis=1)
    X["weekly_mean_3"] = X[["lag_7", "lag_14", "lag_21"]].mean(axis=1)
    X["weekly_mean_4"] = X[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1)
    X["weekly_median_4"] = X[["lag_7", "lag_14", "lag_21", "lag_28"]].median(axis=1)
    X["mean_change"] = X.mean_7-X.mean_14
    X["relative_recent"] = X.lag_1/X.mean_7
    X["relative_weekly"] = X.lag_7/X.mean_28
    X["recent_low"] = prior.rolling(7).min()
    X["recent_high"] = prior.rolling(7).max()
    X["trend"] = (index-pd.Timestamp("2023-01-01")).days/365.
    for n in range(7):
        X[f"weekday_{n}"] = (day == n).astype(int)
    for n in (2, 3):
        X[f"year_sin_{n}"] = np.sin(n*2*np.pi*fraction)
        X[f"year_cos_{n}"] = np.cos(n*2*np.pi*fraction)
    return X
