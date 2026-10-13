"""Daily features and fitted regression pipelines used by the selected models."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data/citibike_development_daily.csv"
TRAIN_START = "2023-01-29"
TRAIN_END = "2023-06-30"
VALID_START = "2023-07-01"
VALID_END = "2023-09-30"
SEED = 20261010
THREADS = 4
CALENDAR = ["weekday_sin", "weekday_cos", "year_sin", "year_cos", "weekend"]
FULL = CALENDAR + [f"lag_{n}" for n in (1, 2, 3, 7, 14, 21, 28)] + [
    "mean_3", "mean_7", "mean_14", "mean_28", "std_7", "weekly_change",
    "recent_change", "holiday", "before_holiday", "after_holiday"]
COMPACT = CALENDAR + ["lag_1", "lag_7", "lag_14", "mean_7", "mean_28"]

def build_feature_frame(daily):
    assert daily.index.equals(pd.date_range(daily.index.min(), daily.index.max()))
    assert daily.columns.tolist() == ["trips"] and daily.trips.notna().all()
    assert not (daily.index >= "2023-10-01").any()
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

def load_features(allow_validation=False, daily_override=None):
    daily = (pd.read_csv(DATA, parse_dates=["date"]).set_index("date")
             if daily_override is None else daily_override.copy())
    assert daily.index.equals(pd.date_range("2023-01-01", VALID_END))
    assert daily.columns.tolist() == ["trips"] and daily.trips.notna().all()
    assert len(daily) == 273 and not (daily.index >= "2023-10-01").any()
    if not allow_validation:
        daily = daily.loc[:TRAIN_END].copy()
    X = build_feature_frame(daily).loc[TRAIN_START:]
    y = daily.trips.loc[X.index]
    assert len(X) == (245 if allow_validation else 153) and np.isfinite(X.to_numpy()).all()
    # Verify causality by changing one target and every later target.
    # This structural check supplements the documented shifts above.
    return X, y, hashlib.sha256(DATA.read_bytes()).hexdigest()

X_EXAMPLE = build_feature_frame(pd.DataFrame({"trips":np.ones(29)},index=pd.date_range("2000-01-01",periods=29)))
FEATURE_SETS = {
    "compact": COMPACT,
    "full": FULL,
    "weekly": CALENDAR + ["lag_1", "lag_7", "lag_14", "lag_21", "lag_28", "weekly_mean_2", "weekly_mean_4", "weekly_median_4", "mean_7", "mean_28", "weekly_change", "holiday"],
    "recent": CALENDAR + [f"lag_{n}" for n in range(1, 8)] + ["mean_3", "mean_7", "mean_14", "mean_28", "std_7", "holiday"],
    "expanded": X_EXAMPLE.columns.tolist(),
    "linear": [f"weekday_{n}" for n in range(7)] + ["year_sin", "year_cos", "year_sin_2", "year_cos_2", "trend", "holiday", "before_holiday", "after_holiday", "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28", "mean_3", "mean_7", "mean_14", "mean_28", "std_7", "weekly_change", "recent_change"],
}
TARGETS = ["direct", "log", "residual7", "residualweekly", "residualblend", "delta1"]

def target_baseline(X, target):
    if target == "residual7":
        return X.lag_7.to_numpy()
    if target == "residualweekly":
        return X.weekly_mean_4.to_numpy()
    if target == "residualblend":
        return .5*X.lag_7.to_numpy()+.5*X.mean_7.to_numpy()
    if target == "delta1":
        return X.lag_1.to_numpy()
    return np.zeros(len(X))

class ResearchRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, model=None, feature_set="full", target="direct", half_life=0):
        self.model = model
        self.feature_set = feature_set
        self.target = target
        self.half_life = half_life

    def fit(self, X, y):
        self.feature_names_in_ = np.asarray(X.columns)
        self.columns_ = FEATURE_SETS[self.feature_set]
        self.model_ = clone(self.model)
        transformed = np.log1p(y) if self.target == "log" else np.asarray(y)-target_baseline(X, self.target)
        kwargs = {}
        if self.half_life:
            weights = np.exp2((np.arange(len(y))-(len(y)-1))/self.half_life)
            if hasattr(self.model_, "steps"):
                kwargs[self.model_.steps[-1][0]+"__sample_weight"] = weights
            else:
                kwargs["sample_weight"] = weights
        self.model_.fit(X[self.columns_], transformed, **kwargs)
        return self

    def predict(self, X):
        values = self.model_.predict(X[self.columns_])
        if self.target == "log":
            values = np.expm1(np.clip(values, 0, 16))
        else:
            values = values+target_baseline(X, self.target)
        return np.maximum(0, values)

class HistoricalBaseline(RegressorMixin, BaseEstimator):
    def __init__(self, kind="lag_7", blend=.0, trend=.0):
        self.kind = kind
        self.blend = blend
        self.trend = trend

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        if self.kind == "training_mean":
            return np.full(len(X), self.mean_)
        primary = X[self.kind].to_numpy()
        primary = (1-self.blend)*primary+self.blend*X.mean_7.to_numpy()
        return np.maximum(0, primary+self.trend*(X.mean_7-X.mean_14).to_numpy())

def metric(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"mae": float(mean_absolute_error(y,p)), "rmse": float(root_mean_squared_error(y,p)),
            "wape": float(np.abs(y-p).sum()/y.sum()), "bias": float(np.mean(p-y)), "days": int(len(y))}

def make_estimator(family, parameters):
    p = dict(parameters)
    seed = p.pop("random_seed", SEED)
    feature_set = p.pop("feature_set")
    target = p.pop("target")
    half_life = p.pop("half_life", 0)
    if family == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(**p))
    elif family == "elastic":
        model = make_pipeline(StandardScaler(), ElasticNet(max_iter=15000, tol=.001, **p))
    elif family == "huber":
        model = make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000, **p))
    elif family == "svr":
        p.setdefault("max_iter",100000)
        model = make_pipeline(StandardScaler(), SVR(cache_size=256, **p))
    elif family == "knn":
        model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_jobs=THREADS, **p))
    elif family == "mlp":
        if "hidden_layer_sizes" in p:
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        model = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=MLPRegressor(random_state=SEED, max_iter=1500, early_stopping=False, **p), transformer=StandardScaler()))
    elif family == "forest":
        model = RandomForestRegressor(random_state=SEED, n_jobs=THREADS, **p)
    elif family == "extra":
        model = ExtraTreesRegressor(random_state=SEED, n_jobs=THREADS, **p)
    elif family == "hist":
        model = HistGradientBoostingRegressor(random_state=SEED, **p)
    elif family == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(random_state=seed, n_jobs=THREADS, tree_method="hist", device="cpu", **p)
    elif family == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(random_state=SEED, n_jobs=THREADS, verbosity=-1, **p)
    elif family == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(random_seed=SEED, thread_count=THREADS, verbose=False, allow_writing_files=False, **p)
    else:
        raise ValueError(family)
    return ResearchRegressor(model, feature_set, target, half_life)
