"""Check forecasting information boundaries and chronological evaluation."""
from pathlib import Path
import sys
import unittest

if sys.version_info[:2] != (3, 12):
    raise unittest.SkipTest('Run Citi Bike checks in its Python 3.12 environment.')

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from citibike_forecast import (COMPACT, ForecastRegressor, chronological_cv, development_data,
                               feature_table, prediction_row, validate_daily)


class ForecastBoundaryTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2023-01-01', periods=80)
        self.daily = pd.DataFrame({'trips': np.arange(80) + 100}, index=dates)

    def test_target_and_future_counts_cannot_change_a_forecast_row(self):
        original = feature_table(self.daily)
        date = self.daily.index[40]
        changed = self.daily.copy()
        changed.loc[date:, 'trips'] = 999999
        pd.testing.assert_series_equal(original.loc[date], feature_table(changed).loc[date])
        self.assertEqual(original.loc[date, 'lag_1'], self.daily.loc[date - pd.Timedelta(days=1), 'trips'])
        self.assertEqual(original.loc[date, 'mean_7'], self.daily.loc[date - pd.Timedelta(days=7):date - pd.Timedelta(days=1), 'trips'].mean())

    def test_api_history_produces_the_same_features_without_a_target(self):
        date = self.daily.index[40]
        history = self.daily.loc[date - pd.Timedelta(days=28):date - pd.Timedelta(days=1)]
        pd.testing.assert_frame_equal(prediction_row(history, date), feature_table(self.daily).loc[[date]], check_freq=False)

    def test_wrong_history_dates_and_bad_counts_are_rejected(self):
        date = self.daily.index[40]
        history = self.daily.iloc[12:40]
        for bad in [history.iloc[:-1], history.shift(1).dropna(), pd.concat([history, history.iloc[[0]]])]:
            with self.assertRaises(ValueError):
                prediction_row(bad, date)
        for count in [-1, np.nan, np.inf, 1.5]:
            bad = history.astype(float).copy()
            bad.iloc[0, 0] = count
            with self.assertRaises(ValueError):
                validate_daily(bad)

    def test_reserved_dates_are_absent_and_cv_never_trains_on_future(self):
        X, y, training, validation, _ = development_data()
        self.assertEqual((len(training), len(validation), len(X)), (153, 92, 245))
        self.assertEqual(str(training.max().date()), '2023-06-30')
        self.assertEqual(str(validation.min().date()), '2023-07-01')
        self.assertLess(X.index.max(), pd.Timestamp('2023-10-01'))
        self.assertTrue(X.index.equals(y.index))
        for fit, check in chronological_cv().split(X.loc[training]):
            self.assertLess(training[fit].max(), training[check].min())
            self.assertEqual(len(check), 21)
            self.assertFalse(set(training[fit]) & set(training[check]))

    def test_scaling_is_fitted_on_training_only(self):
        X = feature_table(self.daily).dropna()
        train = X.iloc[:20]
        model = ForecastRegressor(make_pipeline(StandardScaler(), Ridge()), 'compact').fit(train, self.daily.trips.loc[train.index])
        scaler = model.model_.named_steps['standardscaler']
        before = scaler.mean_.copy()
        np.testing.assert_allclose(before, train[COMPACT].mean())
        model.predict(X.iloc[20:] * 100)
        np.testing.assert_array_equal(before, scaler.mean_)

    def test_seasonal_residual_preserves_the_anchor_and_clone_contract(self):
        X = feature_table(self.daily).dropna()
        y = X.lag_7 + 17
        model = clone(ForecastRegressor(DummyRegressor(strategy='mean'), seasonal_residual=True)).fit(X, y)
        np.testing.assert_allclose(model.predict(X), X.lag_7 + 17)
        other = X.copy()
        other['lag_7'] += 100
        np.testing.assert_allclose(model.predict(other), other.lag_7 + 17)


if __name__ == '__main__':
    unittest.main()
