"""Stronger seasonal baseline and independently tuned random forest."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from citibike_forecast import (ROOT, REPORTS, ForecastRegressor, chronological_cv,
                               development_data, record_model, write_json)


class SeasonalAverage(RegressorMixin, BaseEstimator):
    """Average four previous matching weekdays, with a bounded recent-level correction."""
    def __init__(self, decay=1., adjustment=0.):
        self.decay = decay
        self.adjustment = adjustment

    def fit(self, X, y):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X):
        weights = self.decay ** np.arange(4)
        same_weekdays = X[['lag_7', 'lag_14', 'lag_21', 'lag_28']].to_numpy()
        base = np.average(same_weekdays, axis=1, weights=weights)
        ratio = np.clip(X.mean_7.to_numpy() / np.maximum(1., X.mean_28.to_numpy()), .5, 2.)
        return base * ratio ** self.adjustment


def run_search(family):
    X, y, train, _, _ = development_data()
    if family == 'seasonal_average':
        estimator = SeasonalAverage()
        grid = {'decay': [.5, .75, 1.], 'adjustment': [0., .5, 1.]}
    elif family == 'random_forest':
        estimator = ForecastRegressor(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1))
        grid = {'model__max_depth': [3, 6, None], 'model__min_samples_leaf': [2, 5, 10],
                'model__max_features': [.7, 1.], 'seasonal_residual': [False, True]}
    else:
        raise ValueError('Unknown search family.')
    start = time.perf_counter()
    search = GridSearchCV(estimator, grid, scoring='neg_mean_absolute_error',
                          cv=chronological_cv(), n_jobs=4, error_score='raise', return_train_score=True)
    with threadpool_limits(limits=4):
        search.fit(X.loc[train], y.loc[train])
    results = pd.DataFrame(search.cv_results_)
    results.drop(columns='params').to_csv(REPORTS / f'{family}_grid.csv', index=False)
    details = {'method': 'GridSearchCV', 'configurations': len(results), 'cv_fits': 3 * len(results),
               'best_parameters': search.best_params_, 'best_cv_mae': -float(search.best_score_),
               'elapsed_seconds': time.perf_counter() - start, 'validation_used_for_search': False}
    result = record_model(f'{family}_tuned', clone(search.best_estimator_), details)
    result['source_sha256']['citibike_additional.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(REPORTS / f'{family}_tuned_run.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=['seasonal_average', 'random_forest'])
    run_search(parser.parse_args().family)


if __name__ == '__main__':
    import citibike_additional as models
    models.main()
