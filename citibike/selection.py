"""Construct the fixed two-model blends used by the app and AWS notebooks."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from research_models import make_estimator


class OOFBlend(RegressorMixin,BaseEstimator):
    def __init__(self,members=None,weights=None):
        self.members=members
        self.weights=weights

    def fit(self,X,y):
        self.models_=[clone(model).fit(X,y) for model in self.members]
        return self

    def predict(self,X):
        values=np.column_stack([model.predict(X) for model in self.models_])
        return np.maximum(0,values@np.asarray(self.weights))

def underlying_family(family):
    for prefix in ["sequence_history_weather_","sequence_history_","normalized_history_weather_","normalized_history_","history_weather_","history_","weather_"]:
        if family.startswith(prefix):
            return family[len(prefix):]
    return family

def estimator_for_record(record):
    if record['family'] == 'blend':
        config = record['configuration']
        return OOFBlend([estimator_for_record(r) for r in config['members']], config['weights'])
    if record['family'].startswith('sequence_'):
        from sequence_search import sequence_estimator
        return sequence_estimator(record['configuration'])
    return make_estimator(underlying_family(record['family']), record['configuration'])
