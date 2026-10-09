"""Check threshold ties, held-out calibration and persisted experiment provenance."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mushroom_error_analysis as analysis
from mushroom_workflow import REPORTS, load_development


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    fits = []
    predictions = []

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.training_ids_ = set(X.index)
        type(self).fits.append(self.training_ids_)
        return self

    def predict_proba(self, X):
        type(self).predictions.append((self.training_ids_, set(X.index)))
        p = .05 + .9 * (X.index.to_numpy() % 11) / 10
        return np.column_stack([1 - p, p])


class ThresholdTests(unittest.TestCase):
    def test_tied_scores_and_recall_constraint(self):
        y, p = [1, 1, 0, 0], [.8, .4, .4, .2]
        threshold = analysis.select_threshold(y, p, 1.)
        self.assertEqual(threshold, .4)
        measured = analysis.operating_metrics(y, p, threshold)
        self.assertEqual((measured['tn'], measured['fp'], measured['fn'], measured['tp']), (1, 1, 0, 2))
        table = analysis.threshold_table(y, p)
        self.assertEqual(table.threshold.eq(.4).sum(), 1)
        for row in table.itertuples():
            metrics = analysis.operating_metrics(y, p, row.threshold)
            for key in ['tn', 'fp', 'fn', 'tp', 'poison_recall', 'specificity', 'poison_precision']:
                self.assertAlmostEqual(metrics[key], getattr(row, key))

    def test_bad_calibration_inputs_rejected(self):
        for p in [[float('nan'), .5], [-.1, .5], [.2, 1.1]]:
            with self.assertRaises(ValueError):
                analysis.select_threshold([0, 1], p)
        with self.assertRaises(ValueError):
            analysis.select_threshold([0, 0], [.2, .5])
        with self.assertRaises(ValueError):
            analysis.select_threshold([0, 1], [.2, .5], 0)

    def test_nested_policy_keeps_outer_labels_out_of_threshold_selection(self):
        X, y, splits = load_development()
        RecordingClassifier.fits = []
        RecordingClassifier.predictions = []
        selector = analysis.select_threshold
        calibration_ids = []

        def record_selection(labels, probabilities):
            calibration_ids.append(set(labels.index))
            return selector(labels, probabilities)

        with patch.object(analysis, 'select_threshold', side_effect=record_selection):
            folds, predictions, audit = analysis.evaluate_threshold_policy(X, y, RecordingClassifier())
        self.assertEqual(len(RecordingClassifier.fits), 20)
        self.assertEqual(len(folds), 5)
        self.assertEqual(set(predictions.source_row), set(X.index))
        for fold, record in enumerate(audit):
            validation = set(record['outer_validation_source_rows'])
            training = set(X.index) - validation
            self.assertEqual(calibration_ids[fold], training)
            self.assertEqual(RecordingClassifier.fits[4 * fold + 3], training)
            for fitted in RecordingClassifier.fits[4 * fold:4 * fold + 3]:
                self.assertTrue(fitted < training)
                self.assertFalse(fitted & validation)
        for training, predicted in RecordingClassifier.predictions:
            self.assertFalse(training & predicted)
            self.assertFalse((training | predicted) & set(splits['test']))
        forbidden = X.iloc[:2].copy()
        forbidden.index = splits['test'][:2]
        with self.assertRaisesRegex(ValueError, 'development rows only'):
            analysis.evaluate_threshold_policy(forbidden, pd.Series([0, 1], index=forbidden.index), RecordingClassifier())

    def test_error_slices_use_class_denominators(self):
        data = pd.DataFrame({'label': [1, 0, 1, 1, 0, 0], 'missing_feature_count': [0, 2, 3, 4, 5, 10]})
        slices = analysis.error_slices(data, [0, 1, 0, 1, 0, 1], 'example', 'fixed').set_index('missing_features')
        self.assertEqual(slices.loc['3-4', 'false_negative_rate'], .5)
        self.assertEqual(slices.loc['5-10', 'false_positive_rate'], .5)
        self.assertTrue(pd.isna(slices.loc['5-10', 'false_negative_rate']))

    def test_recorded_results_match_source_and_frozen_development(self):
        X, y, splits = load_development()
        for name in ['baseline', 'candidates', 'threshold']:
            report = json.loads((REPORTS / f'{name}_run.json').read_text())
            for source, digest in report['source_sha256'].items():
                self.assertEqual(hashlib.sha256((ROOT / 'src' / source).read_bytes()).hexdigest(), digest)
            self.assertFalse(report['contract']['test_model_predictions_computed'])
        report = json.loads((REPORTS / 'threshold_run.json').read_text())
        ids = [i for fold in report['outer_folds'] for i in fold['outer_validation_source_rows']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(X.index))
        self.assertFalse(set(ids) & set(splits['test']))
        self.assertFalse(report['test_model_predictions_computed'])


if __name__ == '__main__':
    unittest.main()
