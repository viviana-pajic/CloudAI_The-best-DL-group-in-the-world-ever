"""Focused checks for data isolation, preparation and actual artifact/API parity."""
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
from mushrooms import FEATURE_COLUMNS, clean, load_data, preprocessor
from mushroom_workflow import candidates, evaluate_model, feature_groups, load_development
from server import handler_for, load_bundle


class MushroomContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_data()
        cls.X, cls.y, cls.splits = load_development(cls.frame)

    def test_frozen_partition_and_collision_policy(self):
        manifest = json.loads((ROOT / "mushrooms/split_manifest.json").read_text())
        self.assertEqual(self.splits, manifest["splits"])
        self.assertEqual(len(self.X), 4000)
        self.assertEqual(len(self.splits["test"]), 1000)
        self.assertFalse(set(self.X.index) & set(self.splits["test"]))
        self.assertEqual(set(sum(self.splits.values(), [])), set(range(5000)))
        self.assertEqual(self.frame.shape[0], clean(self.frame).shape[0])
        groups = feature_groups(clean(self.frame)[FEATURE_COLUMNS])
        self.assertEqual(len(groups) - len(set(groups)), 5)
        self.assertEqual(groups[232], groups[3164])
        self.assertEqual(groups[3164], groups[4244])

    def test_evaluator_refuses_reserved_test_rows(self):
        # This fails before fitting or scoring anything.
        ids = self.splits["test"][:2]
        X = clean(self.frame.loc[ids])[FEATURE_COLUMNS]
        with self.assertRaisesRegex(ValueError, "development rows only"):
            evaluate_model("forbidden", candidates(self.X)["random_forest"],
                           X, pd.Series([0, 1], index=X.index))

    def test_imputation_and_unknown_categories_use_training_only(self):
        training = pd.DataFrame({"size": [1., 3., np.nan], "color": ["a", "a", np.nan]})
        validation = pd.DataFrame({"size": [999., np.nan], "color": ["unseen", np.nan]})
        transformer = preprocessor(training, dense=True)
        transformer.fit(training)
        before = transformer.named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
        transformed = transformer.transform(validation)
        np.testing.assert_array_equal(before, [2.])
        np.testing.assert_array_equal(transformer.named_transformers_["numeric"].named_steps["impute"].statistics_, before)
        self.assertEqual(transformed[1, 0], 2.)
        self.assertTrue(np.isfinite(transformed).all())

    def test_saved_artifact_and_api_agree_on_real_pipeline(self):
        model, schema, info = load_bundle(ROOT / "models/mushroom_candidate.joblib")
        self.assertEqual(list(schema), FEATURE_COLUMNS)
        self.assertFalse(info["final_test_evaluated"])
        sample = self.X.iloc[[0]]
        payload = {k: None if pd.isna(v) else v for k, v in sample.iloc[0].items()}
        handler_class = handler_for(model, schema, info)
        handler = object.__new__(handler_class)
        handler.path = "/predict"
        body = json.dumps(payload).encode()
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        responses = []
        handler.send = lambda status, data: responses.append((status, data))
        handler.do_POST()
        self.assertEqual(responses[0][0], 200)
        expected = model.predict_proba(sample)[0, list(model.classes_).index(1)]
        self.assertAlmostEqual(responses[0][1]["poison_probability"], expected)
        self.assertEqual(responses[0][1]["prediction"], "p" if expected >= info["threshold"] else "e")


if __name__ == "__main__":
    unittest.main()
