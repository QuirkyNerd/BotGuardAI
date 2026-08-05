from __future__ import annotations

import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from backend.ml.calibration import CalibratedModelWrapper
from backend.ml.model import load_model, predict_human_probability, _ensure_model_loaded
from backend.services.feature_engineering import FEATURE_NAMES


class TestProbabilityCalibration(unittest.TestCase):

    def setUp(self) -> None:
        # Create a small dummy estimator
        X = np.random.randn(50, 9)
        y = np.array([0] * 25 + [1] * 25)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        self.rf = rf
        self.wrapper = CalibratedModelWrapper(rf, calibration_method="test")

    def test_predict_human_probability_range(self) -> None:
        sample = list(np.random.randn(9))
        prob = self.wrapper.predict_human_probability(sample)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_predict_human_class_indexing(self) -> None:
        # Ensure class 1 probability is explicitly extracted even if classes_ order is reversed
        self.assertEqual(list(self.wrapper.classes_), [0, 1])
        sample = list(np.random.randn(9))
        raw_proba = self.rf.predict_proba(np.array(sample).reshape(1, -1))[0]
        prob = self.wrapper.predict_human_probability(sample)
        self.assertAlmostEqual(prob, float(raw_proba[1]), places=5)

    def test_model_loader_integration(self) -> None:
        load_model()
        wrapper = _ensure_model_loaded()
        self.assertIsNotNone(wrapper)
        dummy_features = [100.0, 50.0, 0.5, 0.2, 0.1, 50.0, 10.0, 5.0, 2.0]
        prob = predict_human_probability(dummy_features)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


if __name__ == "__main__":
    unittest.main()
