from __future__ import annotations

import unittest
import numpy as np

from backend.ml.anomaly_detector import BehavioralAnomalyDetector
from backend.services.feature_engineering import FEATURE_NAMES


class TestBehavioralAnomalyDetector(unittest.TestCase):

    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        # Synthetic human feature training set
        self.X_human_train = rng.normal(loc=10.0, scale=2.0, size=(100, 9))
        self.X_human_val = rng.normal(loc=10.0, scale=2.0, size=(25, 9))
        # Synthetic anomalous bot set
        self.X_bot = rng.normal(loc=50.0, scale=10.0, size=(25, 9))

        self.detector = BehavioralAnomalyDetector(algorithm="isolation_forest", contamination=0.03, seed=42)
        self.detector.fit(self.X_human_train, self.X_human_val)

    def test_predict_anomaly_score_range(self) -> None:
        sample = list(self.X_human_train[0])
        score = self.detector.predict_anomaly_score(sample)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_outlier_detection_higher_score(self) -> None:
        human_score = self.detector.predict_anomaly_score(self.X_human_train[0])
        bot_score = self.detector.predict_anomaly_score(self.X_bot[0])
        # Anomaly score for obvious outlier should be significantly higher
        self.assertGreater(bot_score, human_score)

    def test_deterministic_behavior(self) -> None:
        d1 = BehavioralAnomalyDetector(algorithm="isolation_forest", seed=123)
        d1.fit(self.X_human_train)

        d2 = BehavioralAnomalyDetector(algorithm="isolation_forest", seed=123)
        d2.fit(self.X_human_train)

        s1 = d1.predict_anomaly_score(self.X_human_train[5])
        s2 = d2.predict_anomaly_score(self.X_human_train[5])
        self.assertAlmostEqual(s1, s2, places=4)


if __name__ == "__main__":
    unittest.main()
