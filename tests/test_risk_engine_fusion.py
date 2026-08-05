from __future__ import annotations

import unittest
import numpy as np

from backend.config import (
    COMPOSITE_ALLOW_RISK_THRESHOLD,
    COMPOSITE_CHALLENGE_RISK_THRESHOLD,
)
from backend.ml.intelligence_engine import (
    MultiEnginePrediction,
    load_intelligence_stack,
    run_multi_engine_prediction,
)
from backend.models.schemas import BehaviorBatch, BrowserMetadata, RiskLevel
from backend.security.risk_engine import compute_risk_score_v2
from backend.services.decision_engine import evaluate_session
from backend.simulation.adversarial_simulator import generate_adversarial_bot_session


class TestRiskEngineFusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        load_intelligence_stack()

    def test_multi_engine_prediction_and_range_bounds(self) -> None:
        dummy_features = [500.0, 2e8, 0.6, 0.15, 0.005, 2500.0, 1000.0, 6.0, 1.0]
        batch = generate_adversarial_bot_session(level=5, session_id="test_fusion", seed=42)

        pred = run_multi_engine_prediction(dummy_features, batch=batch)

        self.assertTrue(pred.rf_available)
        self.assertTrue(pred.anomaly_available)
        self.assertTrue(pred.temporal_available)

        self.assertGreaterEqual(pred.human_probability, 0.0)
        self.assertLessEqual(pred.human_probability, 1.0)

        self.assertGreaterEqual(pred.anomaly_score, 0.0)
        self.assertLessEqual(pred.anomaly_score, 100.0)

        self.assertGreaterEqual(pred.temporal_human_probability, 0.0)
        self.assertLessEqual(pred.temporal_human_probability, 1.0)

    def test_risk_engine_v2_component_bounds_and_overrides(self) -> None:
        dummy_features = [500.0, 2e8, 0.6, 0.15, 0.005, 2500.0, 1000.0, 6.0, 1.0]
        pred = MultiEnginePrediction(
            rf_available=True, human_probability=0.90, rf_risk=10.0,
            anomaly_available=True, anomaly_score=20.0, anomaly_risk=20.0, is_anomaly=False,
            temporal_available=True, temporal_human_probability=0.95, temporal_risk=5.0,
        )

        res = compute_risk_score_v2(pred, dummy_features)
        self.assertGreaterEqual(res.composite_risk_score, 0.0)
        self.assertLessEqual(res.composite_risk_score, 100.0)
        self.assertIn("behavioral_ml", res.risk_components)
        self.assertIn("anomaly", res.risk_components)
        self.assertIn("temporal", res.risk_components)

        # Test Webdriver Escalation Override
        meta_wd = BrowserMetadata(
            user_agent="HeadlessChrome",
            webdriver=True,
            touch_points=0,
            device_entropy=1000.0,
        )
        res_wd = compute_risk_score_v2(pred, dummy_features, browser_metadata=meta_wd)
        self.assertGreaterEqual(res_wd.composite_risk_score, 85.0)
        self.assertIn("escalation_override_webdriver", res_wd.triggered_indicators)

    def test_missing_model_fault_tolerance(self) -> None:
        # Simulate RF only available, Anomaly/Temporal missing
        pred_partial = MultiEnginePrediction(
            rf_available=True, human_probability=0.20, rf_risk=80.0,
            anomaly_available=False,
            temporal_available=False,
        )
        dummy_features = [500.0, 2e8, 0.6, 0.15, 0.005, 2500.0, 1000.0, 6.0, 1.0]
        res = compute_risk_score_v2(pred_partial, dummy_features)
        self.assertGreaterEqual(res.composite_risk_score, 50.0)

    def test_decision_engine_mapping_and_schema_compatibility(self) -> None:
        dummy_features = [500.0, 2e8, 0.6, 0.15, 0.005, 2500.0, 1000.0, 6.0, 1.0]
        batch = generate_adversarial_bot_session(level=1, session_id="test_decision", seed=42)

        resp = evaluate_session("test_decision", dummy_features, batch=batch)

        self.assertEqual(resp.session_id, "test_decision")
        self.assertIn(resp.recommended_action, ["allow", "challenge", "block"])
        self.assertIsInstance(resp.risk_level, RiskLevel)

        # Verify expanded fields present
        self.assertIsNotNone(resp.anomaly_score)
        self.assertIsNotNone(resp.temporal_human_probability)
        self.assertIsNotNone(resp.risk_components)
        self.assertIsNotNone(resp.triggered_indicators)


if __name__ == "__main__":
    unittest.main()
