from __future__ import annotations

import unittest
import numpy as np

from backend.services.feature_engineering import FEATURE_NAMES, compute_features_from_batches
from backend.simulation.adversarial_simulator import (
    cubic_bezier,
    generate_bezier_trajectory,
    generate_adversarial_bot_session,
)


class TestAdversarialBenchmark(unittest.TestCase):

    def test_cubic_bezier_endpoints(self) -> None:
        p0 = (0.0, 0.0)
        p1 = (10.0, 20.0)
        p2 = (30.0, 40.0)
        p3 = (50.0, 50.0)

        # B(0) should equal P0
        b_start = cubic_bezier(p0, p1, p2, p3, 0.0)
        self.assertAlmostEqual(b_start[0], p0[0], places=5)
        self.assertAlmostEqual(b_start[1], p0[1], places=5)

        # B(1) should equal P3
        b_end = cubic_bezier(p0, p1, p2, p3, 1.0)
        self.assertAlmostEqual(b_end[0], p3[0], places=5)
        self.assertAlmostEqual(b_end[1], p3[1], places=5)

    def test_bezier_trajectory_generation(self) -> None:
        rng = np.random.default_rng(42)
        p0 = (100.0, 100.0)
        p3 = (500.0, 500.0)
        pts, dts = generate_bezier_trajectory(rng, p0, p3, n_points=20, with_overshoot=False)

        self.assertEqual(len(pts), 20)
        self.assertEqual(len(dts), 20)
        for dt in dts:
            self.assertGreater(dt, 0.0)

    def test_adversarial_bot_levels_feature_compatibility(self) -> None:
        """
        Verify every adversarial attack level generates valid raw event streams
        that pass cleanly through the production feature engineering pipeline.
        """
        for lvl in range(1, 6):
            batch = generate_adversarial_bot_session(level=lvl, session_id=f"test_l{lvl}", seed=42)
            self.assertEqual(batch.session_id, f"test_l{lvl}")

            # Pass through production feature engineering
            fv = compute_features_from_batches(batch.session_id, [batch])
            self.assertEqual(len(fv.values), 9)
            self.assertEqual(len(fv.values), len(FEATURE_NAMES))

            # Ensure all feature values are numeric non-NaN floats
            for name, val in zip(FEATURE_NAMES, fv.values):
                self.assertFalse(np.isnan(val), f"Feature {name} was NaN for Level {lvl}")
                self.assertFalse(np.isinf(val), f"Feature {name} was Inf for Level {lvl}")

    def test_deterministic_reproducibility(self) -> None:
        b1 = generate_adversarial_bot_session(level=5, session_id="repeat_test", seed=123)
        b2 = generate_adversarial_bot_session(level=5, session_id="repeat_test", seed=123)

        fv1 = compute_features_from_batches("repeat_test", [b1])
        fv2 = compute_features_from_batches("repeat_test", [b2])

        np.testing.assert_allclose(fv1.values, fv2.values, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
