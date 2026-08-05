from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier

from backend.ml.evaluation import calculate_evaluation_metrics


class CalibratedModelWrapper:
    """
    Wrapper around scikit-learn models (base estimator or CalibratedClassifierCV)
    to guarantee safe, explicit extraction of Human class probability (P(Y=1)).

    Exposes predict_human_probability(features) safely inspecting classes_.
    """

    def __init__(self, model: object, calibration_method: str = "uncalibrated") -> None:
        self.model = model
        self.calibration_method = calibration_method

    @property
    def classes_(self) -> np.ndarray:
        if hasattr(self.model, "classes_"):
            return getattr(self.model, "classes_")
        elif hasattr(self.model, "estimator") and hasattr(self.model.estimator, "classes_"):
            return getattr(self.model.estimator, "classes_")
        return np.array([0, 1])

    def predict_human_probability(self, features: List[float] | np.ndarray) -> float:
        """
        Predict human probability for a single 9-element feature vector or array.
        Explicitly identifies the index corresponding to class 1 (Human).
        """
        arr = np.array(features, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        probas = self.model.predict_proba(arr)[0]

        # Explicitly locate class 1 (Human)
        classes = list(self.classes_)
        if 1 in classes:
            human_idx = classes.index(1)
        else:
            human_idx = 1 if len(probas) > 1 else 0

        prob = float(probas[human_idx])
        # Ensure result strictly within [0.0, 1.0]
        return max(0.0, min(1.0, prob))

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Predict human probability vector for a 2D batch array X.
        """
        probas = self.model.predict_proba(X)
        classes = list(self.classes_)
        if 1 in classes:
            human_idx = classes.index(1)
        else:
            human_idx = 1 if probas.shape[1] > 1 else 0

        probs = probas[:, human_idx].astype(float)
        return np.clip(probs, 0.0, 1.0)


def train_and_evaluate_calibration(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> Tuple[Dict[str, CalibratedModelWrapper], Dict[str, Any]]:
    """
    Train base Random Forest on X_train, fit calibrators on held-out X_calib,
    and evaluate Uncalibrated, Sigmoid, and Isotonic approaches on X_test.

    Prevents calibration data leakage.
    """
    logger.info("Training base RandomForest on {} training samples...", len(X_train))
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    # 1. Uncalibrated Model
    raw_wrapper = CalibratedModelWrapper(rf, calibration_method="uncalibrated")

    # 2. Sigmoid (Platt Scaling) Calibration on held-out calibration split
    logger.info("Fitting Sigmoid (Platt) calibrator on {} calibration samples...", len(X_calib))
    sigmoid_cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv="prefit")
    sigmoid_cal.fit(X_calib, y_calib)
    sigmoid_wrapper = CalibratedModelWrapper(sigmoid_cal, calibration_method="sigmoid")

    # 3. Isotonic Calibration on held-out calibration split
    logger.info("Fitting Isotonic calibrator on {} calibration samples...", len(X_calib))
    isotonic_cal = CalibratedClassifierCV(estimator=rf, method="isotonic", cv="prefit")
    isotonic_cal.fit(X_calib, y_calib)
    isotonic_wrapper = CalibratedModelWrapper(isotonic_cal, calibration_method="isotonic")

    models = {
        "uncalibrated": raw_wrapper,
        "sigmoid": sigmoid_wrapper,
        "isotonic": isotonic_wrapper,
    }

    # Evaluate all 3 methods on untouched X_test
    results: Dict[str, Any] = {}
    for name, wrapper in models.items():
        y_prob = wrapper.predict_proba_batch(X_test)
        metrics = calculate_evaluation_metrics(y_test, y_prob, threshold=0.5)

        # Compute Reliability Curve Bins (10 bins)
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
        reliability_bins = [
            {"predicted_prob": float(p_pred), "actual_freq": float(p_true)}
            for p_pred, p_true in zip(prob_pred, prob_true)
        ]

        metrics["reliability_curve_bins"] = reliability_bins
        metrics["probability_stats"] = {
            "min": float(np.min(y_prob)),
            "max": float(np.max(y_prob)),
            "mean": float(np.mean(y_prob)),
            "std": float(np.std(y_prob)),
            "median": float(np.median(y_prob)),
        }
        results[name] = metrics

    return models, results
