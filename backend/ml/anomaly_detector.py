from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from backend.services.feature_engineering import FEATURE_NAMES


class BehavioralAnomalyDetector:
    """
    Unsupervised behavioral anomaly detector trained exclusively on human behavioral data.
    Provides normalized anomaly score S in [0, 100] where:
      0 = strongly consistent with learned human distribution
      100 = strongly anomalous behavior / outlier
    """

    def __init__(
        self,
        algorithm: str = "isolation_forest",
        contamination: float = 0.03,
        seed: int = 42,
    ) -> None:
        self.algorithm = algorithm.lower()
        self.contamination = contamination
        self.seed = seed

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[Any] = None

        # Score normalization statistics (fitted purely on human training set)
        self.q_min: float = 0.0
        self.q_max: float = 1.0
        self.anomaly_threshold: float = 50.0

    def fit(self, X_human_train: np.ndarray, X_human_val: Optional[np.ndarray] = None) -> None:
        """
        Fit scaler, anomaly model, and normalization quantiles strictly on human-only data.
        """
        logger.info("Fitting {} on {} human-only training samples...", self.algorithm, len(X_human_train))

        # Standardize features for One-Class SVM; optional for Isolation Forest but useful for consistency
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_human_train)

        if self.algorithm == "one_class_svm":
            self.model = OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=self.contamination,
            )
            self.model.fit(X_scaled)
        else:  # Default: Isolation Forest
            self.model = IsolationForest(
                n_estimators=200,
                contamination=self.contamination,
                random_state=self.seed,
                n_jobs=-1,
            )
            self.model.fit(X_scaled)

        # Compute raw scores on human training data to establish baseline normalization bounds
        raw_scores_train = self._compute_raw_anomaly_score(X_scaled)

        # Set quantiles: 1st percentile as 0, 99th percentile as 80
        self.q_min = float(np.percentile(raw_scores_train, 1))
        self.q_max = float(np.percentile(raw_scores_train, 99))
        if abs(self.q_max - self.q_min) < 1e-6:
            self.q_max = self.q_min + 1.0

        # Evaluate on validation split to establish 3% human false anomaly threshold
        if X_human_val is not None:
            val_scores = self.predict_anomaly_score_batch(X_human_val)
            self.anomaly_threshold = float(np.percentile(val_scores, (1.0 - self.contamination) * 100.0))
            logger.info("Established human anomaly threshold = {:.2f} (Target Human False Anomaly Rate ~ {:.1f}%)",
                        self.anomaly_threshold, self.contamination * 100.0)

    def _compute_raw_anomaly_score(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Extract raw decision function scores where HIGHER values mean MORE ANOMALOUS.
        """
        if self.algorithm == "one_class_svm":
            # OneClassSVM decision_function: positive inside boundary, negative outside boundary
            return -self.model.decision_function(X_scaled)
        else:
            # IsolationForest score_samples: negative values; more negative = more anomalous
            return -self.model.score_samples(X_scaled)

    def predict_anomaly_score(self, features: List[float] | np.ndarray) -> float:
        """
        Compute normalized anomaly score S in [0, 100] for a single feature vector.
        """
        arr = np.array(features, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        scores = self.predict_anomaly_score_batch(arr)
        return float(scores[0])

    def predict_anomaly_score_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Compute normalized anomaly scores S in [0, 100] for a 2D feature matrix.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Anomaly detector is not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        raw = self._compute_raw_anomaly_score(X_scaled)

        # Linear mapping into [0, 100] using human training quantiles
        normalized = (raw - self.q_min) / (self.q_max - self.q_min) * 75.0
        # Clamp to [0, 100]
        return np.clip(normalized, 0.0, 100.0)

    def is_anomalous(self, features: List[float] | np.ndarray) -> bool:
        """
        Check whether feature vector exceeds the human anomaly threshold.
        """
        score = self.predict_anomaly_score(features)
        return score >= self.anomaly_threshold

    def evaluate_inference_latency(self, X_sample: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark average and P95 inference latency per session.
        """
        latencies_ms: List[float] = []
        for i in range(num_runs):
            row = X_sample[i % len(X_sample)].reshape(1, -1)
            t0 = time.perf_counter()
            _ = self.predict_anomaly_score(row)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt)

        return {
            "mean_latency_ms": float(np.mean(latencies_ms)),
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        }
