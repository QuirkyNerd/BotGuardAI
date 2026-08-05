from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import torch
from loguru import logger

from backend.config import (
    ANOMALY_DETECTOR_PATH,
    CALIBRATED_MODEL_PATH,
    MODEL_PATH,
    PROVISIONAL_ANOMALY_THRESHOLD,
    TEMPORAL_MODEL_PATH,
)
from backend.ml.anomaly_detector import BehavioralAnomalyDetector
from backend.ml.calibration import CalibratedModelWrapper
from backend.ml.temporal_model import Temporal1DCNN, extract_raw_event_sequence
from backend.models.schemas import BehaviorBatch


@dataclass
class MultiEnginePrediction:
    """
    Unified multi-engine inference predictions and availability status.
    """

    # Calibrated Random Forest (Aggregate Features)
    rf_available: bool = False
    human_probability: float = 0.5
    rf_risk: float = 50.0

    # Behavioral Anomaly Detector (Isolation Forest)
    anomaly_available: bool = False
    anomaly_score: float = 0.0
    anomaly_risk: float = 0.0
    is_anomaly: bool = False

    # Temporal 1D CNN (Raw Event Sequences)
    temporal_available: bool = False
    temporal_human_probability: float = 0.5
    temporal_risk: float = 50.0

    # Latency timing breakdown
    rf_latency_ms: float = 0.0
    anomaly_latency_ms: float = 0.0
    temporal_latency_ms: float = 0.0


class MultiEngineOrchestrator:
    """
    Singleton orchestrator for loading and running inference across all 3 intelligence engines.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rf_wrapper: Optional[CalibratedModelWrapper] = None
        self._anomaly_detector: Optional[BehavioralAnomalyDetector] = None
        self._temporal_cnn: Optional[Temporal1DCNN] = None
        self._initialized: bool = False

    def load_all_models(
        self,
        rf_path: Optional[Path] = None,
        anomaly_path: Optional[Path] = None,
        temporal_path: Optional[Path] = None,
    ) -> None:
        """
        Load all intelligence artifacts into memory once during application startup.
        Fault tolerant: Failures in individual optional models do not crash application startup.
        """
        with self._lock:
            if self._initialized:
                return

            logger.info("Initializing BotGuard AI Multi-Engine Intelligence Stack...")

            # 1. Load Calibrated Random Forest
            target_rf = rf_path or CALIBRATED_MODEL_PATH
            if not target_rf.exists():
                target_rf = MODEL_PATH

            if target_rf.exists():
                try:
                    logger.info("Loading Random Forest model artifact from {}", target_rf)
                    obj = joblib.load(target_rf)
                    if isinstance(obj, CalibratedModelWrapper):
                        self._rf_wrapper = obj
                    else:
                        self._rf_wrapper = CalibratedModelWrapper(obj, calibration_method="loaded")
                    logger.info("Random Forest model initialized successfully.")
                except Exception as exc:
                    logger.warning("Failed to load Random Forest model artifact: {}", exc)
            else:
                logger.warning("Random Forest model artifact not found at {}", target_rf)

            # 2. Load Isolation Forest Anomaly Detector
            target_anom = anomaly_path or ANOMALY_DETECTOR_PATH
            if target_anom.exists():
                try:
                    logger.info("Loading Isolation Forest anomaly detector from {}", target_anom)
                    self._anomaly_detector = joblib.load(target_anom)
                    logger.info("Isolation Forest anomaly detector initialized successfully.")
                except Exception as exc:
                    logger.warning("Failed to load Anomaly Detector artifact: {}", exc)
            else:
                logger.warning("Anomaly Detector artifact not found at {}", target_anom)

            # 3. Load PyTorch Temporal 1D CNN
            target_temp = temporal_path or TEMPORAL_MODEL_PATH
            if target_temp.exists():
                try:
                    logger.info("Loading Temporal 1D CNN model weights from {}", target_temp)
                    cnn = Temporal1DCNN(in_channels=7, seq_len=60)
                    cnn.load_state_dict(torch.load(target_temp, map_location=torch.device("cpu")))
                    cnn.eval()
                    self._temporal_cnn = cnn
                    logger.info("Temporal 1D CNN model initialized successfully.")
                except Exception as exc:
                    logger.warning("Failed to load Temporal 1D CNN model: {}", exc)
            else:
                logger.warning("Temporal 1D CNN model artifact not found at {}", target_temp)

            self._initialized = True
            logger.info("Multi-Engine Intelligence Stack initialization complete.")

    @property
    def rf_model(self) -> Optional[CalibratedModelWrapper]:
        """Expose the Calibrated Random Forest wrapper for read-only access."""
        return self._rf_wrapper

    def predict_all(
        self,
        features: List[float],
        batch: Optional[BehaviorBatch] = None,
    ) -> MultiEnginePrediction:
        """
        Execute multi-engine inference across all available intelligence models.
        """
        if not self._initialized:
            self.load_all_models()

        res = MultiEnginePrediction()

        # 1. Calibrated Random Forest Inference
        if self._rf_wrapper is not None:
            try:
                t0 = time.perf_counter()
                prob = self._rf_wrapper.predict_human_probability(features)
                res.rf_latency_ms = (time.perf_counter() - t0) * 1000.0
                res.rf_available = True
                res.human_probability = float(prob)
                res.rf_risk = (1.0 - float(prob)) * 100.0
            except Exception as exc:
                logger.warning("RF inference failed: {}", exc)

        # 2. Isolation Forest Anomaly Inference
        if self._anomaly_detector is not None:
            try:
                t0 = time.perf_counter()
                anom_score = self._anomaly_detector.predict_anomaly_score(features)
                res.anomaly_latency_ms = (time.perf_counter() - t0) * 1000.0
                res.anomaly_available = True
                res.anomaly_score = float(anom_score)
                res.anomaly_risk = float(anom_score)  # 0-100 normalized risk scale
                res.is_anomaly = float(anom_score) >= PROVISIONAL_ANOMALY_THRESHOLD
            except Exception as exc:
                logger.warning("Anomaly inference failed: {}", exc)

        # 3. Temporal 1D CNN Inference
        if self._temporal_cnn is not None and batch is not None:
            try:
                t0 = time.perf_counter()
                seq = extract_raw_event_sequence(batch, max_len=60)
                # Reshape to (1, 7, 60) for PyTorch Conv1d
                tensor_in = torch.from_numpy(seq).unsqueeze(0).transpose(1, 2)
                with torch.no_grad():
                    temp_prob = self._temporal_cnn(tensor_in).item()
                res.temporal_latency_ms = (time.perf_counter() - t0) * 1000.0
                res.temporal_available = True
                res.temporal_human_probability = float(temp_prob)
                res.temporal_risk = (1.0 - float(temp_prob)) * 100.0
            except Exception as exc:
                logger.warning("Temporal 1D CNN inference failed: {}", exc)

        return res


# Global singleton instance
_orchestrator = MultiEngineOrchestrator()


def load_intelligence_stack() -> None:
    """Entrypoint for FastAPI startup lifespan to load all ML artifacts into memory."""
    _orchestrator.load_all_models()


def get_orchestrator() -> MultiEngineOrchestrator:
    """Return the global orchestrator instance (for read-only access e.g. explainability)."""
    return _orchestrator


def run_multi_engine_prediction(
    features: List[float],
    batch: Optional[BehaviorBatch] = None,
) -> MultiEnginePrediction:
    """Run multi-engine prediction using the global orchestrator."""
    return _orchestrator.predict_all(features, batch)

