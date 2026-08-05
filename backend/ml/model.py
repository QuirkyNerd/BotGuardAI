from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from loguru import logger

from backend.config import CALIBRATED_MODEL_PATH, MODEL_PATH
from backend.ml.calibration import CalibratedModelWrapper
from backend.services.feature_engineering import FEATURE_NAMES
from backend.services.metrics import MODEL_INFERENCE_LATENCY

_model_wrapper: Optional[CalibratedModelWrapper] = None
_model_lock = threading.Lock()


def load_model(model_path_str: Optional[str] = None) -> None:
    """
    Load the ML model (calibrated wrapper or base model) into a global thread-safe cache.
    Prefers calibrated model artifact if available, falling back safely to base model artifact.
    """
    global _model_wrapper
    with _model_lock:
        if _model_wrapper is not None:
            return

        target_path = Path(model_path_str) if model_path_str else MODEL_PATH

        # Safe fallback logic: Check calibrated model artifact first
        if CALIBRATED_MODEL_PATH.exists():
            logger.info("Loading calibrated ML model artifact from {}", CALIBRATED_MODEL_PATH)
            loaded_obj = joblib.load(CALIBRATED_MODEL_PATH)
        elif target_path.exists():
            logger.info("Calibrated model not found; loading base ML model artifact from {}", target_path)
            loaded_obj = joblib.load(target_path)
        else:
            raise FileNotFoundError(f"No ML model artifact found at {CALIBRATED_MODEL_PATH} or {target_path}")

        # Wrap in CalibratedModelWrapper to guarantee safe human class indexing
        if isinstance(loaded_obj, CalibratedModelWrapper):
            _model_wrapper = loaded_obj
        else:
            _model_wrapper = CalibratedModelWrapper(loaded_obj, calibration_method="loaded")

        logger.info("ML model initialized successfully with expected features: {}", FEATURE_NAMES)


def _ensure_model_loaded() -> CalibratedModelWrapper:
    if _model_wrapper is None:
        # Auto-attempt load if not yet loaded
        load_model()
    if _model_wrapper is None:
        raise RuntimeError("ML model is not loaded. Call load_model() at startup.")
    return _model_wrapper


def predict_human_probability(features: List[float]) -> float:
    """
    Run inference and return the calibrated probability P(Y=Human) for the session.

    :param features: feature vector ordered by FEATURE_NAMES
    """
    wrapper = _ensure_model_loaded()
    with MODEL_INFERENCE_LATENCY.time():
        prob = wrapper.predict_human_probability(features)
    logger.debug("Predicted calibrated human probability: {:.4f}", prob)
    return prob
