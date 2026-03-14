from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import List, Optional

import joblib
import numpy as np
from loguru import logger

from backend.services.feature_engineering import FEATURE_NAMES
from backend.services.metrics import MODEL_INFERENCE_LATENCY

_model = None
_model_lock = threading.Lock()


def load_model(model_path: str) -> None:
    """
    Load the RandomForest model from disk into a global cache.
    This is safe to call multiple times; subsequent calls are no-ops.
    """
    global _model
    with _model_lock:
        if _model is not None:
            return
        logger.info("Loading ML model from {}", model_path)
        _model = joblib.load(model_path)
        logger.info("Model loaded with expected features: {}", FEATURE_NAMES)


def _ensure_model_loaded() -> object:
    if _model is None:
        raise RuntimeError("ML model is not loaded. Call load_model() at startup.")
    return _model


def predict_human_probability(features: List[float]) -> float:
    """
    Run inference and return the probability that the session is human.

    :param features: feature vector ordered by FEATURE_NAMES
    """
    model = _ensure_model_loaded()
    input_array = np.array(features, dtype=float).reshape(1, -1)
    with MODEL_INFERENCE_LATENCY.time():
        proba = model.predict_proba(input_array)[0]
    # Assume class ordering [bot, human] in training script.
    human_probability = float(proba[1])
    logger.debug("Model probabilities (bot, human): {}", proba)
    return human_probability
