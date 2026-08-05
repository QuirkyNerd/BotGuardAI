from __future__ import annotations

import os
from pathlib import Path

# Base Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default Artifact Paths
DEFAULT_MODEL_PATH = PROJECT_ROOT / "backend" / "ml" / "artifacts" / "human_bot_model.pkl"
DEFAULT_CALIBRATED_MODEL_PATH = PROJECT_ROOT / "backend" / "ml" / "artifacts" / "human_bot_model_calibrated.pkl"
DEFAULT_ANOMALY_DETECTOR_PATH = PROJECT_ROOT / "backend" / "ml" / "artifacts" / "anomaly_detector.pkl"
DEFAULT_TEMPORAL_MODEL_PATH = PROJECT_ROOT / "backend" / "ml" / "artifacts" / "temporal_model.pt"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "backend" / "ml" / "artifacts" / "model_registry.json"

# Configurable Paths via Environment Variables
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
CALIBRATED_MODEL_PATH = Path(os.getenv("CALIBRATED_MODEL_PATH", str(DEFAULT_CALIBRATED_MODEL_PATH)))
ANOMALY_DETECTOR_PATH = Path(os.getenv("ANOMALY_DETECTOR_PATH", str(DEFAULT_ANOMALY_DETECTOR_PATH)))
TEMPORAL_MODEL_PATH = Path(os.getenv("TEMPORAL_MODEL_PATH", str(DEFAULT_TEMPORAL_MODEL_PATH)))
MODEL_REGISTRY_PATH = Path(os.getenv("MODEL_REGISTRY_PATH", str(DEFAULT_REGISTRY_PATH)))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# Decision Engine Risk Thresholds (Composite Risk Score [0, 100])
# PROVISIONAL / SIMULATION-DERIVED: Evaluated on benchmark suite
COMPOSITE_ALLOW_RISK_THRESHOLD = float(os.getenv("COMPOSITE_ALLOW_RISK_THRESHOLD", "35.0"))
COMPOSITE_CHALLENGE_RISK_THRESHOLD = float(os.getenv("COMPOSITE_CHALLENGE_RISK_THRESHOLD", "65.0"))

# Legacy Probability Thresholds (Kept for backward compatibility)
ALLOW_THRESHOLD = float(os.getenv("ALLOW_THRESHOLD", "0.85"))
CHALLENGE_THRESHOLD = float(os.getenv("CHALLENGE_THRESHOLD", "0.60"))

# Multi-Engine Risk Fusion Weights (Must sum to 1.0)
WEIGHT_RF = float(os.getenv("WEIGHT_RF", "0.35"))
WEIGHT_ANOMALY = float(os.getenv("WEIGHT_ANOMALY", "0.30"))
WEIGHT_TEMPORAL = float(os.getenv("WEIGHT_TEMPORAL", "0.35"))

# Anomaly Threshold (PROVISIONAL: Derived from Human Validation P97)
PROVISIONAL_ANOMALY_THRESHOLD = float(os.getenv("PROVISIONAL_ANOMALY_THRESHOLD", "58.62"))
