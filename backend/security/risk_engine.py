from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from backend.config import (
    PROVISIONAL_ANOMALY_THRESHOLD,
    WEIGHT_ANOMALY,
    WEIGHT_RF,
    WEIGHT_TEMPORAL,
)
from backend.ml.intelligence_engine import MultiEnginePrediction
from backend.models.schemas import BrowserMetadata


@dataclass
class RiskEvaluationResult:
    """
    Structured output of Risk Engine 2.0 evaluation.
    """

    composite_risk_score: float
    human_probability: float
    anomaly_score: float
    temporal_human_probability: float
    risk_components: Dict[str, float]
    triggered_indicators: List[str] = field(default_factory=list)
    model_availability: Dict[str, bool] = field(default_factory=dict)


def _score_browser_fingerprint(metadata: Optional[BrowserMetadata]) -> Tuple[float, List[str]]:
    indicators: List[str] = []
    if not metadata:
        return 5.0, ["missing_browser_metadata"]
    score = 0.0
    if metadata.webdriver:
        score += 20.0
        indicators.append("webdriver_detected")
    if metadata.touch_points is not None and metadata.touch_points == 0:
        score += 5.0
        indicators.append("zero_touch_points")
    entropy = metadata.device_entropy or 0.0
    if entropy == 0:
        score += 5.0
        indicators.append("zero_device_entropy")
    elif entropy < 1e4:
        score += 3.0
        indicators.append("low_device_entropy")
    return score, indicators


def _score_security_flags(flags: Dict[str, Any]) -> Tuple[float, List[str]]:
    indicators: List[str] = []
    score = 0.0
    if flags.get("suspicious"):
        score += 15.0
        indicators.append("suspicious_security_context")
    recent_count = int(flags.get("recent_request_count", 0))
    if recent_count > 60:
        score += 10.0
        indicators.append("high_request_rate")
    if recent_count > 120:
        score += 10.0
        indicators.append("extreme_request_rate")
    return score, indicators


def _score_interaction_anomaly(features: List[float]) -> Tuple[float, List[str]]:
    indicators: List[str] = []
    if len(features) < 9:
        return 0.0, []
    interaction_density = features[7]
    click_interval_std = features[3]
    avg_idle_duration = features[8]

    score = 0.0
    if interaction_density > 10.0:
        score += 10.0
        indicators.append("high_interaction_density")
    if click_interval_std < 0.05:
        score += 8.0
        indicators.append("constant_click_timing")
    if avg_idle_duration < 0.5:
        score += 5.0
        indicators.append("zero_idle_duration")
    return score, indicators


def compute_risk_score_v2(
    pred: MultiEnginePrediction,
    features: List[float],
    browser_metadata: Optional[BrowserMetadata] = None,
    security_flags: Optional[Dict[str, Any]] = None,
) -> RiskEvaluationResult:
    """
    Risk Engine 2.0 Entrypoint.
    Combines 3 ML intelligence signals + non-ML security indicators + high-confidence escalation overrides.
    """
    triggered_indicators: List[str] = []

    # 1. Multi-Engine ML Risk Fusion
    weights: List[float] = []
    risks: List[float] = []

    if pred.rf_available:
        weights.append(WEIGHT_RF)
        risks.append(pred.rf_risk)
    if pred.anomaly_available:
        weights.append(WEIGHT_ANOMALY)
        risks.append(pred.anomaly_risk)
    if pred.temporal_available:
        weights.append(WEIGHT_TEMPORAL)
        risks.append(pred.temporal_risk)

    if len(weights) > 0:
        total_w = sum(weights)
        ml_composite_risk = sum(w * r for w, r in zip(weights, risks)) / total_w
    else:
        # Default safety fallback if all models unavailable
        ml_composite_risk = 50.0
        triggered_indicators.append("all_ml_models_unavailable")

    # 2. Non-ML Security Indicators
    fp_score, fp_ind = _score_browser_fingerprint(browser_metadata)
    sec_score, sec_ind = _score_security_flags(security_flags or {})
    interaction_score, int_ind = _score_interaction_anomaly(features)

    triggered_indicators.extend(fp_ind + sec_ind + int_ind)

    # Base combined risk
    non_ml_risk = min(30.0, fp_score + sec_score + interaction_score)
    raw_composite = 0.70 * ml_composite_risk + 0.30 * (non_ml_risk / 30.0 * 100.0)

    # 3. High-Confidence Security Escalation Overrides
    final_risk = raw_composite

    # Override A: Webdriver automation flag
    if browser_metadata and browser_metadata.webdriver:
        final_risk = max(final_risk, 85.0)
        triggered_indicators.append("escalation_override_webdriver")

    # Override B: Multi-engine confirmation (Anomaly Forest + Temporal CNN both confirm anomaly)
    if pred.anomaly_available and pred.temporal_available:
        if pred.is_anomaly and pred.temporal_human_probability < 0.50:
            final_risk = max(final_risk, 75.0)
            triggered_indicators.append("escalation_override_multi_engine_anomaly")

    # Clamp composite risk to [0, 100]
    composite_risk_score = round(float(max(0.0, min(100.0, final_risk))), 2)

    components = {
        "behavioral_ml": round(pred.rf_risk, 2),
        "anomaly": round(pred.anomaly_risk, 2),
        "temporal": round(pred.temporal_risk, 2),
        "browser_fingerprint": round(fp_score, 2),
        "security_flags": round(sec_score, 2),
        "interaction_anomaly": round(interaction_score, 2),
    }

    availability = {
        "rf_available": pred.rf_available,
        "anomaly_available": pred.anomaly_available,
        "temporal_available": pred.temporal_available,
    }

    logger.debug(
        "Risk Engine 2.0 evaluation: ML={:.1f}, NonML={:.1f} -> Composite={:.1f} (Indicators: {})",
        ml_composite_risk,
        non_ml_risk,
        composite_risk_score,
        len(triggered_indicators),
    )

    return RiskEvaluationResult(
        composite_risk_score=composite_risk_score,
        human_probability=round(pred.human_probability, 4),
        anomaly_score=round(pred.anomaly_score, 2),
        temporal_human_probability=round(pred.temporal_human_probability, 4),
        risk_components=components,
        triggered_indicators=list(set(triggered_indicators)),
        model_availability=availability,
    )
