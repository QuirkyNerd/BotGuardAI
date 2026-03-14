from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from backend.models.schemas import BrowserMetadata


@dataclass
class RiskContext:
    """
    Aggregated context for risk scoring.
    """

    human_probability: float
    features: List[float]
    browser_metadata: Optional[BrowserMetadata]
    security_flags: Dict[str, Any]


def _score_ml_component(human_probability: float) -> float:
    # Inverse of human probability mapped into [0, 70].
    return (1.0 - human_probability) * 70.0


def _score_browser_fingerprint(metadata: Optional[BrowserMetadata]) -> float:
    if not metadata:
        return 5.0
    score = 0.0
    if metadata.webdriver:
        score += 20.0
    if metadata.touch_points is not None and metadata.touch_points == 0:
        score += 5.0
    entropy = metadata.device_entropy or 0.0
    if entropy == 0:
        score += 5.0
    elif entropy < 1e4:
        score += 3.0
    return score


def _score_security_flags(flags: Dict[str, Any]) -> float:
    score = 0.0
    if flags.get("suspicious"):
        score += 15.0
    recent_count = int(flags.get("recent_request_count", 0))
    if recent_count > 60:
        score += 10.0
    if recent_count > 120:
        score += 10.0
    return score


def _score_interaction_anomaly(features: List[float]) -> float:
    # Features are ordered per FEATURE_NAMES.
    if len(features) < 9:
        return 0.0
    interaction_density = features[7]
    click_interval_std = features[3]
    avg_idle_duration = features[8]

    score = 0.0
    if interaction_density > 10.0:
        score += 10.0
    if click_interval_std < 0.05:
        score += 8.0
    if avg_idle_duration < 0.5:
        score += 5.0
    return score


def compute_risk_score(ctx: RiskContext) -> float:
    """
    Compute a multi-factor risk score between 0 and 100.
    """
    ml_score = _score_ml_component(ctx.human_probability)
    fp_score = _score_browser_fingerprint(ctx.browser_metadata)
    sec_score = _score_security_flags(ctx.security_flags)
    interaction_score = _score_interaction_anomaly(ctx.features)

    raw = ml_score + fp_score + sec_score + interaction_score
    risk_score = max(0.0, min(100.0, raw))

    logger.debug(
        "Risk scoring components ml={}, fp={}, sec={}, interaction={} -> {}",
        ml_score,
        fp_score,
        sec_score,
        interaction_score,
        risk_score,
    )
    return risk_score
