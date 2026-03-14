from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from backend.ml.model import predict_human_probability
from backend.models.schemas import BrowserMetadata, RiskLevel, VerifyResponse
from backend.security.risk_engine import RiskContext, compute_risk_score


@dataclass
class DecisionThresholds:
    allow_threshold: float = 0.85
    challenge_threshold: float = 0.60


THRESHOLDS = DecisionThresholds()


def _classify_risk(human_probability: float) -> RiskLevel:
    if human_probability >= THRESHOLDS.allow_threshold:
        return RiskLevel.LOW
    if human_probability >= THRESHOLDS.challenge_threshold:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


def _recommended_action_for_risk(risk: RiskLevel) -> str:
    if risk is RiskLevel.LOW:
        return "allow"
    if risk is RiskLevel.MEDIUM:
        return "challenge"
    return "block"


def evaluate_session(
    session_id: str,
    features: List[float],
    browser_metadata: Optional[BrowserMetadata] = None,
    security_flags: Optional[Dict[str, object]] = None,
) -> VerifyResponse:
    """
    Core decision engine entrypoint. Takes a feature vector, invokes the ML model,
    and maps the resulting probability to a risk level and recommended action.
    """
    human_probability = predict_human_probability(features)
    risk_level = _classify_risk(human_probability)
    ctx = RiskContext(
        human_probability=human_probability,
        features=features,
        browser_metadata=browser_metadata,
        security_flags=security_flags or {},
    )
    risk_score = compute_risk_score(ctx)
    recommended_action = _recommended_action_for_risk(risk_level)

    logger.info(
        "Session {} evaluated: human_probability={:.3f}, risk_level={}, action={}, risk_score={:.1f}",
        session_id,
        human_probability,
        risk_level.value,
        recommended_action,
        risk_score,
    )

    return VerifyResponse(
        session_id=session_id,
        human_probability=human_probability,
        risk_level=risk_level,
        recommended_action=recommended_action,
        risk_score=risk_score,
    )
