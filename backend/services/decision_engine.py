from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from backend.config import (
    COMPOSITE_ALLOW_RISK_THRESHOLD,
    COMPOSITE_CHALLENGE_RISK_THRESHOLD,
)
from backend.ml.intelligence_engine import run_multi_engine_prediction
from backend.models.schemas import BehaviorBatch, BrowserMetadata, RiskLevel, VerifyResponse
from backend.security.risk_engine import compute_risk_score_v2


@dataclass
class RiskThresholds:
    allow_risk_max: float = COMPOSITE_ALLOW_RISK_THRESHOLD  # Risk < 35.0 => ALLOW
    challenge_risk_max: float = COMPOSITE_CHALLENGE_RISK_THRESHOLD  # Risk < 65.0 => CHALLENGE (else BLOCK)


THRESHOLDS = RiskThresholds()


def _classify_risk_v2(composite_risk: float) -> RiskLevel:
    if composite_risk < THRESHOLDS.allow_risk_max:
        return RiskLevel.LOW
    if composite_risk < THRESHOLDS.challenge_risk_max:
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
    batch: Optional[BehaviorBatch] = None,
) -> VerifyResponse:
    """
    Core Decision Engine 2.0 entrypoint.
    Executes multi-engine inference (RF, Anomaly Detector, Temporal 1D CNN),
    evaluates Risk Engine 2.0 multi-factor fusion, and maps composite risk score
    to action (ALLOW, CHALLENGE, BLOCK).
    """
    # 1. Run Multi-Engine Intelligence Stack Inference
    pred = run_multi_engine_prediction(features, batch=batch)

    # 2. Risk Engine 2.0 Multi-Factor Fusion
    risk_result = compute_risk_score_v2(
        pred=pred,
        features=features,
        browser_metadata=browser_metadata,
        security_flags=security_flags or {},
    )

    # 3. Classify Risk Level & Decision
    risk_level = _classify_risk_v2(risk_result.composite_risk_score)
    recommended_action = _recommended_action_for_risk(risk_level)

    logger.info(
        "Session {} evaluated (Risk Engine 2.0): composite_risk={:.1f}, action={}, rf_prob={:.3f}, anom_score={:.1f}, temp_prob={:.3f}",
        session_id,
        risk_result.composite_risk_score,
        recommended_action,
        risk_result.human_probability,
        risk_result.anomaly_score,
        risk_result.temporal_human_probability,
    )

    return VerifyResponse(
        session_id=session_id,
        human_probability=risk_result.human_probability,
        risk_level=risk_level,
        recommended_action=recommended_action,
        risk_score=risk_result.composite_risk_score,
        anomaly_score=risk_result.anomaly_score,
        temporal_human_probability=risk_result.temporal_human_probability,
        risk_components=risk_result.risk_components,
        triggered_indicators=risk_result.triggered_indicators,
    )
