from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from sqlalchemy.orm import Session

from backend.challenge_engine.service import start_challenge, verify_challenge
from backend.database.session import get_db

from backend.models.schemas import (
    AnalyticsResponse,
    BehaviorBatch,
    ChallengeStartRequest,
    ChallengeStartResponse,
    ChallengeVerifyRequest,
    ChallengeVerifyResponse,
    VerifyRequest,
    VerifyResponse,
)

from backend.services.decision_engine import evaluate_session
from backend.services.feature_engineering import compute_features_from_batches
from backend.services.feature_store import persist_feature_vector
from backend.services.logging_service import log_evaluation_result, read_analytics
from backend.services.metrics import VERIFICATION_LATENCY, VERIFICATION_OUTCOME
from backend.database.repository import (
    SecurityEventRepository,
    SessionRepository,
    TelemetryRepository,
    VerificationRepository,
)

router = APIRouter(prefix="", tags=["verification"])

@router.post("/collect-behavior", status_code=200)
async def collect_behavior(
    batch: BehaviorBatch,
    request: Request,
    db: Session = Depends(get_db),
) -> None:
    """
    Ingest a batch of behavioral telemetry events for a session.
    This endpoint is intentionally lightweight and write-only.
    """
    try:
        SessionRepository.get_or_create(
            db=db,
            session_id=batch.session_id,
            user_agent=batch.metadata.user_agent if batch.metadata else None,
            ip_address=request.client.host if request.client else None,
            browser_metadata=batch.metadata.model_dump() if batch.metadata else None,
        )
        TelemetryRepository.save_batch(db, batch)
        db.commit()
    except Exception as exc:  # pragma: no cover - defensive
        db.rollback()
        logger.exception("Failed to collect behavior batch: {}", exc)
        raise HTTPException(status_code=500, detail="Failed to collect behavior") from exc


@router.post("/verify-session", response_model=VerifyResponse)
async def verify_session(
    request: VerifyRequest,
    http_request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """
    Compute a human confidence score for the given session.
    This uses accumulated behavioral telemetry plus optional latest batch.
    """
    try:
        with VERIFICATION_LATENCY.time():
            # Optional: persist latest behavior batch.
            if request.latest_batch is not None:
                await collect_behavior(
                    batch=request.latest_batch,
                    request=http_request,
                    db=db,
                )

            rows = TelemetryRepository.get_session_batches(db, request.session_id)
            if not rows:
                raise ValueError(f"No telemetry found for session_id={request.session_id}")

            batches = [BehaviorBatch.model_validate(row.payload) for row in rows]
            feature_vector = compute_features_from_batches(request.session_id, batches)

            model_version = os.getenv("MODEL_VERSION", "v1")
            persist_feature_vector(
                db=db,
                session_id=request.session_id,
                model_version=model_version,
                features=feature_vector,
            )

            # Derive browser metadata from the latest batch if present.
            browser_md = batches[-1].metadata if batches and batches[-1].metadata else None
            sec_flags = getattr(http_request.state, "security_context", {}) or {}

            evaluation = evaluate_session(
                session_id=request.session_id,
                features=feature_vector.values,
                browser_metadata=browser_md,
                security_flags=sec_flags,
                batch=batches[-1] if batches else None,
            )

            # Persist verification result via Repository
            VerificationRepository.save_verification_result(
                db=db,
                evaluation=evaluation,
                model_version=model_version,
            )

            # Record triggered security events if present
            if evaluation.triggered_indicators:
                SecurityEventRepository.record_events(
                    db=db,
                    session_id=request.session_id,
                    indicators=evaluation.triggered_indicators,
                    composite_risk=evaluation.risk_score,
                )

            # Update session summary
            SessionRepository.update_summary(
                db=db,
                session_id=request.session_id,
                last_human_probability=evaluation.human_probability,
                last_risk_level=evaluation.risk_level.value,
            )

            db.commit()

            VERIFICATION_OUTCOME.labels(risk_level=evaluation.risk_level.value).inc()
            log_evaluation_result(evaluation)
            return evaluation
    except ValueError as exc:
        db.rollback()
        logger.warning("Verification failed due to bad input: {}", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        db.rollback()
        logger.exception("Verification failed unexpectedly: {}", exc)
        raise HTTPException(status_code=500, detail="Verification failed") from exc



@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(db: Session = Depends(get_db)) -> AnalyticsResponse:
    """
    Return aggregate statistics for dashboard visualization.
    """
    try:
        return read_analytics(db)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to read analytics: {}", exc)
        raise HTTPException(status_code=500, detail="Failed to read analytics") from exc


@router.post("/challenge/start", response_model=ChallengeStartResponse, tags=["challenge"])
async def challenge_start(
    req: ChallengeStartRequest,
    db: Session = Depends(get_db),
) -> ChallengeStartResponse:
    try:
        return start_challenge(db, req)
    except Exception as exc:
        logger.exception("Failed to start challenge: {}", exc)
        raise HTTPException(status_code=500, detail="Failed to start challenge") from exc


@router.post("/challenge/verify", response_model=ChallengeVerifyResponse, tags=["challenge"])
async def challenge_verify(
    req: ChallengeVerifyRequest,
    db: Session = Depends(get_db),
) -> ChallengeVerifyResponse:
    try:
        return verify_challenge(db, req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to verify challenge: {}", exc)
        raise HTTPException(status_code=500, detail="Failed to verify challenge") from exc


@router.post("/simulate-bot", tags=["simulation"])
async def simulate_bot_endpoint(
    session_id: str,
    bot_type: str = "headless",
    db: Session = Depends(get_db),
) -> BehaviorBatch:
    """
    Generate simulated bot telemetry and optionally persist it.
    Used for integration testing and benchmark data generation.
    """
    from backend.simulation.bot_simulator import simulate_bot as _simulate_bot
    batch = _simulate_bot(session_id=session_id, bot_type=bot_type)
    # Persist for analytics and inspection.
    await collect_behavior(batch=batch, request=Request(scope={"type": "http"}), db=db)  # type: ignore[arg-type]
    return batch



@router.get("/session/{session_id}/heatmap", tags=["analytics"])
async def session_heatmap(
    session_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Build a coarse mouse-movement heatmap grid for a session.
    """
    rows = TelemetryRepository.get_session_batches(db, session_id)

    if not rows:
        raise HTTPException(status_code=404, detail="Session not found")

    batches = [BehaviorBatch.model_validate(row.payload) for row in rows]
    width, height = 20, 12
    grid = [[0 for _ in range(width)] for _ in range(height)]

    for batch in batches:
        for mv in batch.mouse_moves:
            if not batch.metadata or not batch.metadata.screen_width or not batch.metadata.screen_height:
                continue
            x_norm = mv.position.x / batch.metadata.screen_width
            y_norm = mv.position.y / batch.metadata.screen_height
            gx = max(0, min(width - 1, int(x_norm * width)))
            gy = max(0, min(height - 1, int(y_norm * height)))
            grid[gy][gx] += 1

    max_count = max(max(row) for row in grid) if grid else 0
    return {"width": width, "height": height, "grid": grid, "maxCount": max_count}


@router.get("/explain/{session_id}", tags=["explainability"])
async def explain_session(
    session_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Return top feature importances (Random Forest impurity-based) for the most recent
    verified session. These reflect overall model-level feature weights rather than
    per-session SHAP contributions.
    """
    from backend.ml.intelligence_engine import get_orchestrator
    from backend.services.feature_engineering import FEATURE_NAMES

    rows = TelemetryRepository.get_session_batches(db, session_id)
    if not rows:
        raise HTTPException(status_code=404, detail="No telemetry found for session")

    batches = [BehaviorBatch.model_validate(row.payload) for row in rows]
    feature_vector = compute_features_from_batches(session_id, batches)

    orchestrator = get_orchestrator()
    rf_wrapper = orchestrator.rf_model
    if rf_wrapper is None:
        raise HTTPException(status_code=503, detail="Intelligence stack not loaded")

    # CalibratedModelWrapper.model is the underlying sklearn estimator
    base_model = rf_wrapper.model
    if hasattr(base_model, "feature_importances_"):
        importances = base_model.feature_importances_
        pairs = list(zip(FEATURE_NAMES, importances))
        pairs.sort(key=lambda p: p[1], reverse=True)
        top = [{"feature": name, "importance": float(imp)} for name, imp in pairs[:5]]
        return {"session_id": session_id, "type": "rf_feature_importance", "top_features": top}
    else:
        # Fallback for CalibratedClassifierCV (calibrated wrapper does not expose importances)
        pairs = list(zip(FEATURE_NAMES, feature_vector.values))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        top = [{"feature": name, "value": float(val)} for name, val in pairs[:5]]
        return {"session_id": session_id, "type": "feature_values", "top_features": top}




@router.post("/protected-resource", tags=["demo"])
async def protected_resource(
    session_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Example of API protection: access is denied when risk_score exceeds a threshold.
    """
    vr = VerificationRepository.get_latest_verification_result(db, session_id)

    if vr is None:
        raise HTTPException(status_code=403, detail="No verification for session")

    risk_score = vr.risk_score if vr.risk_score is not None else (1.0 - vr.human_probability) * 100.0
    threshold = 70.0
    if risk_score > threshold:
        raise HTTPException(status_code=403, detail="Access blocked by BotGuard AI policy")

    return {"status": "ok", "message": "Protected resource accessed", "risk_score": risk_score}
