from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models.db_models import (
    ChallengeEntity,
    FeatureVectorEntity,
    SecurityEventEntity,
    SessionEntity,
    TelemetryBatchEntity,
    VerificationResultEntity,
)
from backend.models.schemas import AnalyticsBucket, AnalyticsResponse, BehaviorBatch, VerifyResponse


def sanitize_telemetry_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce backend privacy sanitization.
    Guarantees that key_presses in the payload NEVER contain typed text, key characters, or form contents.
    Only timing, order, and event metadata are preserved.
    """
    sanitized = dict(payload)
    if "key_presses" in sanitized and isinstance(sanitized["key_presses"], list):
        clean_keys = []
        for item in sanitized["key_presses"]:
            if isinstance(item, dict):
                clean_item = {
                    k: v
                    for k, v in item.items()
                    if k not in ("key", "char", "character", "value", "text", "password", "input")
                }
                clean_keys.append(clean_item)
            else:
                clean_keys.append(item)
        sanitized["key_presses"] = clean_keys
    return sanitized


class SessionRepository:
    """Session entity data access functions."""

    @staticmethod
    def get_or_create(
        db: Session,
        session_id: str,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
        browser_metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionEntity:
        session = (
            db.query(SessionEntity)
            .filter(SessionEntity.session_id == session_id)
            .one_or_none()
        )
        if session is None:
            session = SessionEntity(
                session_id=session_id,
                user_agent=user_agent,
                ip_address=ip_address,
                browser_metadata=browser_metadata,
            )
            db.add(session)
            db.flush()
        else:
            session.last_seen_at = datetime.now(timezone.utc)
            if browser_metadata and not session.browser_metadata:
                session.browser_metadata = browser_metadata
            db.flush()
        return session

    @staticmethod
    def update_summary(
        db: Session,
        session_id: str,
        last_human_probability: float,
        last_risk_level: str,
    ) -> Optional[SessionEntity]:
        session = (
            db.query(SessionEntity)
            .filter(SessionEntity.session_id == session_id)
            .one_or_none()
        )
        if session is not None:
            session.last_human_probability = last_human_probability
            session.last_risk_level = last_risk_level
            db.flush()
        return session


class TelemetryRepository:
    """Telemetry batch data access functions."""

    @staticmethod
    def save_batch(
        db: Session,
        batch: BehaviorBatch,
    ) -> TelemetryBatchEntity:
        event_counts = {
            "mouse_moves": len(batch.mouse_moves),
            "scrolls": len(batch.scrolls),
            "clicks": len(batch.clicks),
            "key_presses": len(batch.key_presses),
            "focus_events": len(batch.focus_events),
        }

        raw_payload = batch.model_dump(mode="json")
        sanitized_payload = sanitize_telemetry_payload(raw_payload)

        entity = TelemetryBatchEntity(
            session_id=batch.session_id,
            started_at_ms=batch.started_at,
            ended_at_ms=batch.ended_at,
            event_counts=event_counts,
            payload=sanitized_payload,
        )
        db.add(entity)
        db.flush()
        return entity

    @staticmethod
    def get_session_batches(db: Session, session_id: str) -> List[TelemetryBatchEntity]:
        return (
            db.query(TelemetryBatchEntity)
            .filter(TelemetryBatchEntity.session_id == session_id)
            .order_by(TelemetryBatchEntity.started_at_ms.asc())
            .all()
        )

    @staticmethod
    def cleanup_old_telemetry_batches(db: Session, max_age_hours: int = 24) -> int:
        """
        Retention cleanup helper: Purge raw telemetry batches older than max_age_hours.
        Derived features and verification results are kept permanently.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        deleted_count = (
            db.query(TelemetryBatchEntity)
            .filter(TelemetryBatchEntity.created_at < cutoff)
            .delete(synchronize_session=False)
        )
        logger.info("Cleaned up {} old telemetry batches created before {}", deleted_count, cutoff)
        return deleted_count


class FeatureRepository:
    """Feature vector data access functions."""

    @staticmethod
    def save_feature_vector(
        db: Session,
        session_id: str,
        model_version: str,
        feature_schema: List[str],
        values: List[float],
    ) -> FeatureVectorEntity:
        entity = FeatureVectorEntity(
            session_id=session_id,
            model_version=model_version,
            feature_schema=feature_schema,
            values=values,
        )
        db.add(entity)
        db.flush()
        return entity


class VerificationRepository:
    """Verification result data access functions."""

    @staticmethod
    def save_verification_result(
        db: Session,
        evaluation: VerifyResponse,
        model_version: str,
    ) -> VerificationResultEntity:
        entity = VerificationResultEntity(
            session_id=evaluation.session_id,
            model_version=model_version,
            human_probability=evaluation.human_probability,
            risk_level=evaluation.risk_level.value,
            recommended_action=evaluation.recommended_action,
            risk_score=evaluation.risk_score,
            anomaly_score=evaluation.anomaly_score,
            temporal_human_probability=evaluation.temporal_human_probability,
            risk_components=evaluation.risk_components,
            triggered_indicators=evaluation.triggered_indicators,
        )
        db.add(entity)
        db.flush()
        return entity

    @staticmethod
    def get_latest_verification_result(
        db: Session,
        session_id: str,
    ) -> Optional[VerificationResultEntity]:
        return (
            db.query(VerificationResultEntity)
            .filter(VerificationResultEntity.session_id == session_id)
            .order_by(VerificationResultEntity.created_at.desc())
            .first()
        )

    @staticmethod
    def read_analytics(db: Session) -> AnalyticsResponse:
        total_sessions = db.query(func.count(VerificationResultEntity.id)).scalar() or 0
        if total_sessions == 0:
            return AnalyticsResponse(
                total_sessions=0,
                average_human_probability=0.0,
                risk_distribution=[],
            )

        avg_prob = (
            db.query(func.avg(VerificationResultEntity.human_probability)).scalar() or 0.0
        )
        rows = (
            db.query(VerificationResultEntity.risk_level, func.count(VerificationResultEntity.id))
            .group_by(VerificationResultEntity.risk_level)
            .all()
        )
        buckets = [AnalyticsBucket(label=label, count=count) for label, count in rows]
        return AnalyticsResponse(
            total_sessions=total_sessions,
            average_human_probability=float(avg_prob),
            risk_distribution=buckets,
        )


class SecurityEventRepository:
    """Security audit event data access functions."""

    @staticmethod
    def record_events(
        db: Session,
        session_id: str,
        indicators: List[str],
        composite_risk: float,
    ) -> List[SecurityEventEntity]:
        events: List[SecurityEventEntity] = []
        for ind in indicators:
            severity = "HIGH" if "escalation" in ind or "webdriver" in ind else "MEDIUM"
            entity = SecurityEventEntity(
                session_id=session_id,
                event_type=ind,
                severity=severity,
                details={"composite_risk": composite_risk},
            )
            db.add(entity)
            events.append(entity)
        if events:
            db.flush()
        return events


class ChallengeRepository:
    """Captcha / Challenge entity data access functions."""

    @staticmethod
    def create_challenge(
        db: Session,
        session_id: str,
        challenge_type: str,
        payload: Dict[str, Any],
        solution: Dict[str, Any],
    ) -> ChallengeEntity:
        entity = ChallengeEntity(
            session_id=session_id,
            challenge_type=challenge_type,
            status="pending",
            payload=payload,
            solution=solution,
        )
        db.add(entity)
        db.flush()
        return entity

    @staticmethod
    def get_challenge(
        db: Session,
        challenge_id: int,
        session_id: str,
    ) -> Optional[ChallengeEntity]:
        return (
            db.query(ChallengeEntity)
            .filter(
                ChallengeEntity.id == challenge_id,
                ChallengeEntity.session_id == session_id,
            )
            .one_or_none()
        )
