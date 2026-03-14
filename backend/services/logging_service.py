from __future__ import annotations

import threading
from collections import Counter
from typing import List

from loguru import logger
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models.db_models import SessionEntity, VerificationResultEntity
from backend.models.schemas import AnalyticsBucket, AnalyticsResponse, VerifyResponse


_evaluations_lock = threading.Lock()
_evaluations: List[VerifyResponse] = []


def init_logging_store() -> None:
    """
    Initialize in-memory evaluation cache.
    """
    with _evaluations_lock:
        _evaluations.clear()
    logger.info("In-memory evaluation cache initialized.")


def log_evaluation_result(result: VerifyResponse) -> None:
    """
    Track evaluation results in memory for quick analytics.
    """
    with _evaluations_lock:
        _evaluations.append(result)
    logger.debug(
        "Logged evaluation result for session {}: prob={:.3f}, risk={}",
        result.session_id,
        result.human_probability,
        result.risk_level.value,
    )


def read_analytics(db: Session) -> AnalyticsResponse:
    """
    Aggregate evaluation history using persisted verification results,
    with a small in-memory cache as a supplement.
    """
    total_sessions = db.query(func.count(VerificationResultEntity.id)).scalar() or 0
    if total_sessions == 0:
        return AnalyticsResponse(
            total_sessions=0,
            average_human_probability=0.0,
            risk_distribution=[],
        )

    avg_probability = (
        db.query(func.avg(VerificationResultEntity.human_probability)).scalar() or 0.0
    )

    rows = (
        db.query(VerificationResultEntity.risk_level, func.count(VerificationResultEntity.id))
        .group_by(VerificationResultEntity.risk_level)
        .all()
    )
    buckets: List[AnalyticsBucket] = [
        AnalyticsBucket(label=label, count=count) for label, count in rows
    ]

    return AnalyticsResponse(
        total_sessions=total_sessions,
        average_human_probability=float(avg_probability),
        risk_distribution=buckets,
    )
