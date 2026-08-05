from __future__ import annotations

import threading
from typing import List

from loguru import logger
from sqlalchemy.orm import Session

from backend.database.repository import VerificationRepository
from backend.models.schemas import AnalyticsResponse, VerifyResponse

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
    Aggregate evaluation history using VerificationRepository.
    """
    return VerificationRepository.read_analytics(db)
