from __future__ import annotations

from datetime import datetime
from typing import Tuple
import random

from loguru import logger
from sqlalchemy.orm import Session

from backend.models.db_models import ChallengeEntity, SessionEntity
from backend.models.schemas import (
    ChallengeStartRequest,
    ChallengeStartResponse,
    ChallengeType,
    ChallengeVerifyRequest,
    ChallengeVerifyResponse,
)


def _pick_type(preferred: ChallengeType | None) -> ChallengeType:
    if preferred:
        return preferred
    return random.choice(
        [ChallengeType.SLIDER, ChallengeType.DRAG_DROP, ChallengeType.PATTERN, ChallengeType.REACTION]
    )


def _build_slider_challenge() -> Tuple[dict, dict]:
    target = random.randint(20, 80)
    tolerance = 5
    token = random.randint(100000, 999999)
    payload = {"type": "slider", "target": target, "tolerance": tolerance, "token": token}
    solution = {"token": token, "min": target - tolerance, "max": target + tolerance}
    return payload, solution


def _build_pattern_challenge() -> Tuple[dict, dict]:
    # Simple numeric pattern 2,4,6,?
    token = random.randint(100000, 999999)
    payload = {
        "type": "pattern",
        "sequence": [2, 4, 6],
        "token": token,
    }
    solution = {"token": token, "answer": 8}
    return payload, solution


def _build_reaction_challenge() -> Tuple[dict, dict]:
    token = random.randint(100000, 999999)
    payload = {
        "type": "reaction",
        "token": token,
        "min_ms": 200,
        "max_ms": 8000,
    }
    solution = {"token": token, "min_ms": 200, "max_ms": 8000}
    return payload, solution


def _build_drag_drop_challenge() -> Tuple[dict, dict]:
    token = random.randint(100000, 999999)
    target_zone = random.choice(["zone-a", "zone-b", "zone-c"])
    payload = {
        "type": "drag_drop",
        "target_zone": target_zone,
        "token": token,
    }
    solution = {"token": token, "target_zone": target_zone}
    return payload, solution


def start_challenge(db: Session, req: ChallengeStartRequest) -> ChallengeStartResponse:
    session = (
        db.query(SessionEntity)
        .filter(SessionEntity.session_id == req.session_id)
        .one_or_none()
    )
    if session is None:
        session = SessionEntity(session_id=req.session_id)
        db.add(session)
        db.flush()

    ctype = _pick_type(req.preferred_type)
    if ctype is ChallengeType.SLIDER:
        payload, solution = _build_slider_challenge()
    elif ctype is ChallengeType.PATTERN:
        payload, solution = _build_pattern_challenge()
    elif ctype is ChallengeType.REACTION:
        payload, solution = _build_reaction_challenge()
    else:
        payload, solution = _build_drag_drop_challenge()

    entity = ChallengeEntity(
        session_id=req.session_id,
        challenge_type=ctype.value,
        status="pending",
        payload=payload,
        solution=solution,
    )
    db.add(entity)
    db.commit()
    db.refresh(entity)

    logger.info("Started challenge {} for session {} type={}", entity.id, req.session_id, ctype.value)

    return ChallengeStartResponse(
        challenge_id=entity.id,
        session_id=req.session_id,
        challenge_type=ctype,
        payload=payload,
    )


def verify_challenge(db: Session, req: ChallengeVerifyRequest) -> ChallengeVerifyResponse:
    entity = (
        db.query(ChallengeEntity)
        .filter(
            ChallengeEntity.id == req.challenge_id,
            ChallengeEntity.session_id == req.session_id,
        )
        .one_or_none()
    )
    if entity is None:
        raise ValueError("Challenge not found")

    entity.attempts += 1

    sol = entity.solution
    resp = req.response_payload
    success = False

    try:
        if entity.challenge_type == "slider":
            success = (
                resp.get("token") == sol.get("token")
                and sol.get("min") <= float(resp.get("value", -1)) <= sol.get("max")
            )
        elif entity.challenge_type == "pattern":
            success = resp.get("token") == sol.get("token") and int(resp.get("answer")) == sol.get("answer")
        elif entity.challenge_type == "reaction":
            success = resp.get("token") == sol.get("token") and sol.get("min_ms") <= float(
                resp.get("reaction_ms", 0)
            ) <= sol.get("max_ms")
        elif entity.challenge_type == "drag_drop":
            success = (
                resp.get("token") == sol.get("token")
                and resp.get("target_zone") == sol.get("target_zone")
            )
    except Exception:
        success = False

    entity.success = success
    entity.status = "solved" if success else "failed"
    entity.solved_at = datetime.utcnow()
    db.commit()

    logger.info(
        "Challenge {} verification for session {} success={}",
        entity.id,
        entity.session_id,
        success,
    )

    return ChallengeVerifyResponse(
        challenge_id=entity.id,
        session_id=entity.session_id,
        success=success,
        updated_risk_score=None,
    )
