from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base


def utc_now() -> datetime:
    """Helper to generate timezone-aware UTC datetime instances."""
    return datetime.now(timezone.utc)


class SessionEntity(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    ip_address: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    browser_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    last_human_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_risk_level: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    telemetry_batches: Mapped[List[TelemetryBatchEntity]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    feature_vectors: Mapped[List[FeatureVectorEntity]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    verification_results: Mapped[List[VerificationResultEntity]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    security_events: Mapped[List[SecurityEventEntity]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class TelemetryBatchEntity(Base):
    __tablename__ = "telemetry_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("sessions.session_id"), index=True
    )

    started_at_ms: Mapped[float] = mapped_column(Float)
    ended_at_ms: Mapped[float] = mapped_column(Float)

    event_counts: Mapped[Dict[str, Any]] = mapped_column(JSON)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    session: Mapped[SessionEntity] = relationship(back_populates="telemetry_batches")


class FeatureVectorEntity(Base):
    __tablename__ = "feature_vectors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    session_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("sessions.session_id"), index=True
    )
    model_version: Mapped[str] = mapped_column(String(64), index=True)

    feature_schema: Mapped[List[str]] = mapped_column(JSON)
    values: Mapped[List[float]] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )

    session: Mapped[SessionEntity] = relationship(back_populates="feature_vectors")


class VerificationResultEntity(Base):
    __tablename__ = "verification_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    session_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("sessions.session_id"), index=True
    )
    model_version: Mapped[str] = mapped_column(String(64), index=True)

    human_probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    recommended_action: Mapped[str] = mapped_column(String(32))
    risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Expanded explainability fields (Risk Engine 2.0)
    anomaly_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temporal_human_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_components: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    triggered_indicators: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )

    session: Mapped[SessionEntity] = relationship(back_populates="verification_results")


class SecurityEventEntity(Base):
    __tablename__ = "security_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    session_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("sessions.session_id"), index=True
    )

    event_type: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16), index=True, default="MEDIUM")
    details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )

    session: Mapped[SessionEntity] = relationship(back_populates="security_events")


class ChallengeEntity(Base):
    __tablename__ = "challenges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("sessions.session_id"), index=True
    )

    challenge_type: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON)
    solution: Mapped[Dict[str, Any]] = mapped_column(JSON)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[Optional[bool]] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )
    solved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    session: Mapped[SessionEntity] = relationship()
