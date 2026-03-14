from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base


class SessionEntity(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    browser_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    last_human_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_risk_level: Mapped[str | None] = mapped_column(String(16), nullable=True)

    telemetry_batches: Mapped[list["TelemetryBatchEntity"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    feature_vectors: Mapped[list["FeatureVectorEntity"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    verification_results: Mapped[list["VerificationResultEntity"]] = relationship(
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

    event_counts: Mapped[dict] = mapped_column(JSON)
    payload: Mapped[dict] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    session: Mapped[SessionEntity] = relationship(back_populates="telemetry_batches")


class FeatureVectorEntity(Base):
    __tablename__ = "feature_vectors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    session_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("sessions.session_id"), index=True
    )
    model_version: Mapped[str] = mapped_column(String(64), index=True)

    feature_schema: Mapped[list[str]] = mapped_column(JSON)
    values: Mapped[list[float]] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, index=True
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
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, index=True
    )

    session: Mapped[SessionEntity] = relationship(back_populates="verification_results")


class ChallengeEntity(Base):
    __tablename__ = "challenges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("sessions.session_id"), index=True
    )

    challenge_type: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    payload: Mapped[dict] = mapped_column(JSON)
    solution: Mapped[dict] = mapped_column(JSON)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, index=True
    )
    solved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    session: Mapped[SessionEntity] = relationship()
