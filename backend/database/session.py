from __future__ import annotations

import os
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./botguard.db",
)

# SQLite requires check_same_thread=False when used across threads (FastAPI/async).
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a SQLAlchemy session per request.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Database session error: {}", exc)
        raise
    finally:
        db.close()
