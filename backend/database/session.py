from __future__ import annotations

import os
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def get_database_url() -> str:
    """
    Authoritative database URL configuration.
    Falls back to local SQLite if DATABASE_URL is not set in environment.
    """
    return os.getenv("DATABASE_URL", "sqlite:///./botguard.db")


def build_engine(database_url: Optional[str] = None) -> Engine:
    """
    Build SQLAlchemy engine configured appropriately for SQLite or PostgreSQL.
    """
    url = database_url or get_database_url()

    is_sqlite = url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}

    engine_kwargs = {
        "connect_args": connect_args,
        "pool_pre_ping": True,
        "future": True,
    }

    if not is_sqlite:
        engine_kwargs.update(
            {
                "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
                "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
            }
        )

    logger.info(
        "Creating database engine for dialect: {}",
        "sqlite" if is_sqlite else "postgresql",
    )
    return create_engine(url, **engine_kwargs)


DATABASE_URL = get_database_url()
engine = build_engine(DATABASE_URL)

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
    Ensures safe session lifecycle: rollback on exception and guaranteed close.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as exc:
        logger.exception("Database session error during request execution: {}", exc)
        db.rollback()
        raise
    finally:
        db.close()
