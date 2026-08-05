from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.api.routes import router as api_router
from backend.services.logging_service import init_logging_store
from backend.services.metrics import metrics_router
from backend.services.security_middleware import SecurityMiddleware
from backend.database.session import engine
from backend.database.base import Base


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """
    FastAPI lifespan context: initialize shared resources like
    ML model, database, and logging.
    """

    log_level = os.getenv("LOG_LEVEL", "INFO")

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=log_level.upper())

    logger.info("Initializing BotGuard AI backend...")


    # Ensure DB tables exist
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized.")

    # Load Multi-Engine Intelligence Stack (RF, Anomaly Detector, Temporal 1D CNN)
    from backend.ml.intelligence_engine import load_intelligence_stack
    logger.info("Loading Multi-Engine Intelligence Stack...")
    load_intelligence_stack()
    logger.info("Multi-Engine Intelligence Stack loaded successfully.")


    # Initialize logging store
    init_logging_store()

    logger.info("Initialization complete. Backend is ready.")

    yield

    logger.info("Shutting down BotGuard AI backend.")


app = FastAPI(
    title="BotGuard AI – Passive Human Verification",
    version="0.1.0",
    description="ML-based passive human verification service as an alternative to CAPTCHAs.",
    lifespan=lifespan,
)


# =========================
# FIXED CORS CONFIGURATION
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow requests from Vercel frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Security middleware
app.add_middleware(SecurityMiddleware)


# API routes
app.include_router(api_router, prefix="/api")
app.include_router(metrics_router)


@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """
    Basic health check for monitoring and readiness probes.
    """
    return {"status": "ok"}