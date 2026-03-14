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
from backend.ml.model import load_model
from backend.ml.model_registry import resolve_model_path
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

    registry_path = os.getenv(
        "MODEL_REGISTRY_PATH",
        "backend/ml/artifacts/model_registry.json"
    )

    model_version = os.getenv("MODEL_VERSION", "latest")
    log_level = os.getenv("LOG_LEVEL", "INFO")

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=log_level.upper())

    logger.info("Initializing BotGuard AI backend...")

    # Ensure DB tables exist
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized.")

    # Load ML model
    model_path = resolve_model_path(
        Path(registry_path),
        requested_version=model_version
    )

    logger.info("Loading ML model from {}", model_path)

    load_model(str(model_path))

    logger.info("Model loaded successfully.")

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