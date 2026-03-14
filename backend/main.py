from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, List
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


def _get_allowed_origins() -> List[str]:
    origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
    return [origin.strip() for origin in origins_env.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:  # pragma: no cover - startup/shutdown wiring
    """
    FastAPI lifespan context: initialize shared resources like the ML model and logging store.
    """
    registry_path = os.getenv(
        "MODEL_REGISTRY_PATH", "backend/ml/artifacts/model_registry.json"
    )
    model_version = os.getenv("MODEL_VERSION", "latest")
    log_level = os.getenv("LOG_LEVEL", "INFO")

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=log_level.upper())

    logger.info("Initializing BotGuard AI backend...")

    # Ensure all database tables exist (creates them if absent).
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized.")

    model_path = resolve_model_path(Path(registry_path), requested_version=model_version)
    logger.info("Loading ML model from {}", model_path)
    load_model(str(model_path))
    logger.info("Model loaded successfully.")

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


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SecurityMiddleware)

app.include_router(api_router, prefix="/api")
app.include_router(metrics_router)


@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """
    Basic health check for readiness probes and monitoring.
    """
    return {"status": "ok"}
