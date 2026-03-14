from __future__ import annotations

import os

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


METRICS_ENABLED = os.getenv("PROMETHEUS_METRICS_ENABLED", "true").lower() == "true"

REQUEST_LATENCY = Histogram(
    "botguard_request_latency_seconds",
    "Latency of HTTP requests",
    ["method", "path"],
)

VERIFICATION_LATENCY = Histogram(
    "botguard_verification_latency_seconds",
    "Latency of /verify-session evaluations",
)

VERIFICATION_OUTCOME = Counter(
    "botguard_verification_outcome_total",
    "Count of verification outcomes by risk level",
    ["risk_level"],
)

MODEL_INFERENCE_LATENCY = Histogram(
    "botguard_model_inference_latency_seconds",
    "Latency of ML model inference",
)


metrics_router = APIRouter()


@metrics_router.get("/metrics")
async def metrics() -> Response:
    """
    Expose Prometheus metrics when enabled.
    """
    if not METRICS_ENABLED:
        return Response(status_code=404)
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

