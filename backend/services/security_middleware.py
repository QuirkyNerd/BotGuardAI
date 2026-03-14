from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.services.metrics import REQUEST_LATENCY


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Lightweight security middleware that performs:
    - Request latency metrics
    - Simple IP-based rate limiting
    - Basic browser / automation framework detection (headless, webdriver)

    This is intentionally in-memory and per-process for simplicity.
    """

    def __init__(self, app: ASGIApp, max_requests_per_minute: int = 120) -> None:
        super().__init__(app)
        self.max_requests_per_minute = max_requests_per_minute
        self._ip_buckets: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        start_time = time.perf_counter()

        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # Naive IP rate limiting
        now = time.time()
        window_start = now - 60
        history = self._ip_buckets.setdefault(client_ip, [])
        history = [ts for ts in history if ts >= window_start]
        history.append(now)
        self._ip_buckets[client_ip] = history

        if len(history) > self.max_requests_per_minute:
            logger.warning("Rate limit exceeded for IP {}", client_ip)
            return Response(status_code=429, content="Too Many Requests")

        # Basic headless / automation fingerprinting from headers.
        user_agent = request.headers.get("user-agent", "")
        webdriver_flag = request.headers.get("x-botguard-webdriver", "false")

        suspicious = False
        if "Headless" in user_agent or "Puppeteer" in user_agent:
            suspicious = True
        if webdriver_flag.lower() == "true":
            suspicious = True

        request.state.security_context = {
            "client_ip": client_ip,
            "user_agent": user_agent,
            "suspicious": suspicious,
            "recent_request_count": len(history),
        }

        response = await call_next(request)

        elapsed = time.perf_counter() - start_time
        REQUEST_LATENCY.labels(method=request.method, path=path).observe(elapsed)

        return response
