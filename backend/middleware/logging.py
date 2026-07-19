"""
Structured JSON logging middleware with correlation IDs.
"""
import time
import json
import uuid
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class StructuredJsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach correlation_id if present
        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id:
            log_entry["correlation_id"] = correlation_id
        # Attach extra structured fields
        for key in ("method", "path", "status", "duration_ms", "client_ip", "user_id"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


def configure_structured_logging(level: str = "INFO"):
    """Replace root logger handlers with structured JSON output."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    # Remove existing handlers to avoid duplicate output
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredJsonFormatter())
    root.addHandler(handler)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests/responses with correlation ID and structured fields."""

    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        start_time = time.time()

        # Attach correlation_id to request state so downstream code can use it
        request.state.correlation_id = correlation_id

        logger.info(
            "Request started",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        response = await call_next(request)

        duration_ms = round((time.time() - start_time) * 1000, 1)

        logger.info(
            "Request completed",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time"] = f"{duration_ms / 1000:.4f}"

        return response
