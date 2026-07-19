"""
Security-headers middleware.

Two CSP profiles based on the request path:

* ``/docs``, ``/redoc``, ``/openapi.json`` — Swagger / ReDoc need inline
  scripts + CDN assets. Ship the permissive profile so staff can still
  browse the spec.
* Everything else — strict profile. The backend is a JSON API; browsers
  should never execute scripts or load styles from its responses, so we
  drop ``unsafe-inline`` / ``unsafe-eval`` entirely and pin
  ``connect-src`` to ``'self'``.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


# Paths that serve HTML (Swagger + ReDoc) — keep loose CSP for these.
_DOCS_PATHS = {"/docs", "/redoc", "/openapi.json"}

_STRICT_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "form-action 'self'; "
    "object-src 'none'; "
    "base-uri 'self'"
)

# Loose profile only for the OpenAPI UI surfaces. Swagger ships inline
# scripts + pulls JS/CSS from jsdelivr; ReDoc does the same.
_DOCS_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
    "img-src 'self' data: https://fastapi.tiangolo.com; "
    "font-src 'self' data: https://fonts.gstatic.com; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "object-src 'none'; "
    "base-uri 'self'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach security headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-site"

        path = request.url.path
        response.headers["Content-Security-Policy"] = (
            _DOCS_CSP if path in _DOCS_PATHS else _STRICT_CSP
        )

        return response
