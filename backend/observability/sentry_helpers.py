"""
Sentry helpers — thin wrappers that never raise when sentry_sdk isn't
installed. Keeps feature code free of try/except boilerplate.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _sdk():
    try:
        import sentry_sdk  # noqa: PLC0415 — lazy
        return sentry_sdk
    except Exception:
        return None


def capture_exception(exc: BaseException, *, extra: Optional[dict] = None) -> None:
    """Send an exception to Sentry with optional extra context. No-op
    when sentry isn't initialized."""
    sdk = _sdk()
    if sdk is None:
        return
    try:
        if extra:
            with sdk.push_scope() as scope:
                for k, v in extra.items():
                    scope.set_extra(k, v)
                sdk.capture_exception(exc)
        else:
            sdk.capture_exception(exc)
    except Exception:
        pass


def add_breadcrumb(*, category: str, message: str, data: Optional[dict] = None,
                   level: str = "info") -> None:
    """Add a Sentry breadcrumb — shows in error reports as context."""
    sdk = _sdk()
    if sdk is None:
        return
    try:
        sdk.add_breadcrumb(
            category=category, message=message,
            data=data or {}, level=level,
        )
    except Exception:
        pass


def set_tag(key: str, value: Any) -> None:
    """Set a tag on the current Sentry scope (shows as filterable column
    in the Sentry UI). Common tags: ``tier``, ``feature``, ``symbol``."""
    sdk = _sdk()
    if sdk is None:
        return
    try:
        sdk.set_tag(key, str(value))
    except Exception:
        pass


def capture_message(message: str, *, level: str = "info",
                    extra: Optional[dict] = None) -> None:
    """Send a non-exception event to Sentry (e.g., a client-side crash
    forwarded from the browser via /api/client-errors). Like
    ``capture_exception`` but without an exception object — Sentry
    treats it as a free-form event with the given level. No-op when
    sentry isn't initialized."""
    sdk = _sdk()
    if sdk is None:
        return
    try:
        if extra:
            with sdk.push_scope() as scope:
                for k, v in extra.items():
                    scope.set_extra(k, v)
                sdk.capture_message(message, level=level)
        else:
            sdk.capture_message(message, level=level)
    except Exception:
        pass
