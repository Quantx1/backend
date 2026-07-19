"""
Observability — Sentry + PostHog + signal-accuracy drift telemetry.

Two surfaces:

- **Sentry** (already initialized in ``app.py`` startup). We expose
  helpers here so feature code can add tags / breadcrumbs / captured
  exceptions without reaching into ``sentry_sdk`` directly.

- **PostHog** (this PR's addition). Product-analytics event emitter —
  one-line calls like ``track("tier_gate_hit", user_id, feature=...)``.
  When ``POSTHOG_API_KEY`` is blank the emitter degrades to a no-op so
  unit tests + dev environments don't need a key.

Public API::

    from backend.observability import track, capture_exception,
                                          set_user_context, EventName

    track(EventName.TIER_GATE_HIT, user_id, {"feature": "debate"})
    capture_exception(exc, extra={"symbol": "TCS"})
"""

from .posthog_events import EventName, set_user_context, track, track_llm_usage
from .sentry_helpers import add_breadcrumb, capture_exception, set_tag

__all__ = [
    "EventName",
    "add_breadcrumb",
    "capture_exception",
    "set_tag",
    "set_user_context",
    "track",
    "track_llm_usage",
]
