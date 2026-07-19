"""
PostHog event emitter — product analytics for Quant X.

Single entry point: ``track(event_name, user_id, properties)``. The
first call lazily initializes the PostHog client. When
``POSTHOG_API_KEY`` is blank, emission is a logged no-op so tests +
dev environments don't need the key.

All event names live in ``EventName`` so grep-across-codebase stays
useful. Adding a new event = add a new constant + use it.

Naming convention: ``<subject>_<past_tense_verb>`` (``signal_executed``,
``tier_gate_hit``, ``copilot_message_sent``).
"""

from __future__ import annotations

import enum
import logging
import threading
from typing import Any, Dict, Optional

from ..core.config import settings

logger = logging.getLogger(__name__)


class EventName(str, enum.Enum):
    # ── Activation / Acquisition ─────────────────────────────────────────
    SIGNUP_COMPLETED = "signup_completed"
    ONBOARDING_QUIZ_COMPLETED = "onboarding_quiz_completed"
    BROKER_CONNECTED = "broker_connected"
    TELEGRAM_CONNECTED = "telegram_connected"

    # ── Monetization / Tier ──────────────────────────────────────────────
    TIER_GATE_HIT = "tier_gate_hit"          # user blocked by RequireTier/RequireFeature
    CREDIT_CAP_HIT = "credit_cap_hit"        # Copilot daily cap reached
    UPGRADE_INITIATED = "upgrade_initiated"  # user clicked Upgrade CTA
    # PR 124 — fires once per page-view per A/B experiment when a
    # variant first becomes visible to the user. Required as the
    # *denominator* for any conversion-rate analysis on the
    # `experiment_variant` field on UPGRADE_INITIATED.
    EXPERIMENT_EXPOSED = "experiment_exposed"
    TIER_UPGRADED = "tier_upgraded"          # Razorpay payment success webhook
    TIER_DOWNGRADED = "tier_downgraded"

    # ── Trading surface ──────────────────────────────────────────────────
    SIGNAL_VIEWED = "signal_viewed"
    SIGNAL_EXECUTED_PAPER = "signal_executed_paper"
    SIGNAL_EXECUTED_LIVE = "signal_executed_live"
    AUTO_TRADE_EXECUTED = "auto_trade_executed"
    AUTO_TRADE_BLOCKED = "auto_trade_blocked"
    KILL_SWITCH_FIRED = "kill_switch_fired"
    # PR 99 — manual close of an open trade / position. Fires from both
    # /api/trades/{id}/close and /api/positions/{id}/close (shared
    # _close_trade_record path) so we track every user-initiated exit.
    POSITION_CLOSED = "position_closed"

    # ── AI features ──────────────────────────────────────────────────────
    COPILOT_MESSAGE_SENT = "copilot_message_sent"
    DEBATE_COMPLETED = "debate_completed"
    FINROBOT_ANALYSIS_COMPLETED = "finrobot_analysis_completed"

    # ── Data quality / ops ───────────────────────────────────────────────
    SCHEDULER_JOB_FAILED = "scheduler_job_failed"
    MODEL_PROMOTED = "model_promoted"
    MODEL_RETIRED = "model_retired"
    DRIFT_DETECTED = "drift_detected"
    # PR 103 — fires from the rate-limiter middleware when an IP×path
    # bucket trips its limit. Lets ops see throttling cohorts without
    # grepping logs ("which paths are getting hammered, by how many
    # distinct clients").
    RATE_LIMITED = "rate_limited"
    # PR 109 — watchlist price-alert scanner crossings.
    PRICE_ALERT_FIRED = "price_alert_fired"

    # ── Client-side reliability (PR 57) ──────────────────────────────────
    CLIENT_ERROR_CAPTURED = "client_error_captured"


# ---------------------------------------------------------------- singleton

_client: Optional[Any] = None
_client_lock = threading.Lock()
_disabled = False


def _get_client():
    """Lazy-init the posthog client. Returns None when key is unset
    or the library isn't installed."""
    global _client, _disabled
    if _disabled:
        return None
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        api_key = settings.POSTHOG_API_KEY
        if not api_key:
            _disabled = True
            return None
        try:
            import posthog
        except Exception as exc:
            logger.info("posthog library missing (%s) — events disabled", exc)
            _disabled = True
            return None
        try:
            client = posthog.Client(
                api_key=api_key,
                host=settings.POSTHOG_HOST,
                flush_at=1,          # low latency for server-side events
                sync_mode=False,     # async HTTP; fire-and-forget
            )
            _client = client
            logger.info("PostHog initialized (host=%s)", settings.POSTHOG_HOST)
            return client
        except Exception as exc:
            logger.warning("PostHog init failed: %s", exc)
            _disabled = True
            return None


# ------------------------------------------------------------------- public


def track(
    event: EventName | str,
    user_id: Optional[str],
    properties: Optional[Dict[str, Any]] = None,
) -> None:
    """Send one event to PostHog AND mirror to ``analytics_events``.

    PR-V (2026-05-28) — wires the Supabase mirror that admin endpoints
    have been silently depending on (admin/observability.py:100 reads
    this table for the experiments summary). Without the mirror, every
    tier-gate-hit / upgrade-initiated / signup event was lost the
    moment PostHog's API key was unset OR the network blipped.

    Both sinks are best-effort: a failure in either never raises.
    """
    event_name = event.value if isinstance(event, EventName) else str(event)
    props = properties or {}

    # ── PostHog (existing) ───────────────────────────────────────────
    client = _get_client()
    if client is not None:
        try:
            client.capture(
                distinct_id=user_id or "anonymous",
                event=event_name,
                properties=props,
            )
        except Exception as exc:
            logger.debug("PostHog capture failed for %s: %s", event_name, exc)

    # ── Supabase mirror (PR-V) ───────────────────────────────────────
    _mirror_to_supabase(event_name, user_id, props)


def _mirror_to_supabase(event_name: str, user_id: Optional[str], props: Dict[str, Any]) -> None:
    """Append-only insert into ``analytics_events``. Best-effort."""
    try:
        from ..api.app import get_supabase_admin  # lazy: avoid import cycles
        sb = get_supabase_admin()
        if sb is None:
            return
        sb.table("analytics_events").insert({
            "event": event_name,
            "user_id": user_id,
            "properties": props,
        }).execute()
    except Exception as exc:
        logger.debug("analytics_events mirror failed for %s: %s", event_name, exc)


def track_llm_usage(
    user_id: Optional[str],
    feature: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    latency_ms: Optional[int] = None,
    request_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one LLM-call cost row to ``llm_usage_events``.

    ``feature`` is the product surface ('copilot' | 'lab' | 'doctor' |
    'debate' | 'finrobot'). ``micros_usd`` is computed at write-time
    from the provider/model price card so the admin cost panel can
    SUM(micros_usd) without re-tokenizing. Pass tokens=0 for unknown.

    Best-effort: failures are swallowed so a stats blip never breaks
    the calling LLM request.
    """
    try:
        from ..api.app import get_supabase_admin  # lazy: avoid import cycles
        from .llm_pricing import micros_usd_for  # lazy too

        micros = micros_usd_for(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        sb = get_supabase_admin()
        if sb is None:
            return
        sb.table("llm_usage_events").insert({
            "user_id": user_id,
            "feature": feature,
            "provider": provider,
            "model": model,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "cache_read_tokens": int(cache_read_tokens or 0),
            "cache_write_tokens": int(cache_write_tokens or 0),
            "micros_usd": int(micros),
            "latency_ms": int(latency_ms) if latency_ms is not None else None,
            "request_id": request_id,
            "metadata": metadata or {},
        }).execute()
    except Exception as exc:
        logger.debug("llm_usage_events insert failed: %s", exc)


def set_user_context(
    user_id: str,
    properties: Optional[Dict[str, Any]] = None,
) -> None:
    """Identify the user in PostHog with current tier / email / etc.
    Call once on login + again on tier change."""
    client = _get_client()
    if client is None:
        return
    try:
        client.identify(distinct_id=user_id, properties=properties or {})
    except Exception as exc:
        logger.debug("PostHog identify failed for %s: %s", user_id, exc)
