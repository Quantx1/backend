"""
================================================================================
TELEMETRY ROUTES — client-side error ingestion (PR 57)
================================================================================
One endpoint: ``POST /api/client-errors``. The frontend error boundaries
(``app/global-error.tsx``, per-route ``error.tsx``, widget
``ErrorBoundary``) send a compact JSON payload here whenever React
can't render. We forward the report to structured logs + PostHog so we
spot rendering regressions without waiting on a Sentry bill.

Auth:
    Public. A crash can happen before auth loads, and we'd rather
    capture the error than lose it. The rate limiter middleware caps
    abuse at 60 req/min per IP (global default); more than enough for
    a browser that's genuinely failing.

Payload size cap:
    8 KB max total. Messages longer than 500 chars and stacks longer
    than 4 KB are truncated with a marker. Protects us from giant
    React component trees in the stack frames.
================================================================================
"""

from __future__ import annotations

import logging
import secrets
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/client-errors", tags=["telemetry"])


_MAX_MESSAGE_CHARS = 500
_MAX_STACK_CHARS = 4000
_MAX_BODY_BYTES = 8 * 1024  # 8 KB


class ClientErrorReport(BaseModel):
    name: str = Field(default="Error", max_length=120)
    message: str = Field(default="", max_length=_MAX_MESSAGE_CHARS)
    stack: Optional[str] = Field(default=None, max_length=_MAX_STACK_CHARS)
    digest: Optional[str] = Field(default=None, max_length=120)
    boundary: Optional[str] = Field(
        default=None, max_length=60,
        description="global | route | widget — which boundary caught it",
    )
    route: Optional[str] = Field(default=None, max_length=256)
    user_id: Optional[str] = Field(default=None, max_length=64)
    app_version: Optional[str] = Field(default=None, max_length=64)


class Ack(BaseModel):
    ok: bool = True
    # PR 102 — server-issued correlation id. Returned to the frontend
    # so a user filing a support ticket can paste it for fast log
    # lookup. Also tagged on the Sentry event + PostHog payload so all
    # three surfaces (server logs, Sentry, PostHog) share one key.
    request_id: Optional[str] = None


def _new_request_id() -> str:
    """8-char URL-safe id. Short enough for a user to read aloud /
    paste in a ticket; long enough that collisions inside a 60s
    window are negligible."""
    return secrets.token_urlsafe(6)


@router.post("", response_model=Ack)
async def ingest_client_error(
    body: ClientErrorReport,
    request: Request,
) -> Ack:
    """Accept one client crash report. Always returns 200 — we don't
    want a telemetry failure to cascade into a user-visible error.

    PR 102 — body-cap check moved to ``Content-Length`` header (the
    previous ``await request.body()`` ran *after* FastAPI/Pydantic had
    already consumed and parsed the body, so it never tripped). The
    header check is the only pre-parse guard available since Pydantic
    enforces per-field max_length, not whole-payload size.
    """
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > _MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="payload_too_large")
        except ValueError:
            # Bogus Content-Length — let Pydantic field limits handle.
            pass

    ua = (request.headers.get("user-agent") or "")[:200]
    request_id = _new_request_id()

    # Attribution comes from the JWT, NEVER the client-supplied body.user_id
    # (which is spoofable). Mirror the derive-from-bearer pattern used by
    # the upgrade-intent / experiment-exposed endpoints below. No token →
    # anonymous (None). Defensive: never raise on a missing/invalid token.
    user_id: Optional[str] = None
    try:
        auth = (request.headers.get("authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            import jwt as pyjwt  # noqa: PLC0415 — lazy
            payload = pyjwt.decode(auth[7:], options={"verify_signature": False})
            user_id = payload.get("sub") if isinstance(payload, dict) else None
    except Exception:
        user_id = None

    # Structured log line — goes to stdout + (if configured) Loki/Datadog.
    # Lead with request_id so log scans grep it as the first column.
    logger.warning(
        "client_error rid=%s boundary=%s name=%s route=%s digest=%s msg=%s",
        request_id,
        body.boundary or "unknown",
        body.name,
        body.route or "-",
        body.digest or "-",
        (body.message or "")[:200].replace("\n", " "),
    )

    # PostHog fan-out — best-effort, never raises.
    try:
        from ..observability import EventName, track
        track(EventName.CLIENT_ERROR_CAPTURED, user_id, {
            "request_id": request_id,
            "name": body.name,
            "message": body.message[:_MAX_MESSAGE_CHARS],
            "boundary": body.boundary,
            "route": body.route,
            "digest": body.digest,
            "app_version": body.app_version,
            "user_agent": ua,
            "has_stack": bool(body.stack),
        })
    except Exception as exc:
        logger.debug("posthog client-error emit skipped: %s", exc)

    # PR 97 — also fan out to Sentry server-side. The frontend has no
    # @sentry/nextjs client; routing client crashes through this server
    # endpoint is how they end up in Sentry. No-op when Sentry isn't
    # initialized (see observability/sentry_helpers.py).
    try:
        from ..observability.sentry_helpers import capture_message, set_tag
        # PR 102 — request_id as a Sentry tag so the dashboard's filter
        # column matches the same id that lands in server logs and the
        # frontend toast (when shown).
        set_tag("request_id", request_id)
        if body.boundary:
            set_tag("error_boundary", body.boundary)
        if body.route:
            set_tag("route", body.route)
        capture_message(
            f"[client {body.boundary or 'unknown'}] {body.name}: {body.message[:200]}",
            level="error",
            extra={
                "request_id": request_id,
                "stack": (body.stack or "")[:_MAX_STACK_CHARS],
                "digest": body.digest,
                "user_id": user_id,
                "app_version": body.app_version,
                "user_agent": ua,
            },
        )
    except Exception as exc:
        logger.debug("sentry client-error emit skipped: %s", exc)

    return Ack(request_id=request_id)


_UPGRADE_TIERS = {"pro", "elite"}
_UPGRADE_SOURCES = {
    # Allowlist so the field stays a clean conversion-funnel column.
    "pricing_page",
    "settings_tier_panel",
    "copilot_quota_modal",
    "tier_gate_block",
    "signal_lock",
    "feature_lock",
    # PR 122 — finer-grained slugs for the quiz-recommendation banner
    # so we can tell whether the personalized banner copy (PR 120) or
    # the "What changes" expand (PR 121) is what actually drives the
    # click — vs the plain Compare-plans CTA. Each surface has its own
    # slug so we don't lose the distinction during analysis.
    "quiz_rec_banner_pricing",
    "quiz_rec_banner_settings",
    "quiz_rec_card_highlight",
    "quiz_rec_what_changes",
}


class UpgradeIntentBody(BaseModel):
    target_tier: str = Field(..., max_length=20)
    source: str = Field(..., max_length=40)
    # PR 123 — optional A/B variant tag so per-arm conversion is
    # decomposable in the funnel report. Allowlisted to avoid free-form
    # cardinality explosions.
    experiment_variant: Optional[str] = Field(default=None, max_length=40)


@router.post("/upgrade-intent", response_model=Ack)
async def upgrade_intent(
    body: UpgradeIntentBody,
    request: Request,
) -> Ack:
    """PR 100 — fires ``UPGRADE_INITIATED`` from any user-facing upgrade
    CTA. Backend allowlists target_tier + source so a malformed client
    payload doesn't pollute the conversion-funnel cohort. Auth is
    optional — anonymous visitors hitting /pricing should also count.
    """
    target = body.target_tier.lower().strip()
    source = body.source.lower().strip()
    if target not in _UPGRADE_TIERS:
        raise HTTPException(status_code=422, detail="invalid_target_tier")
    if source not in _UPGRADE_SOURCES:
        raise HTTPException(status_code=422, detail="invalid_source")

    # Best-effort user resolution — we can't depend on get_current_user
    # because anonymous /pricing visitors should still produce events.
    user_id: Optional[str] = None
    try:
        auth = (request.headers.get("authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            import jwt as pyjwt  # noqa: PLC0415 — lazy
            payload = pyjwt.decode(auth[7:], options={"verify_signature": False})
            user_id = payload.get("sub") if isinstance(payload, dict) else None
    except Exception:
        user_id = None

    try:
        from ..observability import EventName, track
        # PR 123 — allowlist of A/B variant strings so we don't pollute
        # the analytics warehouse with unbounded labels.
        _VALID_VARIANTS = {"feature_led", "outcome_led"}
        variant = (body.experiment_variant or "").lower().strip() or None
        if variant and variant not in _VALID_VARIANTS:
            variant = None
        props: Dict[str, Any] = {"target_tier": target, "source": source}
        if variant:
            props["experiment_variant"] = variant
        track(EventName.UPGRADE_INITIATED, user_id, props)
    except Exception as exc:
        logger.debug("posthog upgrade_intent emit skipped: %s", exc)

    return Ack()


# ============================================================================
# PR 124 — A/B experiment exposure
# ============================================================================
#
# The `experiment_variant` tag on UPGRADE_INITIATED gives us conversion
# *numerators* per arm but no denominators — we don't know how many
# users of each arm even saw the banner. EXPERIMENT_EXPOSED fires once
# per page-view per experiment when a variant first becomes visible.
# Conversion rate per arm = UPGRADE_INITIATED / EXPERIMENT_EXPOSED
# filtered to the same `experiment` + `variant` pair.

_EXPERIMENT_KEYS = {
    "quiz_rec_delta_copy",  # PR 123 — feature_led vs outcome_led
}
_EXPERIMENT_VARIANTS = {
    "feature_led",
    "outcome_led",
}


class ExperimentExposedBody(BaseModel):
    experiment: str = Field(..., max_length=60)
    variant: str = Field(..., max_length=40)
    # PR 127 — optional tier slice so per-arm conversion is decomposable
    # by user tier. Allowlisted so a malformed client can't pollute
    # the warehouse with arbitrary labels.
    current_tier: Optional[str] = Field(default=None, max_length=20)


@router.post("/experiment-exposed", response_model=Ack)
async def experiment_exposed(
    body: ExperimentExposedBody,
    request: Request,
) -> Ack:
    """Record that a user saw a specific A/B variant. Fire-once per
    page-view per experiment from the client; we don't dedupe server-side
    because PostHog handles that downstream via $insert_id.
    """
    experiment = body.experiment.lower().strip()
    variant = body.variant.lower().strip()
    if experiment not in _EXPERIMENT_KEYS or variant not in _EXPERIMENT_VARIANTS:
        # Silent reject — telemetry must never throw at the caller.
        return Ack()

    user_id: Optional[str] = None
    try:
        auth = (request.headers.get("authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            import jwt as pyjwt  # noqa: PLC0415 — lazy
            payload = pyjwt.decode(auth[7:], options={"verify_signature": False})
            user_id = payload.get("sub") if isinstance(payload, dict) else None
    except Exception:
        user_id = None

    # PR 127 — tier allowlist (mirrors _UPGRADE_TIERS but adds free).
    _VALID_TIERS = {"free", "pro", "elite"}
    tier = (body.current_tier or "").lower().strip() or None
    if tier and tier not in _VALID_TIERS:
        tier = None

    try:
        from ..observability import EventName, track
        props: Dict[str, Any] = {
            "experiment": experiment,
            "experiment_variant": variant,
        }
        if tier:
            props["current_tier"] = tier
        track(EventName.EXPERIMENT_EXPOSED, user_id, props)
    except Exception as exc:
        logger.debug("posthog experiment_exposed emit skipped: %s", exc)

    return Ack()


__all__ = ["router"]
