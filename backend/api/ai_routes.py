"""
================================================================================
AI AGENT ROUTES — Copilot / FinRobot / TradingAgents (PR 8)
================================================================================
All three graphs live in ``backend/ai/agents/``. This router is the
HTTP boundary that:
  - authenticates the user (get_current_user)
  - meters credits against the existing AssistantCreditLimiter (Copilot)
  - kicks off the graph run
  - returns the structured output + agent trace for UI rendering.

Endpoints:
  POST /api/ai/copilot/chat       — N1 context-aware chat
  POST /api/ai/finrobot/analyze   — F5/F7 portfolio-doctor 4-agent CoT
  POST /api/ai/debate/signal/{id} — B1 Bull/Bear 7-agent debate (Elite)
================================================================================
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.database import get_supabase_admin
from ..core.tiers import Tier, UserTier
from ..middleware.llm_caps import enforce_llm_cap
from ..middleware.tier_gate import RequireFeature, RequireTier, copilot_daily_cap
from ..ai.agents import (
    run_copilot,
    run_copilot_stream,
    run_finrobot_doctor,
    run_trading_debate,
)
from ..ai.agents.conversation_memory import load_memory, maybe_refresh_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["ai"])


# ============================================================================
# COPILOT — N1 chat (every platform page)
# ============================================================================


class CopilotChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    route: Optional[str] = Field(None, description="Current frontend route for context")
    history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Prior turns [{role: 'user'|'assistant', content: '...'}]",
    )
    mentioned_symbols: Optional[List[str]] = Field(
        default=None,
        description="Symbols the user @-mentioned in the composer",
    )
    # PR-BF.2 — Conversation persistence
    conversation_id: Optional[str] = Field(
        default=None,
        description="When set, append turns to this saved thread; when null + persist=True, create a new one.",
    )
    persist: bool = Field(
        default=False,
        description="When true, save user + assistant turns to copilot_conversations/messages.",
    )
    display_message: Optional[str] = Field(
        default=None,
        max_length=2000,
        description=(
            "What the user actually typed, when `message` carries composer "
            "scaffolding (mode directives, action guards). Persisted/titled "
            "instead of `message` so reopened threads read clean."
        ),
    )


class CopilotChatResponse(BaseModel):
    reply: str
    refused: bool
    intent: Optional[str] = None
    tools_used: List[str] = Field(default_factory=list)
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_id: Optional[str] = None  # populated when persist=True
    grounding: Optional[Dict[str, Any]] = None  # FLAG-ONLY shadow grounding self-check
    progress: List[Dict[str, Any]] = Field(default_factory=list)  # WP-RAILS honest step timeline
    references: List[Dict[str, Any]] = Field(default_factory=list)  # WP-RAILS cited market-data entities


# --------------------------------------------------------------------------
# Shared helpers — used by both the JSON (`/copilot/chat`) and the SSE
# (`/copilot/chat/stream`) endpoints so cap-check, history-load and
# persistence stay in one place.
# --------------------------------------------------------------------------


def _enforce_copilot_cap(user_tier: UserTier) -> None:
    """Consume one daily copilot credit; raise HTTP 402 (structured) when over cap.

    Free = 5/day · Pro = 50 · Elite = 200 (core.tiers.COPILOT_DAILY_CAPS). Admins bypass.
    """
    if user_tier.is_admin:
        return
    from ..services.assistant.credit_limiter import AssistantCreditLimiter

    cap = copilot_daily_cap(user_tier.tier)
    try:
        limiter = AssistantCreditLimiter(get_supabase_admin())
        allowed, _usage = limiter.consume_if_available(
            user_id=user_tier.user_id,
            profile={"tier": user_tier.tier.value},
            cost=1,
        )
    except Exception as exc:
        # On limiter failure (DB down etc.) fail OPEN — never block a paying
        # user's chat on a metering glitch; the $20 kill-switch still applies.
        logger.debug("credit consume skipped (%s) — proceeding", exc)
        return

    if not allowed:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "credit_cap",
                "current_tier": user_tier.tier.value,
                "credits_limit": cap,
                "upgrade_url": "/pricing",
            },
        )


def _owns_conversation(sb, conversation_id: str, user_id: str) -> bool:
    """True only if this conversation belongs to this user. The admin client
    bypasses RLS, so every conversation_id from the client MUST be ownership-
    checked before read/write (prevents IDOR across users' threads)."""
    if not conversation_id or not user_id:
        return False
    try:
        row = (
            sb.table("copilot_conversations")
            .select("user_id")
            .eq("id", conversation_id)
            .limit(1)
            .execute()
            .data
        )
        return bool(row) and row[0].get("user_id") == user_id
    except Exception as exc:  # noqa: BLE001
        logger.debug("conversation ownership check failed: %s", exc)
        return False


def _load_copilot_history(body: "CopilotChatRequest", user_id: str) -> List[Dict[str, str]]:
    """Load prior turns from the saved thread when the caller passes a
    conversation_id but no inline history — keeps the frontend thin. Only reads
    the thread when it belongs to the caller (IDOR guard)."""
    if not (body.conversation_id and not body.history and body.persist):
        return body.history or []
    try:
        sb = get_supabase_admin()
        if not _owns_conversation(sb, body.conversation_id, user_id):
            return body.history or []  # not the caller's thread — never leak it
        past = (
            sb.table("copilot_messages")
            .select("role, content")
            .eq("conversation_id", body.conversation_id)
            .order("created_at", desc=False)
            .limit(20)
            .execute()
            .data
            or []
        )
        return [{"role": m["role"], "content": m["content"]} for m in past]
    except Exception as exc:
        logger.debug("history load skipped: %s", exc)
        return body.history or []


def _persist_copilot_turns(
    *,
    user_id: str,
    body: "CopilotChatRequest",
    reply: str,
    tools_used: List[str],
    intent: Optional[str],
    refused: bool,
    trace: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Append the user + assistant turns. Lazily creates the conversation
    when the caller asked to persist but passed no id. Returns the id."""
    conversation_id: Optional[str] = body.conversation_id
    if not body.persist:
        return conversation_id
    # Store what the user typed, not the composer's scaffolded message.
    user_text = (body.display_message or body.message).strip() or body.message
    try:
        from datetime import datetime
        sb = get_supabase_admin()
        now_iso = datetime.utcnow().isoformat() + "Z"
        # IDOR guard: never append to a conversation the caller doesn't own —
        # drop the id so a fresh, owned thread is created instead.
        if conversation_id and not _owns_conversation(sb, conversation_id, user_id):
            conversation_id = None
        if not conversation_id:
            title = user_text[:60].strip()
            if len(user_text) > 60:
                title = title + "…"
            new_conv = sb.table("copilot_conversations").insert({
                "user_id": user_id,
                "title": title or "New chat",
            }).execute()
            conversation_id = (new_conv.data or [{}])[0].get("id")
        if conversation_id:
            sb.table("copilot_messages").insert([
                {
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": user_text,
                },
                {
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": reply,
                    "tools_used": tools_used or [],
                    "trace": trace or [],
                    "intent": intent,
                    "refused": bool(refused),
                },
            ]).execute()
            sb.table("copilot_conversations").update({
                "updated_at": now_iso,
            }).eq("id", conversation_id).execute()
        try:
            mem = load_memory(user_id)
            cnt = (sb.table("copilot_messages").select("id", count="exact")
                   .eq("conversation_id", conversation_id).eq("role", "assistant").execute())
            total_turns = cnt.count or 0
            maybe_refresh_memory(user_id=user_id, total_turns=total_turns,
                                 recent_turns=[{"role": "user", "content": user_text},
                                               {"role": "assistant", "content": reply}],
                                 prev_summary=mem["summary"],
                                 prev_turns_summarized=mem["turns_summarized"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("memory refresh hook skipped: %s", exc)
    except Exception as exc:
        logger.warning("conversation persist failed: %s", exc)
    return conversation_id


def _track_copilot_message(user_tier: UserTier, body: "CopilotChatRequest", *, intent, tools_used, refused) -> None:
    try:
        from ..observability import EventName, track
        track(EventName.COPILOT_MESSAGE_SENT, user_tier.user_id, {
            "tier": user_tier.tier.value,
            "intent": intent,
            "tools_used": tools_used or [],
            "refused": refused,
            "route": body.route or "",
        })
    except Exception:
        pass


@router.post("/copilot/chat", response_model=CopilotChatResponse)
async def copilot_chat(
    body: CopilotChatRequest,
    user_tier: UserTier = Depends(RequireFeature("copilot_chat")),
) -> CopilotChatResponse:
    """One-turn AI Copilot chat (non-streaming JSON).

    Tier gate + daily credit cap (canonical core.tiers LLM_FEATURE_CAPS["chat"]):
        Free  = 5 messages / day
        Pro   = 150
        Elite = 400

    Admins bypass the credit cap. Returns HTTP 402 with structured payload
    when the cap is hit so the frontend can show an upgrade CTA.

    The frontend's primary path is the streaming sibling below
    (``/copilot/chat/stream``); this endpoint stays as a back-compat +
    fallback surface (and is used by the embedded per-page agents).
    """
    _enforce_copilot_cap(user_tier)
    history_for_run = _load_copilot_history(body, user_tier.user_id)
    mem = load_memory(user_tier.user_id)

    try:
        result = await run_copilot(
            user_id=user_tier.user_id,
            message=body.message,
            route=body.route or "",
            history=history_for_run,
            mentioned_symbols=body.mentioned_symbols or [],
            memory=mem["summary"],
        )
    except Exception as e:
        logger.error("Copilot run failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Copilot failed")

    conversation_id = _persist_copilot_turns(
        user_id=user_tier.user_id,
        body=body,
        reply=result.get("reply", ""),
        tools_used=result.get("tools_used") or [],
        intent=result.get("intent"),
        refused=bool(result.get("refused", False)),
        trace=result.get("trace") or [],
    )

    _track_copilot_message(
        user_tier, body,
        intent=result.get("intent"),
        tools_used=result.get("tools_used") or [],
        refused=result.get("refused", False),
    )

    return CopilotChatResponse(**result, conversation_id=conversation_id)


@router.post("/copilot/chat/stream")
async def copilot_chat_stream(
    body: CopilotChatRequest,
    user_tier: UserTier = Depends(RequireFeature("copilot_chat")),
) -> StreamingResponse:
    """Token-streaming Copilot chat (Server-Sent Events).

    Same tier-gate, credit cap and persistence as ``/copilot/chat`` — but the
    reply streams in token-by-token, and structured chart/stat **artifacts**
    (built from real tool data) arrive in an early ``meta`` event so the UI can
    render charts before the prose.

    SSE frames (each a single ``data:`` line, newline-delimited)::

        data: {"type":"meta","tools_used":[...],"artifacts":[...],"intent":"..."}
        data: {"type":"token","text":"..."}            # repeated
        data: {"type":"done","reply":"<full>", ...}
        data: {"type":"saved","conversation_id":"..."}  # after persistence
        data: {"type":"error","message":"..."}          # on failure

    The cap-check runs *before* the stream opens so a 402 surfaces as a normal
    JSON error the frontend can turn into the upgrade CTA.
    """
    _enforce_copilot_cap(user_tier)
    history_for_run = _load_copilot_history(body, user_tier.user_id)
    mem = load_memory(user_tier.user_id)

    async def event_stream() -> AsyncIterator[str]:
        reply_text = ""
        intent: Optional[str] = None
        tools_used: List[str] = []
        refused = False
        try:
            async for ev in run_copilot_stream(
                user_id=user_tier.user_id,
                message=body.message,
                route=body.route or "",
                history=history_for_run,
                mentioned_symbols=body.mentioned_symbols or [],
                memory=mem["summary"],
            ):
                etype = ev.get("type")
                if etype == "meta":
                    tools_used = ev.get("tools_used") or tools_used
                    intent = ev.get("intent", intent)
                elif etype == "done":
                    reply_text = ev.get("reply", "") or reply_text
                    intent = ev.get("intent", intent)
                    tools_used = ev.get("tools_used") or tools_used
                    refused = bool(ev.get("refused", False))
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        except Exception as e:
            logger.error("Copilot stream failed: %s", e, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Copilot failed'})}\n\n"
            return

        # Persist after the full reply is known, then tell the client the id.
        conversation_id = _persist_copilot_turns(
            user_id=user_tier.user_id,
            body=body,
            reply=reply_text,
            tools_used=tools_used,
            intent=intent,
            refused=refused,
        )
        yield f"data: {json.dumps({'type': 'saved', 'conversation_id': conversation_id})}\n\n"

        _track_copilot_message(
            user_tier, body, intent=intent, tools_used=tools_used, refused=refused,
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable proxy buffering (nginx) so tokens flush immediately.
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# PR-BF.2 — Conversation history CRUD
# ============================================================================


@router.get("/copilot/conversations")
async def list_copilot_conversations(
    user_tier: UserTier = Depends(RequireFeature("copilot_chat")),
    limit: int = 30,
) -> Dict[str, Any]:
    """List the caller's saved conversations (most recent first).
    Used by the /copilot page sidebar."""
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("copilot_conversations")
            .select("id, title, created_at, updated_at")
            .eq("user_id", user_tier.user_id)
            .is_("archived_at", "null")
            .order("updated_at", desc=True)
            .limit(max(1, min(limit, 100)))
            .execute()
            .data
            or []
        )
        return {"conversations": rows, "count": len(rows)}
    except Exception as exc:
        logger.warning("list_copilot_conversations failed: %s", exc)
        return {"conversations": [], "count": 0}


@router.get("/copilot/conversations/{conversation_id}")
async def get_copilot_conversation(
    conversation_id: str,
    user_tier: UserTier = Depends(RequireFeature("copilot_chat")),
) -> Dict[str, Any]:
    """Full transcript for one conversation. The list endpoint stays
    light (title + timestamps); this one returns every message."""
    sb = get_supabase_admin()
    # Ownership check
    head = (
        sb.table("copilot_conversations")
        .select("id, title, created_at, updated_at, user_id")
        .eq("id", conversation_id)
        .single()
        .execute()
    )
    if not head.data or head.data.get("user_id") != user_tier.user_id:
        raise HTTPException(status_code=404, detail="conversation not found")

    msgs = (
        sb.table("copilot_messages")
        .select("id, role, content, tools_used, trace, intent, refused, created_at")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .limit(200)
        .execute()
        .data
        or []
    )
    return {
        "id": head.data["id"],
        "title": head.data.get("title"),
        "created_at": head.data["created_at"],
        "updated_at": head.data["updated_at"],
        "messages": msgs,
    }


@router.delete("/copilot/conversations/{conversation_id}")
async def archive_copilot_conversation(
    conversation_id: str,
    user_tier: UserTier = Depends(RequireFeature("copilot_chat")),
) -> Dict[str, Any]:
    """Soft-delete (sets archived_at). The conversation can be
    recovered later from the admin tool; users see it gone."""
    from datetime import datetime
    sb = get_supabase_admin()
    # Ownership check + soft delete in one UPDATE
    sb.table("copilot_conversations").update({
        "archived_at": datetime.utcnow().isoformat() + "Z",
    }).eq("id", conversation_id).eq("user_id", user_tier.user_id).execute()
    return {"ok": True}


# ============================================================================
# COPILOT ACTIONS — Cursor-style "agent" mode: propose reviewable actions
# ============================================================================


class CopilotActionsRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    route: Optional[str] = Field(None, description="Current frontend route for context")
    symbol: Optional[str] = Field(None, description="Symbol the page/composer is about")


class CopilotActionsResponse(BaseModel):
    actions: List[Dict[str, Any]] = Field(default_factory=list)


@router.post("/copilot/actions", response_model=CopilotActionsResponse)
async def copilot_actions(
    body: CopilotActionsRequest,
    user_tier: UserTier = Depends(RequireFeature("copilot_chat")),
):
    """Propose reviewable action cards (watchlist add/remove · run screen · order
    prep · strategy draft) from a natural-language request. PROPOSES ONLY — the
    client executes confirmed actions against the existing gated endpoints, so the
    real safety gates (global-halt → kill-switch → tier → broker → validation)
    stay the single authority. Returns [] for questions/analysis."""
    from ..ai.agents.copilot_actions import propose_actions

    actions = await propose_actions(
        message=body.message,
        route=body.route,
        symbol=body.symbol,
        user_id=user_tier.user_id,
    )
    return {"actions": actions}


# ============================================================================
# FINROBOT — F5/F7 Portfolio Doctor / AI SIP per-stock analysis
# ============================================================================


class FinRobotAnalyzeRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    fundamentals: Optional[Dict[str, Any]] = None
    concall_transcript: Optional[str] = None
    management_headlines: Optional[List[str]] = None
    promoter_holding: Optional[Dict[str, Any]] = None
    peers: Optional[List[Dict[str, Any]]] = None


class FinRobotAnalyzeResponse(BaseModel):
    symbol: str
    narrative: str
    action: str  # add | hold | trim | exit
    composite_score: int
    agents: Dict[str, Any]
    trace: List[Dict[str, Any]] = Field(default_factory=list)


@router.post("/finrobot/analyze", response_model=FinRobotAnalyzeResponse)
async def finrobot_analyze(
    body: FinRobotAnalyzeRequest,
    user: UserTier = Depends(RequireTier(Tier.PRO)),
) -> FinRobotAnalyzeResponse:
    """Run the 4-agent CoT graph on one stock (F5 / F7 backend).

    Pro+ tier — single-stock analysis is part of Portfolio Doctor Pro.
    Unlimited reruns are Elite (enforced client-side via feature map;
    backend refusal would require a counter, deferred).

    Caller supplies fundamentals + management data when available;
    missing fields produce neutral outputs rather than failing. The
    full data-fetching pipeline lands in PR 9-11.
    """
    # Fill missing fundamentals from screener.in (server-side) so single-stock
    # analysis grades real numbers even when the caller sends none.
    fundamentals = body.fundamentals
    promoter_holding = body.promoter_holding
    peers = body.peers
    if not fundamentals:
        try:
            import asyncio

            from ..data.fundamentals.screener_in import to_doctor_inputs
            di = await asyncio.to_thread(to_doctor_inputs, body.symbol.upper())
            fundamentals = di["fundamentals"]
            promoter_holding = promoter_holding or di["promoter_holding"]
            peers = peers or di["peers"]
        except Exception as fexc:
            logger.debug("fundamentals fetch failed %s: %s", body.symbol, fexc)

    try:
        result = await run_finrobot_doctor(
            user_id=user.user_id,
            symbol=body.symbol.upper(),
            fundamentals=fundamentals,
            concall_transcript=body.concall_transcript,
            management_headlines=body.management_headlines,
            promoter_holding=promoter_holding,
            peers=peers,
        )
    except Exception as e:
        logger.error("FinRobot run failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="FinRobot analysis failed")

    # PR 16 — analytics event.
    try:
        from ..observability import EventName, track
        track(EventName.FINROBOT_ANALYSIS_COMPLETED, user.user_id, {
            "tier": user.tier.value,
            "symbol": body.symbol.upper(),
            "action": result.get("action"),
            "composite_score": result.get("composite_score"),
        })
    except Exception:
        pass

    return FinRobotAnalyzeResponse(**result)


# ============================================================================
# TRADINGAGENTS — B1 Bull/Bear debate (Elite, high-stakes signals)
# ============================================================================


class DebateRequest(BaseModel):
    fundamentals: Optional[Dict[str, Any]] = None
    stock_snapshot: Optional[Dict[str, Any]] = None
    news_headlines: Optional[List[str]] = None
    regime: Optional[Dict[str, Any]] = None
    vix: Optional[float] = None


class DebateResponse(BaseModel):
    signal_id: Optional[str] = None
    symbol: Optional[str] = None
    decision: str  # enter | skip | half_size | wait
    confidence: int
    summary: str
    transcript: Dict[str, Any]
    trace: List[Dict[str, Any]] = Field(default_factory=list)


@router.post("/debate/signal/{signal_id}", response_model=DebateResponse)
async def trading_debate(
    signal_id: str,
    body: DebateRequest,
    user: UserTier = Depends(RequireFeature("debate")),
    _cap: UserTier = Depends(enforce_llm_cap("debate")),
) -> DebateResponse:
    """Run the 7-agent Bull/Bear debate on one signal.

    Elite-tier only per Step 1 §E5. Gated via ``RequireFeature("debate")``
    → returns HTTP 402 with structured payload for non-Elite callers.
    Admins bypass the gate.
    """
    # Fetch the signal row so analysts see the real trade parameters.
    client = get_supabase_admin()
    rows = client.table("signals").select("*").eq("id", signal_id).limit(1).execute()
    signal = (rows.data or [None])[0]
    if signal is None:
        raise HTTPException(status_code=404, detail=f"signal {signal_id} not found")

    # Per-(signal, day) cache — Elite gets the deep model (separate key).
    from datetime import datetime

    from ..core.config import settings
    from ..ai.agents.response_cache import cache_get, cache_set, seconds_to_ist_eod
    from ..data.market_calendar import IST
    deep = settings.LLM_DEEP_MODE_ENABLED and (user.tier == Tier.ELITE)
    ck = f"debate:{signal_id}:{datetime.now(IST).date().isoformat()}:{'deep' if deep else 'std'}"
    cached = cache_get(ck)
    if cached:
        return DebateResponse(**cached)

    try:
        result = await run_trading_debate(
            user_id=user.user_id,
            signal=signal,
            fundamentals=body.fundamentals,
            stock_snapshot=body.stock_snapshot,
            news_headlines=body.news_headlines,
            regime=body.regime,
            vix=body.vix,
            deep=deep,
            tier=user.tier.value,
        )
    except Exception as e:
        logger.error("TradingAgents debate failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Debate run failed")

    # Persist the debate into the PR 2 signal_debates table.
    try:
        import json as _json

        transcript = result.get("transcript") or {}
        client.table("signal_debates").upsert({
            "signal_id": signal_id,
            "bull_case": _json.dumps(transcript.get("bull") or {}),
            "bear_case": _json.dumps(transcript.get("bear") or {}),
            "risk_assessment": _json.dumps(transcript.get("risk") or {}),
            "trader_verdict": _json.dumps({
                "decision": result.get("decision"),
                "confidence": result.get("confidence"),
                "summary": result.get("summary"),
            }),
        }, on_conflict="signal_id").execute()
    except Exception as persist_err:
        logger.debug("signal_debates persist failed: %s", persist_err)

    # PR 13: emit DEBATE_COMPLETED so the signal detail page can flip
    # the debate tab from "running" → "ready" without polling.
    try:
        from ..platform.events import MessageType, emit_event
        await emit_event(
            MessageType.DEBATE_COMPLETED,
            {
                "signal_id": signal_id,
                "symbol": result.get("symbol"),
                "decision": result.get("decision"),
                "confidence": result.get("confidence"),
                "summary": result.get("summary"),
            },
            user_id=user.user_id,
        )
    except Exception as emit_err:
        logger.debug("DEBATE_COMPLETED emit skipped: %s", emit_err)

    # PR 16 — analytics event.
    try:
        from ..observability import EventName, track
        track(EventName.DEBATE_COMPLETED, user.user_id, {
            "tier": user.tier.value,
            "signal_id": signal_id,
            "symbol": result.get("symbol"),
            "decision": result.get("decision"),
            "confidence": result.get("confidence"),
        })
    except Exception:
        pass

    resp = DebateResponse(**result)
    cache_set(ck, resp.dict(), ttl_seconds=seconds_to_ist_eod(), surface="debate",
              model=settings.LLM_DEEP_MODEL if deep else settings.LLM_STRONG_MODEL)
    return resp
