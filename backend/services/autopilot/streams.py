"""Per-stream AutoPilot toggles — PR-AS.

Each user can independently enable/disable AutoPilot for these streams
and allocate a % of their capital to each. Sum across enabled streams
must not exceed 100%.

  swing      — TFT swing signals (PROD)
  momentum   — Qlib momentum picks
  portfolio  — Long-term AI portfolio (monthly rebalance)
  options    — F&O rule-based strategies (Elite)
  user_strategy — Per-strategy toggles for user-authored DSL strategies

The trade execution layer (AutoPilotService) reads these toggles per-user
before emitting trades. A stream with enabled=False is skipped entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Built-in streams — fixed set so the API can validate stream names without
# a DB call. Per-user-strategy streams use a different shape (user_strategy_id set).
BUILTIN_STREAMS: tuple[str, ...] = (
    "swing",
    "momentum",
    "portfolio",
    "options",
)

# Streams whose backing model is NOT PROD yet — toggling enabled has no
# operational effect, but we accept it so users can "pre-toggle" for the day
# a future engine reaches PROD. (Currently empty: the intraday LSTM stream
# was removed when the model was dropped from v1.)
NOT_YET_PROD_STREAMS: tuple[str, ...] = ()


@dataclass
class StreamState:
    stream: str
    user_strategy_id: Optional[str]
    enabled: bool
    allocated_capital_pct: float
    is_prod: bool         # False for streams without a PROD model behind them
    last_enabled_at: Optional[str]
    last_disabled_at: Optional[str]


def is_builtin(stream: str) -> bool:
    return stream in BUILTIN_STREAMS


def is_prod_stream(stream: str) -> bool:
    return stream not in NOT_YET_PROD_STREAMS


# ─────────────────────────────────────────────────────────────────────
# CRUD helpers — pure functions; caller passes the supabase client
# ─────────────────────────────────────────────────────────────────────


def list_streams_for_user(supabase: Any, user_id: str) -> List[StreamState]:
    """Return one StreamState per built-in stream + one per user-strategy
    stream the user has touched. Rows that don't exist yet are filled with
    defaults (disabled, 0% allocation) so the frontend always gets a
    consistent shape."""
    rows = (
        supabase.table("user_autopilot_streams")
        .select("stream, user_strategy_id, enabled, allocated_capital_pct, "
                "last_enabled_at, last_disabled_at")
        .eq("user_id", user_id)
        .limit(200)
        .execute()
    )
    by_key: Dict[tuple, Dict[str, Any]] = {
        (r["stream"], r.get("user_strategy_id")): r for r in (rows.data or [])
    }
    out: List[StreamState] = []
    # Always emit a row for every built-in stream
    for stream in BUILTIN_STREAMS:
        r = by_key.get((stream, None), {})
        out.append(StreamState(
            stream=stream,
            user_strategy_id=None,
            enabled=bool(r.get("enabled", False)),
            allocated_capital_pct=float(r.get("allocated_capital_pct") or 0),
            is_prod=is_prod_stream(stream),
            last_enabled_at=r.get("last_enabled_at"),
            last_disabled_at=r.get("last_disabled_at"),
        ))
    # Plus any per-user-strategy streams the user has touched
    for (stream, sid), r in by_key.items():
        if stream == "user_strategy" and sid is not None:
            out.append(StreamState(
                stream="user_strategy",
                user_strategy_id=sid,
                enabled=bool(r.get("enabled", False)),
                allocated_capital_pct=float(r.get("allocated_capital_pct") or 0),
                is_prod=True,  # user strategies are always allowed
                last_enabled_at=r.get("last_enabled_at"),
                last_disabled_at=r.get("last_disabled_at"),
            ))
    return out


def upsert_stream(
    supabase: Any,
    *,
    user_id: str,
    stream: str,
    user_strategy_id: Optional[str],
    enabled: bool,
    allocated_capital_pct: float,
) -> StreamState:
    """Toggle a stream + set its allocation. Validates allocation bounds
    and the cross-stream sum constraint."""

    if stream == "user_strategy" and user_strategy_id is None:
        raise ValueError("user_strategy stream requires a user_strategy_id")
    if stream != "user_strategy" and not is_builtin(stream):
        raise ValueError(f"unknown stream: {stream}")
    if not (0 <= allocated_capital_pct <= 100):
        raise ValueError(
            f"allocated_capital_pct must be in [0, 100], got {allocated_capital_pct}"
        )

    # Enforce: sum of ENABLED stream allocations must not exceed 100.
    # Compute current sum across the user's other enabled streams and add
    # this update's allocation (if it will be enabled).
    existing = list_streams_for_user(supabase, user_id)
    other_enabled_sum = 0.0
    for s in existing:
        same = (s.stream == stream and s.user_strategy_id == user_strategy_id)
        if same:
            continue
        if s.enabled:
            other_enabled_sum += s.allocated_capital_pct
    incoming = allocated_capital_pct if enabled else 0.0
    if other_enabled_sum + incoming > 100.01:  # 0.01 slack for float rounding
        raise ValueError(
            f"Cannot enable: enabled streams would total "
            f"{other_enabled_sum + incoming:.2f}% > 100%. "
            f"Reduce allocations on other streams first."
        )

    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "user_id": user_id,
        "stream": stream,
        "user_strategy_id": user_strategy_id,
        "enabled": enabled,
        "allocated_capital_pct": allocated_capital_pct,
        "updated_at": now,
    }
    # Track audit timestamps
    if enabled:
        payload["last_enabled_at"] = now
    else:
        payload["last_disabled_at"] = now

    supabase.table("user_autopilot_streams").upsert(
        payload,
        on_conflict="user_id,stream,user_strategy_id",
    ).execute()

    return StreamState(
        stream=stream,
        user_strategy_id=user_strategy_id,
        enabled=enabled,
        allocated_capital_pct=allocated_capital_pct,
        is_prod=is_prod_stream(stream) if stream != "user_strategy" else True,
        last_enabled_at=now if enabled else None,
        last_disabled_at=None if enabled else now,
    )


def total_allocated_pct(states: List[StreamState]) -> float:
    """Sum of allocated_capital_pct across currently-enabled streams."""
    return round(sum(s.allocated_capital_pct for s in states if s.enabled), 2)


def is_stream_enabled(
    supabase: Any,
    *,
    user_id: str,
    stream: str,
    user_strategy_id: Optional[str] = None,
) -> bool:
    """Cheap one-row lookup used by AutoPilotService during rebalance.

    Defaults to ``False`` if the row doesn't exist — opt-in, not opt-out.
    """
    try:
        q = (
            supabase.table("user_autopilot_streams")
            .select("enabled")
            .eq("user_id", user_id)
            .eq("stream", stream)
        )
        if user_strategy_id is not None:
            q = q.eq("user_strategy_id", user_strategy_id)
        rows = q.limit(1).execute()
    except Exception:
        return False
    if not (rows.data or []):
        return False
    return bool(rows.data[0].get("enabled", False))


__all__ = [
    "BUILTIN_STREAMS",
    "NOT_YET_PROD_STREAMS",
    "StreamState",
    "is_builtin",
    "is_prod_stream",
    "is_stream_enabled",
    "list_streams_for_user",
    "total_allocated_pct",
    "upsert_stream",
]
