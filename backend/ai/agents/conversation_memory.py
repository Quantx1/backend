"""Per-user rolling conversation MEMORY for the Copilot.

A single durable `summary` per user (copilot_memory). Loaded once per turn and
injected into the Responder context. Refreshed by a FREE model and THROTTLED —
only every REFRESH_EVERY persisted turns — so a normal turn adds zero LLM work.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

REFRESH_EVERY = 6
_MAX_SUMMARY_CHARS = 1200
_MEMORY_ROLE = "memory_summarizer"

_SUMMARY_SYSTEM = (
    "You maintain a compact running memory of a trader's chat with an Indian-"
    "equities assistant. Merge the existing memory with the new turns into <=6 "
    "terse lines capturing DURABLE facts only: holdings/watchlist mentioned, "
    "stocks they keep asking about, their style (intraday/swing/F&O), risk "
    "appetite, standing preferences. Drop chit-chat and stale items. No "
    "preamble, no markdown headers. If nothing durable, return the existing "
    "memory unchanged."
)


def _sb():
    try:
        from ...api.app import get_supabase_admin
        return get_supabase_admin()
    except Exception:  # noqa: BLE001
        return None


def load_memory(user_id: str) -> Dict[str, Any]:
    if not user_id:
        return {"summary": "", "turns_summarized": 0}
    sb = _sb()
    if sb is None:
        return {"summary": "", "turns_summarized": 0}
    try:
        rows = (sb.table("copilot_memory").select("summary, turns_summarized")
                .eq("user_id", user_id).limit(1).execute().data or [])
        if rows:
            r = rows[0]
            return {"summary": (r.get("summary") or "")[:_MAX_SUMMARY_CHARS],
                    "turns_summarized": int(r.get("turns_summarized") or 0)}
    except Exception as exc:  # noqa: BLE001
        logger.debug("copilot_memory load failed: %s", exc)
    return {"summary": "", "turns_summarized": 0}


def maybe_refresh_memory(*, user_id: str, total_turns: int,
                         recent_turns: List[Dict[str, str]], prev_summary: str,
                         prev_turns_summarized: int) -> None:
    """Throttled free-model refresh. No-op unless >=REFRESH_EVERY new turns."""
    if not user_id or total_turns - prev_turns_summarized < REFRESH_EVERY:
        return
    try:
        from .llm import complete_sync
        prompt = (f"Existing memory:\n{prev_summary or '(none)'}\n\nNew turns (JSON):\n"
                  f"{json.dumps(recent_turns[-(2 * REFRESH_EVERY):], ensure_ascii=False)}\n\n"
                  "Return the updated memory.")
        new_summary = (complete_sync(prompt, role=_MEMORY_ROLE, system=_SUMMARY_SYSTEM,
                       temperature=0.2, feature="copilot_memory", user_id=user_id) or "").strip()
        new_summary = new_summary[:_MAX_SUMMARY_CHARS]
        if not new_summary:
            return
        sb = _sb()
        if sb is None:
            return
        sb.table("copilot_memory").upsert({
            "user_id": user_id, "summary": new_summary,
            "turns_summarized": total_turns,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="user_id").execute()
    except Exception as exc:  # noqa: BLE001
        logger.debug("copilot_memory refresh skipped: %s", exc)
