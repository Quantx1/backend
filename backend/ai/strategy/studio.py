"""
Studio agent — PR-E per v2 design spec §6.2.

Compiles a natural-language strategy description into a validated
Strategy DSL document via an open model on the OpenRouter gateway in
JSON mode. The DSL Pydantic schema (in ``dsl.py``) is the safety
boundary — Studio is allowed to emit anything, but anything invalid is
caught immediately by ``Strategy.model_validate`` and either retried
(once) with the error context, or returned as a 422 to the caller.

Why an open model + JSON mode (not the custom GraphRunner):
  - This is a single-shot text→JSON transformation, not a multi-turn
    agent with tool calls. The GraphRunner is overkill for one call.
  - The gateway's JSON-response-mode forces structured output natively
    so we don't have to parse markdown fences.
  - Latency target: <3s per compile.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from pydantic import ValidationError

from .dsl import (
    INDICATOR_REGISTRY,
    Strategy,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Prompt — built once at import time, never templated dynamically
# ─────────────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """You are the Quant X Strategy Studio compiler.

Your ONLY job: convert a user's natural-language description of a trading strategy into a JSON document that matches the Quant X Strategy DSL schema.

You must NEVER:
- Output prose, explanations, or markdown — only the JSON object.
- Invent indicator names. Use only the names from the indicator registry below.
- Invent engine names. Use only: Alpha, Mood, Regime.
- Set mode to "live" — always emit mode="backtest" so the user can review.
- Recommend a strategy for the user that loses money — refuse via a stub strategy if the request is nonsensical.

You MUST:
- Always emit a stop_loss_pct on any intraday timeframe (1m, 5m, 15m, 30m, 1h).
- Always emit a position_size object with kind ∈ {percent_of_capital, fixed_qty, risk_based}.
- Match the schema exactly — field names, casing, enum values.
- PREFER regime-free strategies (pure price/indicator logic). Only set
  regime_filter (or an engine_signal: Regime condition) when the user
  EXPLICITLY asks for regime-aware behaviour — regime-gated strategies need
  real historical regime data to backtest reliably, so default to leaving it out.

# Schema
The output JSON must be a Strategy document with these fields:

  name             string (1-120 chars)            — describe what the strategy does
  symbol           string or null                  — required only if universe="single"
  universe         enum (see below)
  timeframe        enum: 1m, 5m, 15m, 30m, 1h, 4h, 1d  (any is valid; pick what the user asks for)
  entry            Condition (see below)
  exit             Condition (see below)
  stop_loss_pct    number > 0 < 100 or null         — REQUIRED if timeframe is intraday
  take_profit_pct  number > 0 < 1000 or null
  trailing_stop_pct number > 0 < 100 or null
  square_off_time  string "HH:MM" (IST) or null      — INTRADAY ONLY; set this when
                     the user asks to auto square-off / exit / flatten at a clock
                     time (e.g. "square off at 15:09" → "15:09"). Do NOT encode a
                     square-off as an exit Condition — use this field.
  position_size    {"kind": <PositionSizeKind>, "value": <number>}
  regime_filter    enum: bull_only, bear_only, sideways_only, any
  lookback_days    integer 10..730  (default 90)
  mode             enum: backtest, paper, live  (USE "backtest")

Universe enum:
  single, nifty50, nifty100, nifty500,
  sector:IT, sector:BANK, sector:AUTO, sector:PHARMA, sector:FMCG,
  sector:METAL, sector:ENERGY, sector:INFRA

PositionSizeKind enum (with constraints):
  percent_of_capital  value in (0, 100]   — default reasonable values 2-10
  fixed_qty           value > 0
  risk_based          value in (0, 5]     — % of capital at risk per trade

# Condition shape (recursive)
A Condition is one of:

1. indicator_compare      compare an indicator value to a constant or another indicator
   {"kind": "indicator_compare", "indicator": "<reg>", "op": "<op>", "value": <number|str|[lo,hi]>}

2. indicator_cross        two indicator series crossing
   {"kind": "indicator_cross", "indicator": "<reg>", "op": "crosses_above|crosses_below", "value": "<reg>"}

3. engine_signal          ML engine output
   {"kind": "engine_signal", "engine": "Alpha|Mood|Regime",
    "op": "<op>", "value": <number|str>}

4. composite_and          ALL children must be true
   {"kind": "composite_and", "children": [Condition, Condition, ...]}

5. composite_or           ANY child true
   {"kind": "composite_or", "children": [Condition, Condition, ...]}

Operators:
  <, >, <=, >=, ==, !=, crosses_above, crosses_below, between, outside
  (between/outside require value=[lo, hi])
  (crosses_above/crosses_below require value to be an indicator name string)

# Engine signal values (when condition.kind = engine_signal):
  Regime     → "bull" | "sideways" | "bear"
  Alpha      → numeric (lower = stronger; e.g. rank ≤ 10 means top 10)
  Mood       → numeric in [-1, 1]  (sentiment score)

# Indicator registry (closed set — DO NOT INVENT NAMES)
{indicators}

# Examples

User: "RSI mean reversion on Nifty 50"
Output:
{{
  "name": "RSI Mean Reversion",
  "universe": "nifty50",
  "timeframe": "1d",
  "entry": {{"kind": "indicator_compare", "indicator": "rsi14", "op": "<", "value": 30}},
  "exit":  {{"kind": "indicator_compare", "indicator": "rsi14", "op": ">", "value": 70}},
  "stop_loss_pct": 4,
  "take_profit_pct": 8,
  "position_size": {{"kind": "percent_of_capital", "value": 5}},
  "regime_filter": "any",
  "lookback_days": 180,
  "mode": "backtest"
}}

User: "Bullish regime + EMA crossover on top-100, 5% trailing stop"
Output:
{{
  "name": "Regime + EMA Cross",
  "universe": "nifty100",
  "timeframe": "1d",
  "entry": {{
    "kind": "composite_and",
    "children": [
      {{"kind": "engine_signal", "engine": "Regime", "op": "==", "value": "bull"}},
      {{"kind": "indicator_cross", "indicator": "ema8", "op": "crosses_above", "value": "ema21"}}
    ]
  }},
  "exit": {{"kind": "indicator_cross", "indicator": "ema8", "op": "crosses_below", "value": "ema21"}},
  "trailing_stop_pct": 5,
  "stop_loss_pct": 4,
  "position_size": {{"kind": "percent_of_capital", "value": 4}},
  "regime_filter": "bull_only",
  "lookback_days": 180,
  "mode": "backtest"
}}

# If the request is too vague to compile safely — no instrument named, or neither an entry nor an exit rule — do NOT guess. Instead return ONLY this small object:
#   {"needs_clarification": true, "missing": ["instrument"|"entry"|"exit"|"risk", ...], "question": "<one short question naming what you need>", "assumptions": ["<safe default you'd otherwise apply>", ...]}
# Otherwise, return the Strategy JSON as described.

# Now compile the user's request below into ONE JSON object only.
"""

_PROMPT = _SYSTEM_PROMPT.replace("{indicators}", ", ".join(INDICATOR_REGISTRY))


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# Clarification pre-gate — zero-token deterministic slot detection
# ─────────────────────────────────────────────────────────────────────
# Mirrors the copilot regex fast-path: a cheap deterministic check BEFORE any
# generator call. Fires ONLY for obviously under-specified prompts (no
# instrument, or neither an entry nor an exit trigger). Nuanced gaps are left
# to the generator-fold so we never over-block a real strategy description.

_INSTRUMENT_RE = re.compile(
    r"\b(nifty\s*50|nifty\s*100|nifty\s*500|nifty|bank\s*nifty|banknifty|finnifty|"
    r"sensex|midcap|smallcap|stocks?|shares?|equit(?:y|ies)|indices|index|"
    r"sector\s*:?\s*\w+|top\s*\d+)\b",
    re.IGNORECASE,
)
_TICKER_RE = re.compile(r"\b[A-Z][A-Z&]{2,}\b")  # ALLCAPS ticker-ish, >=3 chars

_ENTRY_RE = re.compile(
    r"\b(buy|enter|long|cross(?:es)?|above|below|breakout|break\s*out|oversold|"
    r"overbought|rsi|ema|sma|macd|vwap|bollinger|momentum|dip|pullback|gap\s*up|"
    r"support|when)\b|[<>]",
    re.IGNORECASE,
)
_EXIT_RE = re.compile(
    r"\b(sell|exit|close|square\s*off|stop[\s-]?loss|stop|target|take[\s-]?profit|"
    r"trailing|book\s*profit|sl|resistance)\b",
    re.IGNORECASE,
)
_RISK_RE = re.compile(
    r"\b(stop[\s-]?loss|stop|sl|risk|max\s*loss|drawdown|percent)\b|%",
    re.IGNORECASE,
)


def _detect_underspec(prompt: str) -> List[str]:
    """Which required slots (instrument/entry/exit/risk) are absent from a raw
    NL prompt. Purely lexical — no generator. Slot detection is deliberately
    generous (favours 'present') so we only ever flag truly bare prompts."""
    text = prompt or ""
    missing: List[str] = []
    if not (_INSTRUMENT_RE.search(text) or _TICKER_RE.search(text)):
        missing.append("instrument")
    if not _ENTRY_RE.search(text):
        missing.append("entry")
    if not _EXIT_RE.search(text):
        missing.append("exit")
    if not _RISK_RE.search(text):
        missing.append("risk")
    return missing


_SLOT_PHRASE = {
    "instrument": "which stock or index to trade (a single symbol like RELIANCE, or a universe like Nifty 50)",
    "entry": "what should trigger an entry (e.g. a moving-average cross, an RSI level, a breakout)",
    "exit": "when the trade should close (e.g. a reverse signal, a profit target, a trailing stop)",
    "risk": "how much to risk per trade (e.g. a stop-loss percent)",
}

_DEFAULT_ASSUMPTIONS = [
    "Daily (1d) timeframe",
    "5% of capital per position",
    "Backtest mode so you can review before going live",
]


@dataclass
class ClarificationNeeded:
    """Returned by compile_strategy when a prompt is too under-specified to
    compile safely. Carries one follow-up question, the missing slots (for UI
    chips), the safe defaults we'd otherwise assume, and the original prompt so
    the caller can append the answer and re-compile."""
    missing: List[str]
    question: str
    assumptions: List[str]
    prompt: str


def _clarify_question(missing: List[str]) -> str:
    core = [m for m in missing if m in ("instrument", "entry", "exit")] or missing
    parts = [_SLOT_PHRASE[m] for m in core if m in _SLOT_PHRASE]
    if not parts:
        joined = "a few more details"
    elif len(parts) == 1:
        joined = parts[0]
    else:
        joined = "; and ".join(parts)
    return f"To build this well, tell me {joined}."


def precheck_clarification(prompt: str) -> Optional[ClarificationNeeded]:
    """Zero-token deterministic pre-gate. Returns a ClarificationNeeded ONLY for
    obviously bare prompts (no instrument, or neither entry nor exit). Returns
    None for anything with enough signal — those proceed to the generator."""
    if not prompt or not prompt.strip():
        return None  # empty is handled by compile_strategy's StudioError guard
    missing = _detect_underspec(prompt)
    bare = ("instrument" in missing) or ("entry" in missing and "exit" in missing)
    if not bare:
        return None
    return ClarificationNeeded(
        missing=missing,
        question=_clarify_question(missing),
        assumptions=list(_DEFAULT_ASSUMPTIONS),
        prompt=prompt.strip(),
    )


# ─────────────────────────────────────────────────────────────────────
# Vision → prompt synthesizer (PURE, deterministic — no generator call)
# ─────────────────────────────────────────────────────────────────────
# Turns a structured chart read into a plain-English strategy prompt that the
# EXISTING compiler can consume. Long-only (the DSL has no short side): bearish
# / no-edge reads return None so we never fabricate a losing long. Absolute
# rupee S/R levels are intentionally dropped — the DSL expresses exits as
# percentages / indicator logic, so we steer toward indicator-based rules that
# APPROXIMATE the read rather than replicate exact levels. Brand firewall: this
# is OUR strategy DSL — never reference external charting tools or model names.

def synthesize_prompt_from_vision(analysis, *, symbol: str, timeframe: str = "1d") -> Optional[str]:
    """Build a natural-language strategy prompt from a VisionAnalysis. Returns
    None for bearish / no-edge / unavailable reads (caller shows an honest note)."""
    if analysis is None or not getattr(analysis, "available", False):
        return None

    sym = (symbol or "").strip().upper().replace(".NS", "")
    if not sym:
        return None

    tf = (timeframe or "1d").strip() or "1d"
    setup = (getattr(analysis, "setup", None) or "").lower()
    trend = (getattr(analysis, "trend", None) or "").lower()

    bullish = ("bullish" in setup) or (trend == "uptrend")
    rangebound = ("range" in setup) or (trend == "range")
    bearish = ("bearish" in setup) or (trend == "downtrend") or ("no edge" in setup)

    # Refuse bearish / no-edge — the DSL is long-only.
    if bearish and not bullish:
        return None

    if bullish:
        return (
            f"On {sym}, {tf} timeframe: enter long when the 8 EMA crosses above "
            f"the 21 EMA and RSI is between 50 and 70 (momentum confirmation of "
            f"the uptrend). Exit when the 8 EMA crosses back below the 21 EMA, "
            f"take profit at +8%, and set a 4% stop loss."
        )
    if rangebound:
        return (
            f"On {sym}, {tf} timeframe: mean-reversion long — buy when RSI drops "
            f"below 30 near the lower edge of the range. Exit when RSI rises above "
            f"65 or take profit at +5%, with a 3% stop loss."
        )

    # Unclear but not clearly bearish → a conservative pullback long.
    return (
        f"On {sym}, {tf} timeframe: enter long when price reclaims the 21 EMA and "
        f"RSI crosses back above 50. Exit when RSI reaches 70 or take profit at "
        f"+6%, with a 4% stop loss."
    )


class StudioError(RuntimeError):
    """Raised when Studio cannot produce valid DSL after retry."""


def is_studio_available() -> bool:
    """True when the generator backend (OpenRouter) is configured."""
    return _resolve_generator() is not None


def _resolve_generator() -> Optional[str]:
    """The open model slug for the strategy generator, or None when OpenRouter
    isn't configured. Uses AGENT_MODEL_MAP['strategy_generator'], else the
    default open model."""
    try:
        from ...core.config import settings
        from ..agents.llm import agent_model_map
        if not settings.OPENROUTER_API_KEY:
            return None
        return agent_model_map().get("strategy_generator") or settings.LLM_DEFAULT_MODEL
    except Exception:  # noqa: BLE001
        return None


def _call_openrouter_json(prompt: str, model: str) -> str:
    """Sync OpenRouter JSON call for the generator. Budget-gated (paid only)
    + free→free→paid fallback chain + usage recording. Returns raw JSON text."""
    import httpx

    from ...core.config import settings
    from ...observability.llm_budget import get_meter
    from ...observability.llm_pricing import is_paid, micros_usd_for
    from ..agents.llm import build_models

    meter = get_meter()
    over = meter.over_budget(settings.LLM_MONTHLY_BUDGET_USD)
    if is_paid("openrouter", model) and over:
        raise StudioError("Monthly AI budget reached — strategy generation is paused.")

    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
    }
    models = build_models(model, allow_paid=not over)
    if models:
        payload["models"] = models
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://quantx.app",
        "X-Title": "Quant X",
    }
    try:
        r = httpx.post(f"{settings.LLM_GATEWAY_BASE_URL}/chat/completions",
                       headers=headers, json=payload, timeout=90.0)
    except Exception as exc:  # noqa: BLE001
        raise StudioError(f"generator call failed: {exc}") from exc
    if r.status_code >= 400:
        raise StudioError(f"generator call failed ({r.status_code}): {r.text[:200]}")

    data = r.json()
    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if not text:
        raise StudioError("generator returned empty response")

    # Best-effort usage recording (cost meter + admin panel).
    used = data.get("model") or model
    usage = data.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens", 0) or 0)
    out_tok = int(usage.get("completion_tokens", 0) or 0)
    try:
        from ...observability import track_llm_usage
        track_llm_usage(user_id=None, feature="studio", provider="openrouter",
                        model=used, input_tokens=in_tok, output_tokens=out_tok)
        meter.record_micros(micros_usd_for("openrouter", used, in_tok, out_tok))
    except Exception:  # noqa: BLE001
        pass
    return text.strip()


def _generate_dsl(prompt: str) -> str:
    """Generate raw DSL JSON via the open generator model (OpenRouter)."""
    gen = _resolve_generator()
    if not gen:
        raise StudioError("Strategy generator is not configured (OPENROUTER_API_KEY unset).")
    return _call_openrouter_json(prompt, gen)


def _coerce_condition_value(v):
    """Flatten the dict the model sometimes emits for a Condition.value into the
    scalar/string the DSL expects (Union[float,int,str,List[float]]). Common
    malformed shapes: {"indicator":"ema21"} -> "ema21", {"value":70} -> 70,
    a single-key dict -> its sole scalar. Anything else is left untouched so the
    validation/retry path still surfaces a real error."""
    if isinstance(v, dict):
        for k in ("indicator", "name", "symbol", "value", "level", "threshold"):
            if k in v and not isinstance(v[k], (dict, list)):
                return v[k]
        if len(v) == 1:
            sole = next(iter(v.values()))
            if not isinstance(sole, (dict, list)):
                return sole
    return v


def _normalize_conditions(node):
    """Recursively repair malformed Condition.value fields in a DSL payload
    (entry/exit and nested composite children) before Pydantic validation."""
    if isinstance(node, dict):
        if "value" in node:
            node["value"] = _coerce_condition_value(node["value"])
        for child in (node.get("children") or []):
            _normalize_conditions(child)
    return node


def compile_strategy(
    user_prompt: str,
    *,
    max_retries: int = 2,
) -> "Strategy | ClarificationNeeded":
    """Compile a natural-language prompt into a validated Strategy.

    If the first LLM response fails Pydantic validation, we retry once
    with the validation errors appended to the prompt so the model can
    self-correct. After ``max_retries``, raises StudioError with all
    accumulated failures.

    Returns the validated Strategy (force mode='backtest' regardless of
    what the LLM emitted — never let Studio compile directly to live).
    """
    if not user_prompt or not user_prompt.strip():
        raise StudioError("user_prompt is empty")

    # Zero-token deterministic pre-gate — a bare prompt short-circuits WITHOUT
    # calling the generator (and is never cached).
    pre = precheck_clarification(user_prompt)
    if pre is not None:
        return pre

    # Result cache: this is a deterministic NL→DSL transform (depends only on the
    # normalized prompt + the static compiler prompt — no regime/vix/news/symbol),
    # so an identical prompt yields an identical validated Strategy. Cache only the
    # successful validated result (never a StudioError). Short-ish TTL for a create
    # action so generator/registry improvements surface promptly.
    from ..agents.response_cache import cache_get, cache_set
    norm_prompt = " ".join(user_prompt.strip().lower().split())
    ck = f"studio:compile:{norm_prompt}"
    cached = cache_get(ck)
    if cached is not None:
        try:
            return Strategy.model_validate(cached)
        except ValidationError:
            pass  # stale/incompatible cache shape — recompile

    full_prompt = f"{_PROMPT}\n\nUser request: {user_prompt.strip()}"
    errors: List[str] = []

    for attempt in range(max_retries + 1):
        raw = _generate_dsl(full_prompt)

        # Best-effort cleanup — strip surrounding code fences if model
        # ignored the JSON-mode instruction.
        cleaned = raw
        if cleaned.startswith("```"):
            # Strip ``` and possible ```json prefix
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            errors.append(f"attempt {attempt + 1}: invalid JSON ({exc})")
            if attempt < max_retries:
                full_prompt += (
                    f"\n\nPREVIOUS ATTEMPT FAILED JSON PARSE: {exc}. "
                    "Re-emit as valid JSON only, no prose."
                )
                continue
            raise StudioError("; ".join(errors))

        # Generator-fold: the model may itself ask for clarification. A token
        # was already spent, so the route's credit is correctly consumed.
        if isinstance(payload, dict) and payload.get("needs_clarification"):
            miss = payload.get("missing")
            ques = payload.get("question")
            assp = payload.get("assumptions")
            if isinstance(miss, list) and isinstance(ques, str) and ques.strip():
                return ClarificationNeeded(
                    missing=[str(m) for m in miss],
                    question=ques.strip(),
                    assumptions=[str(a) for a in (assp if isinstance(assp, list) else [])],
                    prompt=user_prompt.strip(),
                )
            # malformed clarify object → fall through to normal validate/retry

        # Force-coerce mode to 'backtest' — never let Studio go straight to live
        if isinstance(payload, dict):
            payload["mode"] = "backtest"
            # Repair the common dict-as-value emission before validation so the
            # retry loop isn't wasted on a structural error the model won't fix.
            for _k in ("entry", "exit"):
                if isinstance(payload.get(_k), dict):
                    _normalize_conditions(payload[_k])

        try:
            strategy = Strategy.model_validate(payload)
            cache_set(ck, strategy.model_dump(mode="json"), ttl_seconds=3600,
                      surface="studio_compile", model="")
            return strategy
        except ValidationError as exc:
            err_summary = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            )
            errors.append(f"attempt {attempt + 1}: DSL validation failed — {err_summary}")
            if attempt < max_retries:
                full_prompt += (
                    f"\n\nPREVIOUS DSL FAILED VALIDATION: {err_summary}\n"
                    "Re-emit the SAME strategy but fix exactly these errors."
                )
                continue
            raise StudioError("; ".join(errors))

    # Defensive — should be unreachable
    raise StudioError("; ".join(errors) or "studio compile exhausted retries")


__all__ = [
    "ClarificationNeeded",
    "StudioError",
    "compile_strategy",
    "is_studio_available",
    "precheck_clarification",
    "synthesize_prompt_from_vision",
]
