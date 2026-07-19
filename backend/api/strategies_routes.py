"""
Strategy DSL + CRUD routes — PR-D (validate/discovery) + PR-F (registry).

PR-D endpoints (validate + discovery):
    POST /api/strategies/validate
    GET  /api/strategies/indicators
    GET  /api/strategies/engines
    GET  /api/strategies/operators
    GET  /api/strategies/enums

PR-F endpoints (CRUD + state machine):
    POST   /api/strategies           — create draft strategy
    GET    /api/strategies           — list own strategies
    GET    /api/strategies/{id}      — get one
    PATCH  /api/strategies/{id}/dsl  — edit DSL (draft|paused only)
    POST   /api/strategies/{id}/transition — state change (backtest/paper/live/paused/archived)
    DELETE /api/strategies/{id}      — convenience for transition→archived
    GET    /api/strategies/{id}/executions — recent strategy_executions rows

PR-H endpoints (public catalog + template cloning):
    GET    /api/strategies/catalog           — public listing of templates
    GET    /api/strategies/catalog/{slug}    — public template with full DSL
    POST   /api/strategies/from-template/{slug} — clone template into user draft

Auth: most endpoints require a logged-in user. The catalog endpoints
are public so the pre-signup discovery page works. Tier gating for
`live` deploy is enforced in transition_strategy() and per-template
tier requirements are enforced in clone_strategy_from_template().
"""

from __future__ import annotations
from ..core.security import get_current_user
from ..core.tiers import UserTier
from ..middleware.llm_caps import consume_llm_cap_or_raise
from ..middleware.tier_gate import RequireFeature, current_user_tier   # (current_user_tier is already imported)
from ..ai.strategy import registry as strat_registry
from ..ai.strategy.dsl import (
    ConditionKind,
    EngineName,
    INDICATOR_REGISTRY,
    Operator,
    Strategy,
    StrategyMode,
    Timeframe,
    Universe,
)
from ..core.database import get_supabase_admin

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError


# ── catalog cache ─────────────────────────────────────────────────────
# Catalog rows are admin-seeded and rarely change. Caching them in-process
# for ~60s drops /catalog and /catalog/sections from ~1100ms (Supabase
# round-trip + JSON encode) to <2ms on cache hit. Keyed on the full
# filter signature so each filter combo gets its own entry.
_CATALOG_TTL = 60.0
_catalog_cache: Dict[Tuple[Any, ...], Tuple[float, Any]] = {}


def _catalog_cached(key: Tuple[Any, ...], producer):
    now = time.monotonic()
    hit = _catalog_cache.get(key)
    if hit and now - hit[0] < _CATALOG_TTL:
        return hit[1]
    value = producer()
    _catalog_cache[key] = (now, value)
    # Bound the cache so a hostile caller can't blow memory by spraying
    # filter combos. 256 unique keys is far more than any real UI.
    if len(_catalog_cache) > 256:
        oldest = min(_catalog_cache.items(), key=lambda kv: kv[1][0])[0]
        _catalog_cache.pop(oldest, None)
    return value


router = APIRouter(prefix="/api/strategies", tags=["Strategies"])


def _strategy_uses_regime(strategy) -> bool:
    """True if the strategy's regime_filter is non-'any' OR any condition
    in entry/exit references the Regime engine. Used to skip the
    regime_history query for strategies that wouldn't use it anyway."""
    if strategy.regime_filter.value != "any":
        return True

    def _walks_regime(cond) -> bool:
        if cond is None:
            return False
        if cond.kind.value == "engine_signal" and cond.engine is not None and cond.engine.value == "Regime":
            return True
        if cond.children:
            return any(_walks_regime(c) for c in cond.children)
        return False

    return _walks_regime(strategy.entry) or _walks_regime(strategy.exit)


def _maybe_load_engine_signals(sb, strategy, ohlcv):
    """Pre-load engine signals per bar for DSL backtest. Currently only
    populates ``regime`` (the other PROD engines — Alpha + Mood — need
    per-symbol historical model output tables that aren't reliably
    persisted yet, so we leave them None and conditions referencing
    them fail closed.)

    Returns ``{pd.Timestamp: EngineSignals}`` or ``None`` if the strategy
    doesn't use Regime.
    """
    if not _strategy_uses_regime(strategy):
        return None

    try:
        from ..services.regime import resolve_regime_history
        from ..ai.strategy.interpreter import EngineSignals
    except Exception:
        return None

    # Date range from the OHLCV index
    if len(ohlcv) == 0:
        return None
    first = ohlcv.index[0]
    last = ohlcv.index[-1]
    start_d = first.date() if hasattr(first, "date") else first
    end_d = last.date() if hasattr(last, "date") else last

    regime_map = resolve_regime_history(sb, start=start_d, end=end_d)
    out = {}
    for ts in ohlcv.index:
        d = ts.date() if hasattr(ts, "date") else ts
        out[ts] = EngineSignals(regime=regime_map.get(d, "sideways"))
    return out


def _date_of(ts):
    return ts.date() if hasattr(ts, "date") else ts


def _regime_coverage_range(sb, start_d, end_d) -> float:
    """Fraction of [start_d, end_d] covered by REAL detected regime (vs the
    pre-history sideways default). Used to fail-closed the gate on regime
    strategies backtested over windows the regime model didn't cover."""
    from ..services.regime.resolver import resolve_regime_history_with_coverage
    try:
        _m, cov = resolve_regime_history_with_coverage(sb, start=start_d, end=end_d)
        return cov
    except Exception:  # noqa: BLE001
        return 0.0


# Allowed tiers for live deployment. paper is free-tier-allowed.
_LIVE_DEPLOY_TIERS = {"pro", "elite"}


# ─────────────────────────────────────────────────────────────────────
# /validate
# ─────────────────────────────────────────────────────────────────────


@router.post("/validate")
async def validate_strategy(
    body: Dict[str, Any],
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Validate a Strategy DSL document.

    Returns the normalized DSL when valid (Pydantic may coerce types,
    e.g. strip ``symbol`` for non-single universes). When invalid,
    returns 422 with per-field errors so the frontend can mark exactly
    which condition or indicator is bad.
    """
    try:
        s = Strategy.model_validate(body)
    except ValidationError as exc:
        # Surface a flat error list — frontend renders these inline next
        # to the offending DSL field.
        errors = [
            {
                "loc": ".".join(str(p) for p in e["loc"]),
                "msg": e["msg"],
                "type": e["type"],
            }
            for e in exc.errors()
        ]
        raise HTTPException(status_code=422, detail={"valid": False, "errors": errors})
    return {
        "valid": True,
        "strategy": s.model_dump(mode="json"),
    }


# ─────────────────────────────────────────────────────────────────────
# Discovery — Studio UI uses these to populate dropdowns
# ─────────────────────────────────────────────────────────────────────


@router.get("/indicators")
async def list_indicators_endpoint() -> Dict[str, Any]:
    """Return the closed-set indicator registry, grouped for UI."""
    groups = {
        "momentum": [
            "rsi7", "rsi9", "rsi14",
            "stochastic_k", "stochastic_d",
            "williams_r", "mfi", "cci",
        ],
        "trend": [
            "ema5", "ema8", "ema13", "ema21", "ema50", "ema100", "ema200",
            "sma10", "sma20", "sma50", "sma100", "sma200",
            "macd", "macd_signal", "macd_hist",
            "adx", "supertrend", "psar",
        ],
        "volatility": ["atr", "bbands_upper", "bbands_middle", "bbands_lower"],
        "volume": ["vwap", "obv", "volume_sma20"],
        "price": [
            "close", "open", "high", "low",
            "prev_close", "prev_high", "prev_low",
        ],
        "patterns": [
            "pattern_doji", "pattern_hammer", "pattern_inverted_hammer",
            "pattern_bullish_engulfing", "pattern_bearish_engulfing",
            "pattern_morning_star", "pattern_evening_star",
            "pattern_bullish_harami", "pattern_bearish_harami",
            "pattern_three_white_soldiers", "pattern_three_black_crows",
        ],
    }
    return {
        "indicators": list(INDICATOR_REGISTRY),
        "groups": groups,
        "count": len(INDICATOR_REGISTRY),
    }


@router.get("/engines")
async def list_engines_endpoint() -> Dict[str, Any]:
    """Return the whitelist of engine names allowed in engine_signal."""
    return {
        "engines": [e.value for e in EngineName],
        "values_by_engine": {
            "Regime": ["bull", "sideways", "bear"],
            "Alpha": "numeric (rank, lower is stronger)",
            "Mood": "numeric (sentiment_5d_mean, -1 to 1)",
        },
    }


@router.get("/operators")
async def list_operators_endpoint() -> Dict[str, Any]:
    """Return operators grouped by which condition kinds accept them."""
    return {
        "indicator_compare": ["<", ">", "<=", ">=", "==", "!=", "between", "outside"],
        "indicator_cross": ["crosses_above", "crosses_below"],
        "engine_signal": ["<", ">", "<=", ">=", "==", "!="],
    }


@router.get("/enums")
async def list_enums_endpoint() -> Dict[str, Any]:
    """Return all DSL enum values (one-shot fetch for UI initialization)."""
    return {
        "timeframe": [t.value for t in Timeframe],
        "universe": [u.value for u in Universe],
        "mode": [m.value for m in StrategyMode],
        "condition_kind": [k.value for k in ConditionKind],
        "operator": [o.value for o in Operator],
        "engine": [e.value for e in EngineName],
    }


# ─────────────────────────────────────────────────────────────────────
# PR-H Public catalog + template cloning
#
# NOTE: these endpoints MUST be declared before the /{strategy_id}
# catch-all routes below, otherwise "/catalog" gets matched as a
# strategy_id UUID lookup and 404s.
# ─────────────────────────────────────────────────────────────────────


_CATALOG_FIELDS = (
    "id, slug, name, description, category, segment, tier_required, "
    "min_capital, risk_level, tags, icon, supported_symbols, "
    "is_featured, is_exclusive, exclusive_tagline, sort_order, "
    "requires_fo_enabled, engine_compatible, "
    "backtest_total_return, backtest_cagr, backtest_win_rate, "
    "backtest_sharpe, backtest_max_drawdown, backtest_total_trades"
)


@router.get("/catalog")
async def list_strategy_catalog(
    segment: Optional[str] = Query(default=None, pattern="^(EQUITY|OPTIONS|FUTURES)$"),
    category: Optional[str] = Query(default=None, max_length=64),
    tier: Optional[str] = Query(default=None, pattern="^(free|pro|elite)$"),
    featured_only: bool = Query(default=False),
    engine_only: bool = Query(default=False),
    exclusive_only: bool = Query(default=False, description=(
        "Return only templates flagged is_exclusive — drives the "
        "'Exclusive Strategies — Unlock advanced algorithms' section."
    )),
    max_min_capital: Optional[int] = Query(default=None, ge=1000, le=10_000_000,
                                           description="Return only templates whose min_capital is ≤ this value. "
                                           "Drives the 'What can I run with ₹X?' filter chips on /strategies."),
    limit: int = Query(default=200, ge=1, le=500),
) -> Dict[str, Any]:
    """Public listing of strategy templates from ``strategy_catalog``.

    No auth required — drives the /strategies discovery page pre-signup.
    Returns templates with their DSL stripped (use /catalog/{slug} for
    the full DSL) to keep the list payload small.
    """
    cache_key = (
        "catalog", segment, category, tier, featured_only, engine_only,
        exclusive_only, max_min_capital, limit,
    )

    def _produce() -> Dict[str, Any]:
        sb = get_supabase_admin()
        q = sb.table("strategy_catalog").select(_CATALOG_FIELDS).eq("is_active", True)
        if segment:
            q = q.eq("segment", segment)
        if category:
            q = q.eq("category", category)
        if tier:
            q = q.eq("tier_required", tier)
        if featured_only:
            q = q.eq("is_featured", True)
        if engine_only:
            q = q.eq("engine_compatible", True)
        if exclusive_only:
            q = q.eq("is_exclusive", True)
        if max_min_capital is not None:
            q = q.lte("min_capital", max_min_capital)
        rows = q.order("is_featured", desc=True).order("sort_order").limit(limit).execute()
        return {
            "templates": rows.data or [],
            "count": len(rows.data or []),
        }

    return _catalog_cached(cache_key, _produce)


@router.get("/catalog/sections")
async def list_catalog_sections() -> Dict[str, Any]:
    """One-shot endpoint that returns the catalog pre-grouped into the
    discovery-page sections the frontend renders. Saves the client from
    issuing 4-5 filtered calls.

    Sections:
      - exclusive — is_exclusive=True (drives "Unlock advanced algorithms")
      - featured  — is_featured=True
      - intraday  — category=equity_intraday
      - swing     — category=equity_swing
      - options   — segment=OPTIONS
    """
    def _produce() -> Dict[str, Any]:
        sb = get_supabase_admin()

        def _fetch(filter_field: str, filter_value: Any) -> List[Dict[str, Any]]:
            q = (
                sb.table("strategy_catalog")
                .select(_CATALOG_FIELDS)
                .eq("is_active", True)
                .eq(filter_field, filter_value)
                .order("sort_order")
                .limit(50)
            )
            return q.execute().data or []

        sections = {
            "exclusive": {
                "title": "Exclusive Strategies",
                "tagline": "Unlock advanced algorithms for consistent performance.",
                "templates": _fetch("is_exclusive", True),
            },
            "featured": {
                "title": "Featured Strategies",
                "tagline": "Hand-picked beginner-friendly templates.",
                "templates": _fetch("is_featured", True),
            },
            "intraday": {
                "title": "Intraday Strategies",
                "tagline": "5-minute and 15-minute timeframes for active traders.",
                "templates": _fetch("category", "equity_intraday"),
            },
            "swing": {
                "title": "Swing Trading",
                "tagline": "Daily timeframe, 3-10 day holds.",
                "templates": _fetch("category", "equity_swing"),
            },
            "options": {
                "title": "Options Strategies",
                "tagline": "Multi-leg index options — defined and undefined risk.",
                "templates": _fetch("segment", "OPTIONS"),
            },
        }
        return {
            "sections": sections,
            "section_keys": list(sections.keys()),
        }

    return _catalog_cached(("sections",), _produce)


@router.get("/catalog/{slug}")
async def get_strategy_catalog_entry(slug: str) -> Dict[str, Any]:
    """Get one catalog template by slug, including the full DSL document.

    Public — drives the template preview pane on /strategies/[slug].
    """
    sb = get_supabase_admin()
    rows = (
        sb.table("strategy_catalog")
        .select(f"{_CATALOG_FIELDS}, dsl, default_params, configurable_params, strategy_class, template_slug")
        .eq("slug", slug)
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    if not rows.data:
        raise HTTPException(status_code=404, detail="template not found")
    return {"template": rows.data[0]}


@router.post("/from-template/{slug}", status_code=201)
async def clone_strategy_from_template(
    slug: str,
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Clone a catalog template into the caller's user_strategies as a
    new ``draft``. Only works for templates with a populated ``dsl``
    column (i.e. equity templates — multi-leg options templates have
    dsl=NULL and return 409 with a hint to use the rule-based runner)."""
    sb = get_supabase_admin()
    rows = (
        sb.table("strategy_catalog")
        .select("slug, name, description, dsl, engine_compatible, tier_required")
        .eq("slug", slug)
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    if not rows.data:
        raise HTTPException(status_code=404, detail="template not found")

    template = rows.data[0]
    if not template.get("dsl"):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "template_not_dsl_compatible",
                "message": f"Template '{slug}' has no DSL document. "
                f"This template needs to be re-seeded.",
            },
        )

    # Tier gate — pro/elite-only templates can't be cloned by free users
    tpl_tier = (template.get("tier_required") or "free").lower()
    user_tier = _user_tier(user)
    tier_order = {"free": 0, "pro": 1, "elite": 2}
    if tier_order.get(user_tier, 0) < tier_order.get(tpl_tier, 0):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "tier_required",
                "message": f"Template '{slug}' requires {tpl_tier} tier",
                "required_tier": tpl_tier,
                "user_tier": user_tier,
            },
        )

    try:
        row = strat_registry.create_strategy(
            sb,
            user_id=_user_id(user),
            dsl=template["dsl"],
            name=f"{template['name']} (copy)",
            description=template.get("description"),
            template_slug=slug,
            source="template",
        )
    except ValidationError as exc:
        # Stored DSL is invalid — catalog bug, not a client error
        raise HTTPException(
            status_code=500,
            detail={
                "error": "catalog_dsl_invalid",
                "slug": slug,
                "errors": [
                    {"loc": ".".join(str(p) for p in e["loc"]), "msg": e["msg"], "type": e["type"]}
                    for e in exc.errors()
                ],
            },
        )
    return {"strategy": row}


# ─────────────────────────────────────────────────────────────────────
# PR-F CRUD + state machine
# ─────────────────────────────────────────────────────────────────────


class CreateStrategyBody(BaseModel):
    dsl: Dict[str, Any]
    name: Optional[str] = None
    description: Optional[str] = None
    template_slug: Optional[str] = None
    source: str = Field(default="user", pattern="^(user|studio|template)$")


class UpdateDSLBody(BaseModel):
    dsl: Dict[str, Any]


class TransitionBody(BaseModel):
    to: str = Field(pattern="^(draft|backtest|paper|live|paused|archived)$")
    capital_allocated: Optional[float] = Field(default=None, gt=0)


def _user_field(user: Any, name: str) -> Any:
    """Get a field off a user payload regardless of whether it's a dict
    (older tests + JSON) or an object with attrs (the SimpleNamespace
    returned by ``get_current_user`` after JWT decode)."""
    if user is None:
        return None
    if isinstance(user, dict):
        return user.get(name)
    return getattr(user, name, None)


def _user_id(user: Any) -> str:
    """Extract user_id from either shape. Falls back to empty string."""
    return str(_user_field(user, "id") or "")


def _user_tier(user: Any) -> str:
    """Best-effort tier lookup. Defaults to 'free' so we never accidentally
    grant live-trade access from a missing field. Reads via _user_field so
    it works for both the dict shape (older callsites + tests) and the
    SimpleNamespace returned by ``get_current_user`` after PR-Y."""
    explicit = _user_field(user, "tier") or _user_field(user, "subscription_tier")
    if explicit:
        return str(explicit).lower()
    # Fall back to the user_metadata blob on the SimpleNamespace, which
    # carries the Supabase profile fields after PR-Y signup.
    meta = _user_field(user, "user_metadata") or {}
    if isinstance(meta, dict):
        meta_tier = meta.get("tier") or meta.get("subscription_tier")
        if meta_tier:
            return str(meta_tier).lower()
    # Last resort: look up the user_profiles row.
    uid = _user_id(user)
    if uid:
        try:
            from .app import get_supabase_admin  # noqa: PLC0415
            r = get_supabase_admin().table("user_profiles").select("tier").eq("id", uid).single().execute()
            t = (r.data or {}).get("tier") if r.data else None
            if t:
                return str(t).lower()
        except Exception:
            pass
    return "free"


@router.post("", status_code=201)
async def create_strategy(
    body: CreateStrategyBody,
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Create a new strategy in ``draft`` status. DSL is validated first."""
    try:
        row = strat_registry.create_strategy(
            get_supabase_admin(),
            user_id=_user_id(user),
            dsl=body.dsl,
            name=body.name,
            description=body.description,
            template_slug=body.template_slug,
            source=body.source,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"valid": False, "errors": [
                {"loc": ".".join(str(p) for p in e["loc"]), "msg": e["msg"], "type": e["type"]}
                for e in exc.errors()
            ]},
        )
    return {"strategy": row}


@router.get("")
async def list_strategies(
    status: Optional[str] = Query(default=None, pattern="^(draft|backtest|paper|live|paused|archived)$"),
    limit: int = Query(default=100, ge=1, le=500),
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """List the caller's strategies, optionally filtered by status."""
    rows = strat_registry.list_strategies(
        get_supabase_admin(),
        user_id=_user_id(user),
        status=status,
        limit=limit,
    )
    return {"strategies": rows, "count": len(rows)}


class CompareBody(BaseModel):
    strategy_ids: List[str] = Field(min_items=2, max_items=6)


@router.post("/compare")
async def compare_strategies_route(
    body: CompareBody,
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Head-to-head compare of 2–6 of the caller's strategies.

    Loads each strategy (scoped to the user), extracts its out-of-sample
    metrics + the live-promotion gate verdict, and returns a side-by-side
    table with the per-metric winner and a best-overall pick. Missing /
    not-found ids are skipped honestly."""
    from ..ai.strategy.compare import compare_strategies
    sb = get_supabase_admin()
    uid = _user_id(user)
    rows: List[Dict[str, Any]] = []
    for sid in body.strategy_ids:
        row = strat_registry.get_strategy(sb, strategy_id=sid, user_id=uid)
        if row is not None:
            rows.append(row)
    if len(rows) < 2:
        raise HTTPException(
            status_code=404,
            detail="Need at least 2 of your strategies to compare.",
        )
    return compare_strategies(rows)


@router.get("/{strategy_id}")
async def get_strategy(
    strategy_id: str,
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    row = strat_registry.get_strategy(
        get_supabase_admin(),
        strategy_id=strategy_id,
        user_id=_user_id(user),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="strategy not found")
    return {"strategy": row}


@router.patch("/{strategy_id}/dsl")
async def update_strategy_dsl(
    strategy_id: str,
    body: UpdateDSLBody,
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Edit DSL — allowed only when status in {draft, paused}."""
    try:
        row = strat_registry.update_dsl(
            get_supabase_admin(),
            strategy_id=strategy_id,
            user_id=_user_id(user),
            dsl=body.dsl,
        )
    except strat_registry.StrategyStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"valid": False, "errors": [
                {"loc": ".".join(str(p) for p in e["loc"]), "msg": e["msg"], "type": e["type"]}
                for e in exc.errors()
            ]},
        )
    return {"strategy": row}


@router.post("/{strategy_id}/transition")
async def transition_strategy(
    strategy_id: str,
    body: TransitionBody,
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Move strategy through the state machine.

    Two barriers before a strategy can reach **live**:
      1. Tier gate — live deploy requires Pro/Elite.
      2. Quality gate — the strategy's walk-forward / out-of-sample backtest
         must clear the thresholds in ``evaluation.py`` (env-tunable). This is
         the barrier that stops an in-sample-overfit (incl. LLM-generated)
         strategy from trading real money. OPTIONS strategies have no OOS path
         so they're blocked from live by design (paper only).
    """
    from ..ai.strategy.evaluation import GateThresholds, evaluate_gate
    from ..core.config import settings

    sb = get_supabase_admin()

    # 1. Tier gate — live deploy requires pro/elite. paper + backtest + draft + pause are free.
    if body.to == "live" and _user_tier(user) not in _LIVE_DEPLOY_TIERS:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "tier_required",
                "message": "live deployment requires Pro or Elite tier",
                "allowed_tiers": sorted(_LIVE_DEPLOY_TIERS),
            },
        )

    # 2. Quality gate — out-of-sample backtest must pass before live (and paper
    #    too, if STRATEGY_GATE_PAPER_TOO). Skipped for non-deploy transitions
    #    (draft/backtest/paused/archived) and when the gate is disabled.
    gated_targets = {"live"} | ({"paper"} if settings.STRATEGY_GATE_PAPER_TOO else set())
    if settings.STRATEGY_GATE_ENABLED and body.to in gated_targets:
        row = strat_registry.get_strategy(sb, strategy_id=strategy_id, user_id=_user_id(user))
        if row is None:
            raise HTTPException(status_code=404, detail="strategy not found")
        thresholds = GateThresholds(
            min_oos_sharpe=settings.STRATEGY_GATE_MIN_OOS_SHARPE,
            min_trades=settings.STRATEGY_GATE_MIN_TRADES,
            max_drawdown_pct=settings.STRATEGY_GATE_MAX_DRAWDOWN_PCT,
            min_consistency=settings.STRATEGY_GATE_MIN_CONSISTENCY,
            require_holdout_positive=settings.STRATEGY_GATE_REQUIRE_HOLDOUT_POSITIVE,
            min_symbol_breadth=settings.STRATEGY_GATE_MIN_SYMBOL_BREADTH,
            min_regime_coverage=settings.STRATEGY_GATE_MIN_REGIME_COVERAGE,
        )
        gate = evaluate_gate(row.get("last_backtest"), thresholds)
        if not gate.passed:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "gate_failed",
                    "message": (
                        f"This strategy can't go {body.to} yet — it didn't clear the "
                        f"out-of-sample backtest gate. Run a fresh backtest and improve "
                        f"the strategy until these pass:"
                    ),
                    "failures": gate.failures,
                    "metrics": gate.metrics,
                },
            )

    try:
        row = strat_registry.transition_status(
            sb,
            strategy_id=strategy_id,
            user_id=_user_id(user),
            new_status=body.to,
            capital_allocated=body.capital_allocated,
        )
    except strat_registry.StrategyStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"strategy": row}


@router.get("/{strategy_id}/gate")
async def strategy_gate(
    strategy_id: str,
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Read the promotion-gate verdict WITHOUT attempting a transition.

    Runs the same ``evaluate_gate`` the ``→ live`` transition uses, but on
    read only — no state change, no side effects. This lets the Builder show a
    GATE PASS / NEEDS WORK badge (and the specific failures) after a backtest,
    so the user knows where they stand before ever pressing Deploy. Uses the
    default :class:`GateThresholds` (the baseline bar). Owner-scoped → 404 if
    the strategy isn't the caller's.
    """
    from ..ai.strategy.evaluation import evaluate_gate

    row = strat_registry.get_strategy(
        get_supabase_admin(),
        strategy_id=strategy_id,
        user_id=_user_id(user),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="strategy not found")

    last_backtest = row.get("last_backtest")
    gate = evaluate_gate(last_backtest)
    return {
        "has_backtest": bool(last_backtest),
        "passed": gate.passed,
        "failures": gate.failures,
        "metrics": gate.metrics,
    }


@router.delete("/{strategy_id}", status_code=200)
async def archive_strategy(
    strategy_id: str,
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Convenience — DELETE = transition to archived (terminal)."""
    try:
        row = strat_registry.transition_status(
            get_supabase_admin(),
            strategy_id=strategy_id,
            user_id=_user_id(user),
            new_status="archived",
        )
    except strat_registry.StrategyStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"strategy": row}


# ─────────────────────────────────────────────────────────────────────
# PR-E Studio agent (NL → DSL)
# ─────────────────────────────────────────────────────────────────────


class StudioCompileBody(BaseModel):
    prompt: str = Field(min_length=3, max_length=2000)
    save_as_draft: bool = Field(
        default=False,
        description="If true, the compiled DSL is also inserted as a new "
                    "draft strategy and the returned object includes the "
                    "user_strategies row.",
    )


class VisionDraftBody(BaseModel):
    image_b64: str = Field(min_length=16)
    mime: str = Field(default="image/png")
    symbol: Optional[str] = Field(default=None, max_length=32)
    timeframe: str = Field(default="1d", max_length=8)
    compile: bool = Field(default=False)
    save_as_draft: bool = Field(default=False)


@router.post("/studio/compile")
async def studio_compile(
    body: StudioCompileBody,
    user=Depends(get_current_user),
    tier: UserTier = Depends(current_user_tier),
) -> Dict[str, Any]:
    """Compile a natural-language description into a validated Strategy DSL.
    Always emits mode='backtest' — Studio never deploys live directly. If the
    prompt is too under-specified, returns 200 {needs_clarification, ...} with a
    single follow-up question instead of a best-guess strategy. The zero-token
    deterministic pre-gate does NOT burn a credit; a real generator call does."""
    from ..ai.strategy.studio import (
        ClarificationNeeded,
        StudioError,
        compile_strategy,
        is_studio_available,
        precheck_clarification,
    )

    if not is_studio_available():
        raise HTTPException(
            status_code=503,
            detail="Studio compiler unavailable: OPENROUTER_API_KEY not set",
        )

    # Zero-token deterministic pre-gate — an obviously bare prompt returns a
    # structured follow-up WITHOUT calling the generator or burning a credit.
    pre = precheck_clarification(body.prompt)
    if pre is not None:
        return {
            "needs_clarification": True,
            "missing": pre.missing,
            "question": pre.question,
            "assumptions": pre.assumptions,
        }

    # About to spend a generator token — consume the per-feature credit now
    # (raises 402 when over the tier cap). Only reached for prompts with signal.
    consume_llm_cap_or_raise(tier, "strategy_gen")

    try:
        result = compile_strategy(body.prompt)
    except StudioError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "studio_compile_failed", "message": str(exc)},
        )

    # The generator itself may still ask for clarification (a token was spent, so
    # the credit above is correctly consumed).
    if isinstance(result, ClarificationNeeded):
        return {
            "needs_clarification": True,
            "missing": result.missing,
            "question": result.question,
            "assumptions": result.assumptions,
        }

    strategy = result
    dsl = strategy.model_dump(mode="json")
    payload: Dict[str, Any] = {"strategy": dsl}

    if body.save_as_draft:
        try:
            row = strat_registry.create_strategy(
                get_supabase_admin(),
                user_id=_user_id(user),
                dsl=dsl,
                name=strategy.name,
                source="studio",
            )
            payload["saved_row"] = row
        except Exception as exc:  # noqa: BLE001
            payload["save_error"] = str(exc)

    return payload


# ─────────────────────────────────────────────────────────────────────
# WP-VISION — chart image → structured read → synthesized strategy prompt
# ─────────────────────────────────────────────────────────────────────


_VISION_MIME_ALLOW = {"image/png", "image/jpeg", "image/webp"}
_VISION_MAX_BYTES = 5 * 1024 * 1024  # ~5MB decoded


@router.post("/studio/vision-draft")
async def studio_vision_draft(
    body: VisionDraftBody,
    user: UserTier = Depends(RequireFeature("finagent_vision")),
) -> Dict[str, Any]:
    """Chart image → structured read → synthesized strategy prompt. Two-step by
    default (compile=False): the client writes the returned prompt into the
    Builder and the user runs the EXISTING Compile button (which enforces the
    strategy_gen cap + walk-forward gate). Bearish / no-edge / unreadable →
    {prompt: null, note}. Never fabricates a long."""
    import base64

    from ..ai.strategy.studio import (
        ClarificationNeeded, StudioError, compile_strategy, synthesize_prompt_from_vision,
    )
    from ..ai.vision import analyze_chart_image

    mime = (body.mime or "image/png").lower().strip()
    if mime not in _VISION_MIME_ALLOW:
        raise HTTPException(status_code=415, detail={
            "error": "unsupported_media_type",
            "message": "Upload a PNG, JPEG, or WEBP chart image.",
        })

    raw = body.image_b64
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]  # tolerate a data: URL prefix
    try:
        png_bytes = base64.b64decode(raw, validate=False)
    except Exception:
        raise HTTPException(status_code=400, detail={
            "error": "invalid_image", "message": "Could not decode the image."})
    if not png_bytes or len(png_bytes) > _VISION_MAX_BYTES:
        raise HTTPException(status_code=400, detail={
            "error": "invalid_image",
            "message": "Image is empty or larger than 5MB."})

    # A vision call will happen — consume the chart_vision credit (402 if over).
    consume_llm_cap_or_raise(user, "chart_vision")

    analysis = await analyze_chart_image(png_bytes, symbol=(body.symbol or "uploaded"), mime=mime)
    if not analysis.available:
        return {"analysis": asdict(analysis), "prompt": None,
                "note": "Couldn't read this chart — try a clearer screenshot."}

    prompt = synthesize_prompt_from_vision(
        analysis, symbol=(body.symbol or ""), timeframe=body.timeframe)
    if not prompt:
        return {"analysis": asdict(analysis), "prompt": None,
                "note": "This chart doesn't show a long setup with an edge right now."}

    result: Dict[str, Any] = {"analysis": asdict(analysis), "prompt": prompt}

    if body.compile:
        # A generator token will be spent — consume the strategy_gen credit too.
        consume_llm_cap_or_raise(user, "strategy_gen")
        try:
            compiled = compile_strategy(prompt)
        except StudioError as exc:
            raise HTTPException(status_code=422, detail={
                "error": "studio_compile_failed", "message": str(exc)})
        if isinstance(compiled, ClarificationNeeded):
            result["needs_clarification"] = True
            result["missing"] = compiled.missing
            result["question"] = compiled.question
            result["assumptions"] = compiled.assumptions
            return result
        dsl = compiled.model_dump(mode="json")
        result["strategy"] = dsl
        if body.save_as_draft:
            try:
                # `user` here is a UserTier (from RequireFeature), whose id field
                # is `user_id` — _user_id() reads `.id` and would return "".
                row = strat_registry.create_strategy(
                    get_supabase_admin(), user_id=user.user_id,
                    dsl=dsl, name=compiled.name, source="vision")
            except Exception as exc:  # noqa: BLE001
                result["save_error"] = str(exc)
            else:
                result["saved_row"] = row

    return result


# ─────────────────────────────────────────────────────────────────────
# PR-G real-time backtest
# ─────────────────────────────────────────────────────────────────────


class BacktestBody(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    lookback_days: Optional[int] = Field(default=None, ge=30, le=730)
    initial_capital: float = Field(default=500_000.0, ge=10_000, le=100_000_000)


def _fetch_tf_ohlcv(provider, symbol: str, *, tf_cfg, lookback_days: int):
    """Fetch + normalize + (4h-)resample OHLCV at the strategy's timeframe.

    Daily derives its history window from ``lookback_days``; intraday uses the
    timeframe's provider history limit (e.g. 60d for 5m, 2y for 1h). 4h is
    fetched as 1h then resampled. Returns a lowercase-column DataFrame or None.
    """
    from ..ai.strategy.indicators import MIN_LOOKBACK
    from ..ai.strategy.timeframes import resample_ohlcv

    if tf_cfg.timeframe == "1d":
        period_days = lookback_days + MIN_LOOKBACK + 20
        period = "1y" if period_days <= 252 else "2y" if period_days <= 504 else "5y"
    else:
        period = tf_cfg.fetch_period

    df = provider.get_historical(symbol, period=period, interval=tf_cfg.fetch_interval)
    if df is None or len(df) == 0:
        return None
    df.columns = [c.lower() for c in df.columns]
    if tf_cfg.resample_to:
        df = resample_ohlcv(df, tf_cfg.resample_to)
    return df


@router.post("/{strategy_id}/backtest")
async def backtest_strategy(
    strategy_id: str,
    body: BacktestBody,
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Run the strategy through the walk-forward backtest at its OWN timeframe
    (5m/15m/1h/4h/1d — user/LLM-decided) and persist the summary (incl. the
    out-of-sample block) to user_strategies.last_backtest.

    Dispatch:
      • OPTIONS          → options walk-forward (now has an OOS block too).
      • EQUITY single    → single-symbol walk-forward on body.symbol.
      • EQUITY universe  → multi-symbol walk-forward across the declared
                           universe (capped) → breadth-gated.
    """
    from ..ai.strategy.backtest import (
        run_options_walk_forward,
        run_universe_walk_forward,
        run_walk_forward,
    )
    from ..ai.strategy.dsl import Strategy as DSLStrategy
    from ..ai.strategy.indicators import MIN_LOOKBACK
    from ..ai.strategy.timeframes import annualization_periods, tf_config
    from ..core.config import settings
    from ..data.market import get_market_data_provider

    sb = get_supabase_admin()
    row = strat_registry.get_strategy(sb, strategy_id=strategy_id, user_id=_user_id(user))
    if row is None:
        raise HTTPException(status_code=404, detail="strategy not found")

    try:
        strategy = DSLStrategy.model_validate(row["dsl"])
    except ValidationError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "stored_dsl_invalid", "errors": exc.errors()},
        )

    # Timeframe drives fetch interval/period + Sharpe annualization.
    tf_cfg = tf_config(strategy.timeframe)
    ppy = annualization_periods(strategy.timeframe)
    lookback = body.lookback_days or strategy.lookback_days
    folds = settings.STRATEGY_GATE_FOLDS
    provider = get_market_data_provider()

    is_options = strategy.instrument_segment.value == "OPTIONS"
    is_universe = (not is_options) and strategy.universe.value != "single"
    uses_regime = _strategy_uses_regime(strategy)
    regime_coverage = None  # set only when the strategy actually uses regime

    try:
        if is_universe:
            # Multi-symbol walk-forward across the declared universe (capped).
            from ..services.strategy_runner.universe_expander import expand_universe
            symbols = expand_universe(strategy.universe.value)[: settings.STRATEGY_GATE_UNIVERSE_MAX_SYMBOLS]
            ohlcv_by_symbol: Dict[str, Any] = {}
            for sym in symbols:
                try:
                    df = _fetch_tf_ohlcv(provider, sym, tf_cfg=tf_cfg, lookback_days=lookback)
                except Exception:  # noqa: BLE001 — one bad symbol shouldn't fail the batch
                    continue
                if df is not None and len(df) >= MIN_LOOKBACK + 10:
                    ohlcv_by_symbol[sym] = df
            if not ohlcv_by_symbol:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "insufficient_history",
                        "message": f"no symbols in {strategy.universe.value} had enough "
                        f"{tf_cfg.timeframe} history ({MIN_LOOKBACK + 10}+ bars).",
                    },
                )
            # Per-symbol regime injection (regime is market-wide, aligned to each
            # symbol's bars). Only when the strategy uses regime — else None +
            # no coverage cost.
            engine_signals_by_symbol = (
                {s: _maybe_load_engine_signals(sb, strategy, d) for s, d in ohlcv_by_symbol.items()}
                if uses_regime else None
            )
            if uses_regime:
                starts = [d.index[0] for d in ohlcv_by_symbol.values()]
                ends = [d.index[-1] for d in ohlcv_by_symbol.values()]
                regime_coverage = _regime_coverage_range(sb, _date_of(min(starts)), _date_of(max(ends)))
            result = run_universe_walk_forward(
                strategy, ohlcv_by_symbol,
                universe=strategy.universe.value,
                folds=folds,
                initial_capital=body.initial_capital,
                periods_per_year=ppy,
                engine_signals_by_symbol=engine_signals_by_symbol,
            )
        else:
            # Single symbol (equity or options).
            try:
                ohlcv = _fetch_tf_ohlcv(provider, body.symbol, tf_cfg=tf_cfg, lookback_days=lookback)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=502,
                    detail={"error": "market_data_unavailable", "message": str(exc)},
                )
            if ohlcv is None or len(ohlcv) < MIN_LOOKBACK + 10:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "insufficient_history",
                        "message": f"{body.symbol}: got {0 if ohlcv is None else len(ohlcv)} "
                        f"{tf_cfg.timeframe} bars, need >= {MIN_LOOKBACK + 10}.",
                    },
                )
            engine_signals_by_date = _maybe_load_engine_signals(sb, strategy, ohlcv)
            if uses_regime:
                regime_coverage = _regime_coverage_range(sb, _date_of(ohlcv.index[0]), _date_of(ohlcv.index[-1]))
            wf = run_options_walk_forward if is_options else run_walk_forward
            result = wf(
                strategy,
                ohlcv,
                symbol=body.symbol,
                folds=folds,
                initial_capital=body.initial_capital,
                engine_signals_by_date=engine_signals_by_date,
                periods_per_year=ppy,
            )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "backtest_failed", "message": str(exc)})

    # Persist summary (incl. the out_of_sample block + regime provenance) so
    # /strategies list + the promotion gate can read it without re-running.
    summary = result.to_summary_dict()
    if regime_coverage is not None:
        # Records that this strategy USES regime + how much of the backtest
        # window ran on REAL regime. The gate fails-closed when coverage is low.
        summary["regime"] = {"used": True, "coverage": round(regime_coverage, 4)}
    strat_registry.record_backtest(
        sb,
        strategy_id=strategy_id,
        user_id=_user_id(user),
        summary=summary,
    )

    full = result.to_full_dict()
    if regime_coverage is not None:
        full["regime"] = summary["regime"]
    return full


# ─────────────────────────────────────────────────────────────────────
# AI Backtesting Assistant — explain the stored backtest + suggest fixes
# ─────────────────────────────────────────────────────────────────────


@router.post("/{strategy_id}/explain-backtest")
async def explain_strategy_backtest(
    strategy_id: str,
    use_llm: bool = Query(default=False, description=(
        "Also generate the grounded AI narrative (cached per backtest/day). "
        "Deterministic drivers + suggestions are always returned."
    )),
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """AI read of the strategy's persisted last_backtest: deterministic
    gate-aware drivers + improvement suggestions (always, 0 tokens) and an
    optional grounded narrative (``use_llm=true``). The promotion-gate math
    decides the verdict — the LLM only narrates. Honest 404 when the strategy
    has no stored backtest yet."""
    from ..services.explain.backtest_explainer import explain_backtest

    sb = get_supabase_admin()
    row = strat_registry.get_strategy(sb, strategy_id=strategy_id, user_id=_user_id(user))
    if row is None:
        raise HTTPException(status_code=404, detail="strategy not found")
    metrics = row.get("last_backtest") or {}
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="no backtest on record — run a backtest first",
        )
    return explain_backtest(metrics, row.get("dsl"), use_llm=use_llm, user_id=_user_id(user))


# ─────────────────────────────────────────────────────────────────────
# PR-AB — universe backtest (multi-symbol parallel)
# ─────────────────────────────────────────────────────────────────────


class UniverseBacktestBody(BaseModel):
    """Multi-symbol backtest request.

    ``universe`` overrides ``strategy.universe`` when set. Pass e.g.
    "nifty50" to test the strategy against the 50-stock list without
    editing the saved DSL. ``max_symbols`` caps wall-clock so the
    request stays under the 5-min Vercel timeout — defaults to 30,
    full universe support via background job is a future PR.
    """
    universe: Optional[str] = Field(
        default=None,
        description="Override strategy.universe. nifty50 / nifty100 / nifty500 / sector:IT etc.",
    )
    lookback_days: Optional[int] = Field(default=None, ge=30, le=730)
    initial_capital_per_symbol: float = Field(
        default=100_000.0,
        ge=10_000,
        le=10_000_000,
        description="Per-symbol allocation. Total = this × number of symbols.",
    )
    max_symbols: int = Field(default=30, ge=1, le=200)


@router.post("/{strategy_id}/backtest/universe")
async def backtest_strategy_universe(
    strategy_id: str,
    body: UniverseBacktestBody,
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Run the strategy across every symbol in a universe and return
    both an aggregate summary and a per-symbol breakdown.

    Universes supported: nifty50, nifty100, nifty500, sector:IT/BANK/
    AUTO/PHARMA/FMCG/METAL/ENERGY/INFRA, single (uses strategy.symbol).

    Concurrency is capped so we don't melt the market-data provider;
    a full nifty500 run is rejected with ``max_symbols`` for now —
    use 100 + filter your strategy to a narrower segment for v1.
    """
    import asyncio

    from ..ai.strategy.backtest import run_dsl_backtest
    from ..ai.strategy.dsl import Strategy as DSLStrategy
    from ..ai.strategy.indicators import MIN_LOOKBACK
    from ..data.market import get_market_data_provider
    from ..services.strategy_runner.universe_expander import expand_universe

    sb = get_supabase_admin()
    row = strat_registry.get_strategy(sb, strategy_id=strategy_id, user_id=_user_id(user))
    if row is None:
        raise HTTPException(status_code=404, detail="strategy not found")

    try:
        strategy = DSLStrategy.model_validate(row["dsl"])
    except ValidationError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "stored_dsl_invalid", "errors": exc.errors()},
        )

    universe_name = body.universe or strategy.universe.value
    try:
        symbols = expand_universe(
            universe_name,
            single_symbol=strategy.symbol if universe_name == "single" else None,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "bad_universe", "message": str(exc)},
        )

    if not symbols:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "empty_universe",
                "message": f"universe '{universe_name}' resolved to 0 symbols",
            },
        )

    if len(symbols) > body.max_symbols:
        symbols = symbols[: body.max_symbols]

    lookback = body.lookback_days or strategy.lookback_days
    period_days = lookback + MIN_LOOKBACK + 20
    period_str = (
        "1y" if period_days <= 252
        else "2y" if period_days <= 504
        else "5y"
    )

    provider = get_market_data_provider()
    engine_signals_cache: Dict[str, Any] = {}

    # Bound concurrency — provider rate-limits otherwise.
    semaphore = asyncio.Semaphore(5)

    async def _run_one(sym: str) -> Dict[str, Any]:
        async with semaphore:
            try:
                ohlcv = await asyncio.to_thread(
                    provider.get_historical, sym, period_str, "1d",
                )
                if ohlcv is None or len(ohlcv) < MIN_LOOKBACK + 10:
                    return {
                        "symbol": sym,
                        "status": "skipped",
                        "reason": f"insufficient_history ({0 if ohlcv is None else len(ohlcv)} bars)",
                    }
                ohlcv.columns = [c.lower() for c in ohlcv.columns]

                if sym not in engine_signals_cache:
                    engine_signals_cache[sym] = _maybe_load_engine_signals(
                        sb, strategy, ohlcv,
                    )

                if strategy.instrument_segment.value == "OPTIONS":
                    from ..ai.strategy.options_backtest import run_options_backtest
                    r = await asyncio.to_thread(
                        run_options_backtest,
                        strategy, ohlcv,
                        symbol=sym,
                        initial_capital=body.initial_capital_per_symbol,
                        engine_signals_by_date=engine_signals_cache[sym],
                    )
                else:
                    r = await asyncio.to_thread(
                        run_dsl_backtest,
                        strategy, ohlcv,
                        symbol=sym,
                        initial_capital=body.initial_capital_per_symbol,
                        engine_signals_by_date=engine_signals_cache[sym],
                    )
                summary = r.to_summary_dict()
                # Compute the absolute INR P&L so the aggregate is correct.
                pnl_inr = r.final_capital - body.initial_capital_per_symbol
                return {
                    "symbol": sym,
                    "status": "ok",
                    "total_return_pct": summary.get("total_return_pct"),
                    "sharpe_ratio": summary.get("sharpe_ratio"),
                    "win_rate": summary.get("win_rate"),
                    "max_drawdown_pct": summary.get("max_drawdown_pct"),
                    "total_trades": summary.get("total_trades"),
                    "final_capital": r.final_capital,
                    "pnl_inr": pnl_inr,
                }
            except Exception as exc:
                return {
                    "symbol": sym,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
                }

    results = await asyncio.gather(*[_run_one(s) for s in symbols])

    # Aggregate the ``ok`` rows. Skipped + failed counted separately so
    # the user sees "we ran on 27 of 30 symbols, here are the 3 reasons".
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "failed"]

    if not ok:
        return {
            "universe": universe_name,
            "symbols_attempted": len(symbols),
            "results": results,
            "aggregate": None,
            "skipped_count": len(skipped),
            "failed_count": len(failed),
        }

    def _avg(field: str) -> float:
        vals = [r[field] for r in ok if r.get(field) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    total_capital = body.initial_capital_per_symbol * len(ok)
    total_pnl = sum(r["pnl_inr"] for r in ok)
    winners = [r for r in ok if r["pnl_inr"] > 0]
    losers = [r for r in ok if r["pnl_inr"] <= 0]

    aggregate = {
        "symbols_run": len(ok),
        "winners": len(winners),
        "losers": len(losers),
        "win_pct": len(winners) / len(ok) if ok else 0,
        "total_capital_deployed": total_capital,
        "total_pnl_inr": total_pnl,
        "total_return_pct": (total_pnl / total_capital * 100) if total_capital > 0 else 0,
        "avg_return_pct_per_symbol": _avg("total_return_pct"),
        "avg_sharpe": _avg("sharpe_ratio"),
        "avg_win_rate": _avg("win_rate"),
        "avg_max_drawdown_pct": _avg("max_drawdown_pct"),
        "sum_trades": sum(int(r.get("total_trades") or 0) for r in ok),
    }

    # Sort symbol results: best return first.
    ok.sort(key=lambda r: -(r.get("total_return_pct") or 0))

    return {
        "universe": universe_name,
        "lookback_days": lookback,
        "initial_capital_per_symbol": body.initial_capital_per_symbol,
        "symbols_attempted": len(symbols),
        "aggregate": aggregate,
        "results": ok,
        "skipped": skipped,
        "failed": failed,
    }


@router.get("/deployed")
async def list_deployed_strategies(
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """One-call aggregate for the /strategies/deployed dashboard.

    For every strategy the caller has on paper or live status, returns:
      - basic info (name, status, deployed_at, symbol/universe)
      - open positions with live mark-to-market P&L
      - lifetime stats (entries, exits, realized PnL, win rate)
      - last 6 events (entry/exit) for activity feed
      - backtest win-rate baseline so the user can compare paper vs
        what the strategy promised on history

    Designed to power the deployed-panel surface in one round-trip so
    the page renders instantly instead of fanning out N HTTP requests.
    """
    sb = get_supabase_admin()
    user_id = _user_id(user)

    # ── 1. Load every deployed strategy ──
    strat_rows = (
        sb.table("user_strategies")
        .select(
            "id, name, status, dsl, deployed_at, updated_at, last_backtest, "
            "strategy_intent, template_slug"
        )
        .eq("user_id", user_id)
        .in_("status", ["paper", "live"])
        .order("deployed_at", desc=True)
        .limit(50)
        .execute()
        .data
        or []
    )
    if not strat_rows:
        return {"deployed": [], "count": 0}

    strategy_ids = [s["id"] for s in strat_rows]

    # ── 2. Bulk-load every open position across all those strategies ──
    pos_rows = (
        sb.table("strategy_positions")
        .select(
            "id, strategy_id, symbol, side, quantity, entry_price, "
            "stop_loss, target_1, status, last_evaluated_at, exit_reason, exit_price"
        )
        .eq("user_id", user_id)
        .in_("strategy_id", strategy_ids)
        .order("last_evaluated_at", desc=True)
        .limit(500)
        .execute()
        .data
        or []
    )
    open_positions = [p for p in pos_rows if p.get("status") == "open"]
    closed_positions = [p for p in pos_rows if p.get("status") == "closed"]

    # ── 3. Live mark-to-market: batch quote every unique open symbol ──
    open_symbols = list({p["symbol"] for p in open_positions if p.get("symbol")})
    quote_map: Dict[str, Dict[str, Any]] = {}
    if open_symbols:
        try:
            from ..data.market import get_market_data_provider
            provider = get_market_data_provider()
            raw_quotes = provider.get_quotes_batch(open_symbols[:200])
            for sym, q in raw_quotes.items():
                if not q:
                    continue
                ltp = getattr(q, "ltp", None) or (q.get("ltp") if isinstance(q, dict) else None)
                if ltp:
                    quote_map[sym] = {"ltp": float(ltp)}
        except Exception:
            pass

    # ── 4. Bulk-load recent entry/exit signals for the activity feed ──
    signal_rows = (
        sb.table("signals")
        .select("id, strategy_id, symbol, action, entry_price, created_at, market_context")
        .eq("user_id", user_id)
        .in_("strategy_id", strategy_ids)
        .eq("source", "user_strategy")
        .order("created_at", desc=True)
        .limit(300)
        .execute()
        .data
        or []
    )

    # ── 5. Per-strategy assembly ──
    by_id: Dict[str, Dict[str, Any]] = {}
    for s in strat_rows:
        dsl = s.get("dsl") or {}
        last_backtest = s.get("last_backtest") or {}
        by_id[s["id"]] = {
            "id": s["id"],
            "name": s.get("name") or "Untitled strategy",
            "status": s.get("status"),
            "deployed_at": s.get("deployed_at"),
            "template_slug": s.get("template_slug"),
            "universe": dsl.get("universe") or "single",
            "symbol": dsl.get("symbol"),
            "stop_loss_pct": dsl.get("stop_loss_pct"),
            "take_profit_pct": dsl.get("take_profit_pct"),
            "backtest_win_rate_pct": last_backtest.get("win_rate"),
            "backtest_total_return_pct": last_backtest.get("total_return_pct"),
            "backtest_sharpe": last_backtest.get("sharpe_ratio"),
            "open_positions": [],
            "stats": {
                "open_count": 0, "total_signals": 0,
                "entries_emitted": 0, "exits_emitted": 0,
                "winning_exits": 0, "losing_exits": 0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "total_pnl": 0.0,
                "win_rate_pct": None,
            },
            "recent_events": [],
        }

    # 5a — open positions + unrealized PnL
    for pos in open_positions:
        sid = pos.get("strategy_id")
        if sid not in by_id:
            continue
        sym = pos.get("symbol")
        qty = int(pos.get("quantity") or 0)
        entry_p = float(pos.get("entry_price") or 0)
        ltp = quote_map.get(sym, {}).get("ltp")
        unrealized = (float(ltp) - entry_p) * qty if ltp and entry_p else 0.0
        unrealized_pct = ((float(ltp) - entry_p) / entry_p * 100) if ltp and entry_p else 0.0
        by_id[sid]["open_positions"].append({
            "id": pos["id"],
            "symbol": sym,
            "quantity": qty,
            "entry_price": entry_p,
            "current_price": float(ltp) if ltp else None,
            "stop_loss": pos.get("stop_loss"),
            "target_1": pos.get("target_1"),
            "unrealized_pnl": round(unrealized, 2),
            "unrealized_pnl_pct": round(unrealized_pct, 2),
        })
        by_id[sid]["stats"]["open_count"] += 1
        by_id[sid]["stats"]["unrealized_pnl"] += unrealized

    # 5b — closed positions feed realized PnL + win/loss counters
    for pos in closed_positions:
        sid = pos.get("strategy_id")
        if sid not in by_id:
            continue
        entry_p = float(pos.get("entry_price") or 0)
        exit_p = float(pos.get("exit_price") or 0)
        qty = int(pos.get("quantity") or 0)
        if not entry_p or not exit_p or not qty:
            continue
        pnl = (exit_p - entry_p) * qty
        by_id[sid]["stats"]["realized_pnl"] += pnl
        if pnl > 0:
            by_id[sid]["stats"]["winning_exits"] += 1
        else:
            by_id[sid]["stats"]["losing_exits"] += 1

    # 5c — recent events + signal counters (latest 6 events per strategy)
    for sig in signal_rows:
        sid = sig.get("strategy_id")
        if sid not in by_id:
            continue
        by_id[sid]["stats"]["total_signals"] += 1
        action = sig.get("action") or ""
        if action == "buy":
            by_id[sid]["stats"]["entries_emitted"] += 1
            kind = "entry"
        elif action.startswith("close") or action == "sell":
            by_id[sid]["stats"]["exits_emitted"] += 1
            kind = "exit"
        else:
            continue
        if len(by_id[sid]["recent_events"]) < 6:
            ctx = sig.get("market_context") or {}
            by_id[sid]["recent_events"].append({
                "kind": kind,
                "symbol": sig.get("symbol"),
                "price": float(sig.get("entry_price") or 0),
                "at": sig.get("created_at"),
                "reason": ctx.get("exit_reason") if kind == "exit" else "entry",
            })

    # 5d — derive totals + win rate
    for s in by_id.values():
        st = s["stats"]
        st["total_pnl"] = round(st["realized_pnl"] + st["unrealized_pnl"], 2)
        st["realized_pnl"] = round(st["realized_pnl"], 2)
        st["unrealized_pnl"] = round(st["unrealized_pnl"], 2)
        closed = st["winning_exits"] + st["losing_exits"]
        st["win_rate_pct"] = (
            round(st["winning_exits"] / closed * 100, 1) if closed > 0 else None
        )

    deployed = list(by_id.values())
    return {"deployed": deployed, "count": len(deployed)}


@router.get("/{strategy_id}/executions")
async def list_executions(
    strategy_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Recent strategy_executions rows for forensics / dashboard."""
    sb = get_supabase_admin()
    # Verify ownership first
    if strat_registry.get_strategy(sb, strategy_id=strategy_id, user_id=_user_id(user)) is None:
        raise HTTPException(status_code=404, detail="strategy not found")
    rows = (
        sb.table("strategy_executions")
        .select("*")
        .eq("strategy_id", strategy_id)
        .order("tick_at", desc=True)
        .limit(limit)
        .execute()
    )
    return {"executions": rows.data or [], "count": len(rows.data or [])}
