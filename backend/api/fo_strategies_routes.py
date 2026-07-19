"""
================================================================================
F&O STRATEGIES ROUTES — F6 Elite (PR 30)
================================================================================
HTTP surface for ``/fo-strategies`` — weekly options strategy
recommendations for index underliers. The inputs:

    - 5-day VIX slope direction (rising/stable/falling) derived from the
      ``regime_history`` table — the HMM job writes one row per trading
      day with the day's VIX close. We compute the 5-day mean and
      compare to current VIX (locked 2026-05-17 when vix_tft was
      dropped from v1; pure rule-based per Step 1 §F6 tertiary stack).
    - Current HMM market regime (``regime_history`` latest row)
    - Current spot price       (market_data / yfinance fallback)

The recommender (``backend/ai/fo/strategies.py``) turns those into
1-2 ranked strategy proposals per symbol, each with priced legs + BS
Greeks + max-profit / max-loss / breakevens / probability of profit.

Endpoints (all gated by ``RequireFeature("fo_strategies")`` = Elite):

    GET  /api/fo-strategies/overview               — recs + VIX + regime
    GET  /api/fo-strategies/recommend/{symbol}     — single-symbol ranked list
    POST /api/fo-strategies/price                  — price a specific strategy
================================================================================
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.database import get_supabase_admin
from ..core.tiers import UserTier
from ..middleware.llm_caps import enforce_llm_cap
from ..middleware.tier_gate import RequireFeature
from ..ai.fo import recommend_strategies, price_strategy, StrategyProposal
from ..data.market_calendar import IST

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fo-strategies", tags=["fo-strategies"])

SUPPORTED_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY"]


# ============================================================================
# Pydantic
# ============================================================================


class PriceRequest(BaseModel):
    strategy: str = Field(..., description="iron_condor / bull_call_spread / bear_put_spread / long_straddle / short_strangle / iron_butterfly")
    symbol: str = Field("NIFTY")
    expiry: Optional[str] = Field(None, description="YYYY-MM-DD; defaults to next weekly")


# ============================================================================
# helpers
# ============================================================================


def _load_vix_history(n_days: int = 5) -> List[float]:
    """Return the last ``n_days`` daily VIX closes, newest first.

    Reads from ``regime_history`` — the HMM job writes one row per
    trading day with the day's VIX close in the ``vix`` column. Used
    by ``_compute_vix_direction`` to derive rising/stable/falling
    without an ML forecaster (vix_tft was dropped from v1; per Step 1
    §F6 the tertiary rule-based stack is sufficient for launch).
    """
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("regime_history")
            .select("vix, detected_at")
            .order("detected_at", desc=True)
            .limit(n_days)
            .execute()
        )
        return [
            float(r["vix"]) for r in (rows.data or [])
            if r.get("vix") is not None
        ]
    except Exception as exc:
        logger.debug("regime_history VIX lookup failed: %s", exc)
        return []


def _load_latest_regime() -> Dict[str, Any]:
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("regime_history")
            .select("regime, prob_bull, prob_sideways, prob_bear, vix, nifty_close, detected_at")
            .order("detected_at", desc=True)
            .limit(1)
            .execute()
        )
        if rows.data:
            return rows.data[0]
    except Exception as exc:
        logger.debug("regime_history lookup failed: %s", exc)
    return {}


def _spot_for(symbol: str, regime_row: Dict[str, Any]) -> float:
    """Best-effort spot. For NIFTY use regime_history.nifty_close as
    fallback; for others try MarketData. Returns 0 if unknown — caller
    must handle."""
    try:
        from ..data.market import MarketData
        md = MarketData()
        q = md.get_quote(symbol)
        if q and q.ltp and q.ltp > 0:
            return float(q.ltp)
    except Exception as exc:
        logger.debug("spot lookup failed for %s: %s", symbol, exc)
    if symbol.upper() == "NIFTY" and regime_row.get("nifty_close"):
        return float(regime_row["nifty_close"])
    # Reasonable defaults to avoid zero-divide in BS. Mirrors prices
    # around the time of this PR — only kicks in when live feed is out.
    return {"NIFTY": 22850.0, "BANKNIFTY": 48200.0, "FINNIFTY": 20400.0}.get(symbol.upper(), 1000.0)


def _proposal_to_dict(p: StrategyProposal) -> Dict[str, Any]:
    d = asdict(p)
    # dataclasses serialize legs via asdict too — fine.
    return d


#: 5-day VIX slope thresholds. Current must be ≥ +5% over the mean to
#: count as "rising", ≤ -5% to count as "falling". The thresholds are
#: deliberate noise floors — India VIX moves ~3-4% daily inside a stable
#: regime, so a 5% delta filters intra-week wobble while still firing on
#: real volatility shifts (pre-earnings spikes, Fed-decision weeks, etc).
_VIX_RISING_FACTOR = 1.05
_VIX_FALLING_FACTOR = 0.95


def _compute_vix_direction(
    vix_history: List[float],
    current_vix: Optional[float],
) -> tuple[str, Optional[float]]:
    """Derive 'rising' / 'stable' / 'falling' from a 5-day VIX slope.

    Returns (direction, 5d_mean). The 5d_mean is surfaced in the API
    payload so customers can see the comparison the model used — same
    transparency contract the old TFT-forecast payload had.

    Args:
        vix_history: last N days of VIX closes (any order; we just need
            the mean). Empty → returns ("stable", None).
        current_vix: latest VIX value. None → returns ("stable", mean).
    """
    if not vix_history:
        return "stable", None
    try:
        mean_5d = float(sum(vix_history) / len(vix_history))
    except (TypeError, ValueError, ZeroDivisionError):
        return "stable", None
    if current_vix is None:
        return "stable", mean_5d
    try:
        cv = float(current_vix)
    except (TypeError, ValueError):
        return "stable", mean_5d
    if cv >= mean_5d * _VIX_RISING_FACTOR:
        return "rising", mean_5d
    if cv <= mean_5d * _VIX_FALLING_FACTOR:
        return "falling", mean_5d
    return "stable", mean_5d


# ============================================================================
# routes
# ============================================================================


@router.get("/overview")
async def get_overview(
    user: UserTier = Depends(RequireFeature("fo_strategies")),
) -> Dict[str, Any]:
    """Primary page payload: VIX + regime + per-symbol recommendations."""
    regime_row = _load_latest_regime()
    regime = (regime_row.get("regime") or "sideways").lower()
    current_vix = regime_row.get("vix")
    vix_history = _load_vix_history(n_days=5)
    direction, vix_5d_mean = _compute_vix_direction(vix_history, current_vix)

    recs: Dict[str, List[Dict[str, Any]]] = {}
    for sym in SUPPORTED_SYMBOLS:
        spot = _spot_for(sym, regime_row)
        try:
            props = recommend_strategies(
                symbol=sym, spot=spot,
                vix=current_vix if current_vix is not None else (vix_5d_mean or 15.0),
                vix_direction=direction, regime=regime,
            )
        except Exception as exc:
            logger.exception("recommend_strategies failed for %s: %s", sym, exc)
            props = []
        recs[sym] = [_proposal_to_dict(p) for p in props]

    return {
        "as_of": datetime.now(IST).isoformat(),
        "regime": {
            "name": regime,
            "prob_bull": regime_row.get("prob_bull"),
            "prob_sideways": regime_row.get("prob_sideways"),
            "prob_bear": regime_row.get("prob_bear"),
        } if regime_row else None,
        "vix": {
            "current": current_vix,
            "direction": direction,
            "mean_5d": round(vix_5d_mean, 2) if vix_5d_mean is not None else None,
            "n_history_days": len(vix_history),
            "method": "5d_slope_rule",
        },
        "symbols": SUPPORTED_SYMBOLS,
        "recommendations": recs,
    }


@router.get("/recommend/{symbol}")
async def recommend(
    symbol: str,
    user: UserTier = Depends(RequireFeature("fo_strategies")),
) -> Dict[str, Any]:
    sym = symbol.upper()
    if sym not in SUPPORTED_SYMBOLS:
        raise HTTPException(status_code=400, detail="unsupported_symbol")
    regime_row = _load_latest_regime()
    regime = (regime_row.get("regime") or "sideways").lower()
    current_vix = regime_row.get("vix")
    vix_history = _load_vix_history(n_days=5)
    direction, vix_5d_mean = _compute_vix_direction(vix_history, current_vix)
    spot = _spot_for(sym, regime_row)
    props = recommend_strategies(
        symbol=sym, spot=spot,
        vix=current_vix if current_vix is not None else (vix_5d_mean or 15.0),
        vix_direction=direction, regime=regime,
    )
    return {
        "symbol": sym,
        "spot": round(spot, 2),
        "regime": regime,
        "vix_direction": direction,
        "vix_level": current_vix,
        "vix_5d_mean": round(vix_5d_mean, 2) if vix_5d_mean is not None else None,
        "recommendations": [_proposal_to_dict(p) for p in props],
    }


@router.post("/price")
async def price(
    body: PriceRequest,
    user: UserTier = Depends(RequireFeature("fo_strategies")),
) -> Dict[str, Any]:
    sym = body.symbol.upper()
    regime_row = _load_latest_regime()
    current_vix = regime_row.get("vix") or 15.0
    expiry = None
    if body.expiry:
        try:
            expiry = date.fromisoformat(body.expiry)
        except ValueError:
            raise HTTPException(status_code=422, detail="invalid_expiry")
    spot = _spot_for(sym, regime_row)
    prop = price_strategy(
        body.strategy, symbol=sym, spot=spot, vix=float(current_vix), expiry=expiry,
    )
    if prop is None:
        raise HTTPException(status_code=400, detail="unknown_strategy")
    return _proposal_to_dict(prop)


# ============================================================================
# PR-AT — Paper options trading
# ============================================================================


class PaperOpenRequest(BaseModel):
    """Open a multi-leg paper option position.

    Two equivalent ways to specify the legs:
      a) Pass ``template`` (e.g. 'bull_call_spread') + ``symbol`` and we
         resolve the rec via the same recommender used by /recommend.
         Simplest — user clicks "Deploy to paper" on a recommendation.
      b) Pass ``legs`` directly (advanced) — list of LegSpec-shaped
         dicts {side, option_type, strike_anchor, strike_offset, expiry}.
    """
    template: Optional[str] = Field(None, description=(
        "Strategy name: bull_call_spread | bear_put_spread | iron_condor |"
        " long_straddle | short_strangle | iron_butterfly"
    ))
    symbol: str = Field("NIFTY")
    lots: int = Field(default=1, ge=1, le=20,
                      description="Deployment lots (each is symbol's contract lot_size)")
    legs: Optional[List[Dict[str, Any]]] = Field(default=None)


@router.post("/paper/open")
async def paper_open(
    body: PaperOpenRequest,
    user: UserTier = Depends(RequireFeature("fo_strategies")),
) -> Dict[str, Any]:
    """Open a paper options position. Returns the new position id."""
    from ..services.execution.paper_options_executor import open_paper_option_position
    from ..ai.strategy.dsl import LegSpec, OptionSide, OptionType, StrikeAnchor, ExpiryAnchor

    sym = body.symbol.upper()
    regime_row = _load_latest_regime()
    spot = _spot_for(sym, regime_row)
    current_vix = float(regime_row.get("vix") or 15.0)
    # VIX → sigma (annualised). The recommender uses the same convention.
    sigma = current_vix / 100.0

    legs: List[LegSpec] = []
    template_slug: Optional[str] = body.template

    if body.legs:
        for raw in body.legs:
            try:
                legs.append(LegSpec(
                    side=OptionSide(raw["side"]),
                    option_type=OptionType(raw["option_type"]),
                    strike_anchor=StrikeAnchor(raw["strike_anchor"]),
                    strike_offset=float(raw.get("strike_offset", 0)),
                    expiry=ExpiryAnchor(raw.get("expiry", "CURRENT_WEEK")),
                    qty_lots=int(raw.get("qty_lots", 1)),
                ))
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"invalid_leg: {exc}")
    elif body.template:
        # Re-price the named template to extract its legs.
        prop = price_strategy(body.template, symbol=sym, spot=spot, vix=current_vix)
        if prop is None:
            raise HTTPException(status_code=400, detail="unknown_template")
        # Convert StrategyProposal.legs (concrete strike+expiry) → LegSpec
        # using PCT_OFFSET anchor + the existing expiry. The resolver
        # will re-derive the same numbers at open time.
        for L in prop.legs:
            pct = ((float(L.strike) / float(spot)) - 1.0) * 100.0
            legs.append(LegSpec(
                # StrategyLeg exposes .action ("BUY"/"SELL") and no qty_lots;
                # proposals are 1-lot-per-leg defined-risk structures.
                side=OptionSide(str(L.action).lower()),
                option_type=OptionType(L.option_type),
                strike_anchor=StrikeAnchor.PCT_OFFSET,
                strike_offset=round(pct, 4),
                expiry=ExpiryAnchor.CURRENT_WEEK,
                qty_lots=int(getattr(L, "qty_lots", 1) or 1),
            ))
    else:
        raise HTTPException(status_code=422, detail="provide_template_or_legs")

    res = open_paper_option_position(
        supabase=get_supabase_admin(),
        user_id=user.user_id,
        underlying=sym,
        spot=float(spot),
        sigma=float(sigma),
        legs=legs,
        lots=body.lots,
        template_slug=template_slug,
        source="manual",
    )
    if not res.ok:
        raise HTTPException(status_code=400, detail=res.reason or "open_failed")

    return {
        "success": True,
        "position_id": res.position_id,
        "trade_id": res.trade_id,
        "net_premium": res.net_premium,
        "max_profit": res.max_profit,
        "max_loss": res.max_loss,
        "legs": res.legs,
    }


@router.get("/paper/positions")
async def paper_positions(
    user: UserTier = Depends(RequireFeature("fo_strategies")),
) -> Dict[str, Any]:
    """List the caller's open + recently-closed paper option positions
    with fresh mark-to-market on every open row.
    """
    from ..services.execution.paper_options_executor import mark_to_market

    sb = get_supabase_admin()
    rows = (
        sb.table("paper_option_positions")
        .select("*")
        .eq("user_id", user.user_id)
        .order("entry_at", desc=True)
        .limit(50)
        .execute()
        .data
        or []
    )
    # MTM open positions in-place. Closed positions return as-is.
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("status") == "open":
            try:
                mtm = mark_to_market(sb, r)
                r = {**r, "current_value": mtm.current_value,
                     "unrealized_pnl": mtm.unrealized_pnl}
            except Exception as exc:
                logger.debug("paper_options: mtm failed for %s: %s", r.get("id"), exc)
        legs = (
            sb.table("paper_option_legs")
            .select("*")
            .eq("position_id", r["id"])
            .execute()
            .data
            or []
        )
        out.append({**r, "legs": legs})
    return {"positions": out, "count": len(out)}


@router.post("/paper/{position_id}/close")
async def paper_close(
    position_id: str,
    user: UserTier = Depends(RequireFeature("fo_strategies")),
) -> Dict[str, Any]:
    """Manually close a paper option position at current marked prices."""
    from ..services.execution.paper_options_executor import close_paper_option_position
    res = close_paper_option_position(
        supabase=get_supabase_admin(),
        position_id=position_id,
        user_id=user.user_id,
        reason="manual",
        source="manual",
    )
    if not res.ok:
        raise HTTPException(status_code=400, detail=res.reason or "close_failed")
    return {
        "success": True,
        "position_id": res.position_id,
        "realized_pnl": res.realized_pnl,
        "realized_pnl_pct": res.realized_pnl_pct,
    }


# ============================================================================
# PR-AW.2 — Template backtest (no saved strategy required)
# ============================================================================


class BacktestRequest(BaseModel):
    """Backtest a named template against historical underlying data
    without first creating a user_strategies row.

    Mirrors POST /api/strategies/{id}/backtest but synthesises the
    Strategy from a template slug + symbol so users can iterate on
    recommendation cards before saving.
    """
    template: str = Field(..., description=(
        "bull_call_spread | bear_put_spread | iron_condor | "
        "long_straddle | short_strangle | iron_butterfly"
    ))
    symbol: str = Field("NIFTY")
    lookback_days: int = Field(default=180, ge=30, le=730)
    initial_capital: float = Field(default=100_000.0, gt=0)


@router.post("/backtest")
async def backtest_template(
    body: BacktestRequest,
    user: UserTier = Depends(RequireFeature("fo_strategies")),
) -> Dict[str, Any]:
    """Run the rule-based template through historical bars of the underlying.

    Returns the same shape as POST /api/strategies/{id}/backtest's full
    dict — the BacktestViewer component already knows how to render it.
    """
    from ..ai.strategy.dsl import (
        Strategy, InstrumentSegment, LegSpec, OptionSide, OptionType,
        StrikeAnchor, ExpiryAnchor, Universe, Timeframe,
        Condition, ConditionKind, PositionSize,
        RegimeFilter, StrategyMode,
    )
    from ..ai.strategy.options_backtest import run_options_backtest
    from ..data.market import get_market_data_provider

    sym = body.symbol.upper()
    regime_row = _load_latest_regime()
    spot = _spot_for(sym, regime_row)
    current_vix = float(regime_row.get("vix") or 15.0)

    # Use the recommender to materialize the template's legs at current
    # spot, then convert each concrete leg to a LegSpec(PCT_OFFSET)
    # anchor — same conversion the paper deploy endpoint uses.
    prop = price_strategy(body.template, symbol=sym, spot=spot, vix=current_vix)
    if prop is None:
        raise HTTPException(status_code=400, detail="unknown_template")

    legs: List[LegSpec] = []
    for L in prop.legs:
        pct = ((float(L.strike) / float(spot)) - 1.0) * 100.0
        legs.append(LegSpec(
            side=OptionSide(L.side.lower()),
            option_type=OptionType(L.option_type),
            strike_anchor=StrikeAnchor.PCT_OFFSET,
            strike_offset=round(pct, 4),
            expiry=ExpiryAnchor.CURRENT_WEEK,
            qty_lots=int(L.qty_lots),
        ))

    # Always-true entry + always-false exit so the backtester opens on
    # the first eligible bar after expiry close-out (each weekly cycle).
    # The Strategy.legs presence is what triggers the options branch
    # in the backtest engine; entry/exit conditions are unused in the
    # weekly-roll convention.
    placeholder_entry = Condition(
        kind=ConditionKind.INDICATOR,
        left="close", op="gt", right=0,
    )
    placeholder_exit = Condition(
        kind=ConditionKind.INDICATOR,
        left="close", op="lt", right=0,
    )

    try:
        strat = Strategy(
            name=f"{body.template} backtest",
            mode=StrategyMode.BACKTEST,
            instrument_segment=InstrumentSegment.OPTIONS,
            symbol=sym,
            universe=Universe.SINGLE,
            timeframe=Timeframe.DAILY,
            regime_filter=RegimeFilter.ANY,
            position_size=PositionSize.FIVE_PCT,
            entry=placeholder_entry,
            exit=placeholder_exit,
            legs=legs,
        )
    except Exception as exc:
        raise HTTPException(status_code=422,
                            detail={"error": "strategy_synth_failed", "message": str(exc)})

    # Load historical underlying bars
    period_str = (
        "1y" if body.lookback_days <= 252
        else "2y" if body.lookback_days <= 504
        else "5y"
    )
    try:
        ohlcv = get_market_data_provider().get_historical(sym, period=period_str, interval="1d")
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": "market_data_unavailable", "message": str(exc)},
        )
    if ohlcv is None or len(ohlcv) < 30:
        raise HTTPException(
            status_code=422,
            detail={"error": "insufficient_history", "message": f"got {0 if ohlcv is None else len(ohlcv)} bars"},
        )
    ohlcv.columns = [c.lower() for c in ohlcv.columns]

    try:
        result = run_options_backtest(
            strat, ohlcv,
            symbol=sym,
            initial_capital=body.initial_capital,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422,
                            detail={"error": "backtest_failed", "message": str(exc)})

    return result.to_full_dict()


# ============================================================================
# PR-AX — Option chain
# ============================================================================


@router.get("/chain/{symbol}")
async def options_chain(
    symbol: str,
    expiry: Optional[str] = None,
    user: UserTier = Depends(RequireFeature("fo_strategies")),
) -> Dict[str, Any]:
    """Live option chain for ``symbol`` via the user's connected broker.

    Params:
      ``expiry`` — optional ISO date (YYYY-MM-DD). If absent the broker
        returns its nearest expiry.

    Returns:
      ``{ symbol, expiry, rows: [{strike, option_type, ltp, bid, ask,
          oi, volume, iv, tradingsymbol}], source }``
      ``source`` is 'broker' on success or 'unavailable' when the
      caller has no connected broker (frontend can show a "Connect a
      broker to see live chain" empty state).
    """
    from ..services.execution.option_chain import get_option_chain

    sym = symbol.upper().strip()
    expiry_d: Optional[date] = None
    if expiry:
        try:
            expiry_d = date.fromisoformat(expiry)
        except ValueError:
            raise HTTPException(status_code=422, detail="invalid_expiry")

    rows = get_option_chain(
        get_supabase_admin(),
        user_id=user.user_id,
        symbol=sym,
        expiry=expiry_d,
    )
    if rows is None:
        return {"symbol": sym, "expiry": expiry, "rows": [], "source": "unavailable"}

    # Sort by strike (ascending), CE before PE at the same strike.
    rows_sorted = sorted(
        rows,
        key=lambda r: (r.strike, 0 if r.option_type == "CE" else 1),
    )

    # PR-AY — Enrich with IV + Greeks. Brokers return iv=0; we solve
    # implied vol from the LTP and derive delta/gamma/theta/vega per
    # row. Requires spot; fall back to recommender's spot estimate.
    regime_row = _load_latest_regime()
    spot = _spot_for(sym, regime_row)
    today_iso = date.today()
    from ..services.execution.options_greeks import enrich_chain_row

    out_rows: List[Dict[str, Any]] = []
    for r in rows_sorted:
        # Best-effort: skip Greeks calc if expiry isn't parseable.
        try:
            exp_d = date.fromisoformat(r.expiry[:10])
            days = max((exp_d - today_iso).days, 0)
        except Exception:
            days = 0

        greeks = enrich_chain_row(
            spot=spot, strike=r.strike, expiry_days=days,
            ltp=r.ltp, option_type=r.option_type,
        )
        out_rows.append({
            "strike": r.strike,
            "option_type": r.option_type,
            "expiry": r.expiry,
            "ltp": r.ltp,
            "bid": r.bid,
            "ask": r.ask,
            "oi": r.oi,
            "volume": r.volume,
            "iv": greeks.iv if greeks else None,
            "delta": greeks.delta if greeks else None,
            "gamma": greeks.gamma if greeks else None,
            "theta": greeks.theta if greeks else None,
            "vega": greeks.vega if greeks else None,
            "tradingsymbol": r.tradingsymbol,
        })

    return {
        "symbol": sym,
        "expiry": expiry,
        "spot": spot,
        "rows": out_rows,
        "source": "broker",
    }


# ============================================================================
# PR-BD — AI strategy suggestions (advisory; never auto-deploys)
# ============================================================================


class AISuggestRequest(BaseModel):
    """User's view + optional context for the AI to scope its pick."""
    prompt: str = Field(..., min_length=4, max_length=600,
                        description="Free-text user view, e.g. 'bearish on Nifty next week, want defined risk'")
    symbol: str = Field("NIFTY", description="NIFTY | BANKNIFTY | FINNIFTY")
    capital_inr: Optional[float] = Field(None, ge=10_000, le=10_000_000,
                                         description="Capital available; used for sizing recommendation.")
    include_portfolio: bool = Field(default=False, description=(
        "When true, the user's open equity + option positions are loaded "
        "and their net delta exposure is injected into the prompt. "
        "Enables 'hedge my book' style suggestions."
    ))
    focus_symbol: Optional[str] = Field(default=None, description=(
        "PR-BF.1 — When set (with include_portfolio=true), the prompt "
        "focuses the AI on hedging ONLY this underlying's exposure "
        "instead of the net book. Useful for 'hedge my RELIANCE' style "
        "asks. Falls back to net delta if the symbol has no position."
    ))


@router.post("/ai-suggest")
async def ai_suggest_strategy(
    body: AISuggestRequest,
    user: UserTier = Depends(RequireFeature("fo_strategies")),
    _cap: UserTier = Depends(enforce_llm_cap("fno_advisor")),
) -> Dict[str, Any]:
    """LLM-backed strategy advisor (OpenRouter open model).

    Locked memory policy: LLMs never gate trades or auto-deploy. This
    endpoint RETURNS A SUGGESTION only; the user still has to click
    Deploy in the F&O panel for the runner / paper executor to act.

    The model is constrained to pick from the existing rule-based
    template registry (Bull Call Spread, Bear Put Spread, Iron Condor,
    Long Straddle, Short Strangle, Iron Butterfly) so the output plugs
    straight into the existing paper/backtest endpoints. Custom-leg
    suggestions are allowed but flagged.

    Context injected into the prompt:
      - Current symbol spot + recommender's pre-resolved view
      - HMM regime + VIX direction
      - Optional capital figure for size advice
    """
    from ..ai.agents.llm import extract_json, llm_for
    from ..ai.agents.response_cache import cache_get, cache_set, seconds_to_ist_eod

    sym = body.symbol.upper().strip()
    regime_row = _load_latest_regime()
    spot = _spot_for(sym, regime_row)
    current_vix = float(regime_row.get("vix") or 15.0)
    regime_name = str(regime_row.get("regime") or "sideways").lower()
    vix_5d = regime_row.get("vix_5d_mean")
    vix_dir = "rising" if vix_5d and current_vix > vix_5d + 0.5 \
        else "falling" if vix_5d and current_vix < vix_5d - 0.5 \
        else "stable"

    # Result cache — the suggestion is a function of (symbol, the user's
    # normalized view, regime, VIX direction) plus the sizing/portfolio
    # knobs that change the prompt. Output is day-stable (regime + VIX
    # direction only refresh daily) → IST-EOD TTL. We key on vix_dir (a
    # bucket) rather than the raw VIX float so intra-day VIX wobble inside
    # the same direction reuses one entry. include_portfolio rebuilds the
    # prompt from the live book, so it is intentionally NOT cached.
    _prompt_norm = " ".join(body.prompt.lower().split())
    ck = (
        "fo:ai_suggest:"
        f"{sym}:{regime_name}:{vix_dir}:"
        f"{int(body.capital_inr) if body.capital_inr else 'na'}:"
        f"{hashlib.sha256(_prompt_norm.encode()).hexdigest()[:16]}"
    )
    if not body.include_portfolio:
        cached = cache_get(ck)
        if cached:
            return cached

    # All allowed templates with one-line descriptions so the model picks
    # the right tool rather than inventing strategy names.
    allowed_templates = [
        ("bull_call_spread", "Debit · Bullish · Defined risk · profits if spot above upper strike at expiry."),
        ("bear_put_spread", "Debit · Bearish · Defined risk · profits if spot below lower strike at expiry."),
        ("iron_condor", "Credit · Range-bound · Defined risk · profits if spot stays inside short strikes."),
        ("long_straddle", "Debit · Volatility long · Profits on a large move either direction (often before events)."),
        ("short_strangle", "Credit · Range-bound · UNBOUNDED RISK · profits if spot stays inside a wide band."),
        ("iron_butterfly", "Credit · Pinning · Defined risk · tighter than condor; max profit at body strike."),
    ]
    templates_block = "\n".join(f"  - {slug}: {desc}" for slug, desc in allowed_templates)

    system_prompt = (
        "You are Quant X's F&O strategy advisor. The user describes their "
        "market view; you recommend ONE multi-leg options structure that "
        "fits that view AND current market conditions.\n\n"
        "Hard rules:\n"
        " - Output JSON only — no prose, no code fences.\n"
        " - Pick exactly one ``template`` from the registry below.\n"
        " - Reasoning: 2-3 short sentences. NUMBER-FIRST style — cite the\n"
        "   VIX level / regime / spot. No emojis. No 'as an AI'.\n"
        " - Never recommend short_strangle unless the user EXPLICITLY says\n"
        "   they accept unbounded risk OR the regime is strongly\n"
        "   range-bound + VIX falling.\n"
        " - In a bear regime, avoid credit spreads selling premium against\n"
        "   the trend (no bull put spreads). Prefer debit puts or condors.\n"
        " - lots_suggestion must be ≤ capital_inr / 100,000 (rough sizing).\n"
        "   If capital_inr is null, default lots_suggestion = 1.\n"
    )

    # PR-BE — Optional portfolio-aware context. When the user asks for
    # a hedge or wants to factor their existing book into the suggestion,
    # we read positions and compute net delta exposure with the same
    # Greeks helper the chain Greeks view uses.
    # PR-BF.1 — focus_symbol narrows the hedge to a single underlying.
    portfolio_block = ""
    portfolio_ctx_summary: Optional[Dict[str, Any]] = None
    focus_block = ""
    if body.include_portfolio:
        try:
            from ..services.portfolio.portfolio_context import compute_user_book_context
            pctx = compute_user_book_context(
                get_supabase_admin(), user_id=user.user_id,
            )
            portfolio_block = pctx.prompt_block
            portfolio_ctx_summary = {
                "has_positions": pctx.has_positions,
                "equity_delta_inr": pctx.equity_delta_inr,
                "option_delta_inr": sum(p["delta_inr"] for p in pctx.option_positions),
                "net_delta_inr": pctx.net_delta_inr,
                "equity_count": len(pctx.equity_positions),
                "options_count": len(pctx.option_positions),
                "by_symbol": pctx.by_symbol,
            }
            if body.focus_symbol:
                focus_sym = body.focus_symbol.upper().strip()
                sym_data = pctx.by_symbol.get(focus_sym)
                if sym_data:
                    focus_block = (
                        f"\nFOCUS: hedge {focus_sym} specifically. "
                        f"Current exposure: ₹{sym_data['total_delta_inr']:,.0f} "
                        f"({sym_data['bias']}, equity ₹{sym_data['equity_delta_inr']:,.0f} "
                        f"+ options ₹{sym_data['option_delta_inr']:,.0f}). "
                        f"Size the hedge to offset this exposure, not the whole book."
                    )
                else:
                    focus_block = (
                        f"\nFOCUS requested: {focus_sym}, but the user has NO open "
                        f"position in this underlying. Suggest a directional play on "
                        f"{focus_sym} based on the user's view instead of a hedge."
                    )
        except Exception as exc:
            logger.debug("ai_suggest: portfolio context skipped: %s", exc)

    user_prompt = (
        f"User view: {body.prompt}\n"
        f"Symbol: {sym}\n"
        f"Spot: ₹{spot:,.0f}\n"
        f"Current regime: {regime_name}\n"
        f"Current VIX: {current_vix:.2f} ({vix_dir} vs 5-day mean)\n"
        f"Capital available: {body.capital_inr if body.capital_inr else 'not specified'}\n"
        + (f"\n{portfolio_block}\n" if portfolio_block else "")
        + (focus_block if focus_block else "")
        + (
            "\nHedging directive: the user enabled portfolio context. If their\n"
            "NET DELTA is materially long (+₹50k+), favour STRUCTURES that ADD\n"
            "negative delta (debit puts / bear spreads / covered calls). If\n"
            "materially short, favour positive delta (debit calls / bull\n"
            "spreads). For near-flat books, prefer non-directional plays\n"
            "(condors / straddles). Mention the existing exposure number in\n"
            "your reasoning.\n"
            if portfolio_block else ""
        )
        + f"\nAllowed templates (pick ONE):\n{templates_block}\n\n"
        'Return JSON:\n'
        '{\n'
        '  "template": "<slug from registry above>",\n'
        '  "lots_suggestion": <int>,\n'
        '  "reasoning": "<2-3 sentences explaining why this template fits>",\n'
        '  "expected_outcome": "<one line, e.g. \'profit ₹X if spot ≥ Y at expiry\'>",\n'
        '  "risk_summary": "<one line max loss + worst case>",\n'
        '  "confidence": <0.0 to 1.0>\n'
        '}\n'
    )

    llm = llm_for("fno_advisor")
    if not llm.enabled:
        raise HTTPException(
            status_code=503,
            detail={"error": "model_unavailable", "message": "OPENROUTER_API_KEY not configured"},
        )

    try:
        raw = await llm.complete(
            user_prompt,
            system=system_prompt,
            temperature=0.2,
            top_p=0.5,
            user_id=user.user_id,
            feature="fo_strategies_ai_suggest",
            metadata={"symbol": sym, "regime": regime_name},
        )
    except Exception as exc:
        logger.error("ai_suggest llm call failed: %s", exc)
        raise HTTPException(status_code=502, detail="model_error")

    parsed = extract_json(raw)
    template = str(parsed.get("template", "")).strip()
    allowed_slugs = {slug for slug, _ in allowed_templates}
    if template not in allowed_slugs:
        # Fallback to a safe default rather than rejecting the call —
        # users hate "AI returned nothing".
        template = (
            "iron_condor" if regime_name == "sideways"
            else "bull_call_spread" if regime_name == "bull"
            else "bear_put_spread"
        )

    lots = int(parsed.get("lots_suggestion") or 1)
    if body.capital_inr:
        lots = max(1, min(lots, int(body.capital_inr / 100_000)))
    else:
        lots = max(1, min(lots, 5))   # safety cap when capital unknown

    # Pre-price the suggested template so the UI can show breakeven /
    # max profit / max loss WITHOUT making the user click Deploy first.
    suggested_prop = None
    try:
        prop = price_strategy(template, symbol=sym, spot=spot, vix=current_vix)
        if prop is not None:
            suggested_prop = _proposal_to_dict(prop)
    except Exception as exc:
        logger.debug("ai_suggest: pre-price failed: %s", exc)

    result = {
        "template": template,
        "symbol": sym,
        "spot": spot,
        "lots_suggestion": lots,
        "reasoning": str(parsed.get("reasoning") or "")[:600],
        "expected_outcome": str(parsed.get("expected_outcome") or "")[:200],
        "risk_summary": str(parsed.get("risk_summary") or "")[:200],
        "confidence": float(parsed.get("confidence") or 0.5),
        "context": {
            "regime": regime_name,
            "vix": current_vix,
            "vix_direction": vix_dir,
        },
        "portfolio_context": portfolio_ctx_summary,
        "proposal": suggested_prop,  # full pre-priced template; null if pricing failed
    }
    # Cache only the real, portfolio-independent success path (skip the
    # live-book path, which is keyed to the user's positions, not day-stable).
    if not body.include_portfolio:
        cache_set(ck, result, ttl_seconds=seconds_to_ist_eod(),
                  surface="fo_ai_suggest", model="")
    return result


# ============================================================================
# PR-BB — Vol cone (realised vol percentiles vs current IV)
# ============================================================================


@router.get("/vol-cone/{symbol}")
async def vol_cone(
    symbol: str,
    user: UserTier = Depends(RequireFeature("fo_strategies")),
) -> Dict[str, Any]:
    """Historical realised-vol cone for the symbol vs current ATM IV.

    For each window in [7, 14, 21, 30, 60, 90] trading days, compute
    every rolling realised vol over the past year. Report p10, p25, p50,
    p75, p90 of that distribution per window. Plot the current ATM IV
    from the broker chain on top — if it sits above p90, options are
    rich (priced for a tail event); below p10, cheap (mean-reversion
    candidate for vol buyers).
    """
    import math
    import numpy as np
    from ..ai.strategy.dsl import ExpiryAnchor
    from ..ai.strategy.options_resolver import resolve_expiry
    from ..data.market import get_market_data_provider
    from ..services.execution.option_chain import get_option_chain
    from ..services.execution.options_greeks import enrich_chain_row

    sym = symbol.upper().strip()
    regime_row = _load_latest_regime()
    spot = _spot_for(sym, regime_row)
    today = date.today()

    # 1 year of bars is enough to build a stable distribution at the
    # 90-day window. yfinance / Kite both serve daily 1y reliably.
    try:
        bars = get_market_data_provider().get_historical(sym, period="1y", interval="1d")
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"error": "market_data_unavailable",
                                                     "message": str(exc)})
    if bars is None or len(bars) < 100:
        raise HTTPException(status_code=422,
                            detail={"error": "insufficient_history"})
    bars.columns = [c.lower() for c in bars.columns]
    closes = bars["close"].dropna()
    log_ret = np.log(closes / closes.shift(1)).dropna()

    windows = [7, 14, 21, 30, 60, 90]
    out_windows: List[Dict[str, Any]] = []
    for w in windows:
        if len(log_ret) < w + 5:
            continue
        # Rolling annualised realised vol (252 trading days)
        rv = log_ret.rolling(w).std().dropna() * math.sqrt(252)
        if len(rv) < 5:
            continue
        out_windows.append({
            "window_days": w,
            "p10": round(float(np.percentile(rv, 10)), 4),
            "p25": round(float(np.percentile(rv, 25)), 4),
            "p50": round(float(np.percentile(rv, 50)), 4),
            "p75": round(float(np.percentile(rv, 75)), 4),
            "p90": round(float(np.percentile(rv, 90)), 4),
            "current_rv": round(float(rv.iloc[-1]), 4),
            "samples": int(len(rv)),
        })

    # Current ATM IV per expiry (so the UI can drop a dot per window):
    # bucket the available expiries into the closest window.
    current_ivs: List[Dict[str, Any]] = []
    for anchor in (ExpiryAnchor.CURRENT_WEEK, ExpiryAnchor.NEXT_WEEK,
                   ExpiryAnchor.CURRENT_MONTH, ExpiryAnchor.NEXT_MONTH):
        try:
            expiry_d = resolve_expiry(anchor, sym, today=today)
        except Exception:
            continue
        rows = get_option_chain(
            get_supabase_admin(), user_id=user.user_id,
            symbol=sym, expiry=expiry_d,
        )
        if not rows:
            continue
        # Nearest-strike ATM IV
        by_strike: Dict[float, Dict[str, Any]] = {}
        for r in rows:
            by_strike.setdefault(r.strike, {})[r.option_type] = r
        atm = min(by_strike.keys(), key=lambda k: abs(k - spot))
        days = max((expiry_d - today).days, 1)
        ivs: List[float] = []
        for opt in ("CE", "PE"):
            r = by_strike[atm].get(opt)
            if r and r.ltp > 0:
                g = enrich_chain_row(
                    spot=spot, strike=atm, expiry_days=days,
                    ltp=r.ltp, option_type=opt,
                )
                if g and g.iv > 0:
                    ivs.append(g.iv)
        if ivs:
            current_ivs.append({
                "expiry": expiry_d.isoformat(),
                "days": days,
                "iv": round(sum(ivs) / len(ivs), 4),
            })

    # Bucket each current IV into its nearest cone window for plotting
    for iv_row in current_ivs:
        if not out_windows:
            continue
        nearest_w = min(out_windows, key=lambda w: abs(w["window_days"] - iv_row["days"]))
        iv_row["window_days"] = nearest_w["window_days"]

    return {
        "symbol": sym,
        "spot": spot,
        "windows": out_windows,
        "current_ivs": current_ivs,
        "source": "broker" if current_ivs else "rv_only",
    }


# ============================================================================
# PR-BA — Term structure (front-month vs back-month ATM IV)
# ============================================================================


@router.get("/term-structure/{symbol}")
async def options_term_structure(
    symbol: str,
    user: UserTier = Depends(RequireFeature("fo_strategies")),
) -> Dict[str, Any]:
    """ATM IV per available expiry — the term structure curve.

    Pulls chains for the four standard anchors (current week, next
    week, current month, next month), computes ATM IV per chain
    (via the same Greeks solver as /chain), and returns a list
    sorted ascending by days-to-expiry.

    A flat curve = market sees similar near + far vol.
    Backwardation (front > back) = event risk priced into front month
    (earnings, RBI / Fed, expiry effect).
    Contango (back > front) = normal — far month has more event windows
    so longer vega exposure commands a premium.
    """
    from ..ai.strategy.dsl import ExpiryAnchor
    from ..ai.strategy.options_resolver import resolve_expiry
    from ..services.execution.option_chain import get_option_chain
    from ..services.execution.options_greeks import enrich_chain_row

    sym = symbol.upper().strip()
    regime_row = _load_latest_regime()
    spot = _spot_for(sym, regime_row)
    today = date.today()

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for anchor in (
        ExpiryAnchor.CURRENT_WEEK,
        ExpiryAnchor.NEXT_WEEK,
        ExpiryAnchor.CURRENT_MONTH,
        ExpiryAnchor.NEXT_MONTH,
    ):
        try:
            expiry_d = resolve_expiry(anchor, sym, today=today)
        except Exception:
            continue
        if expiry_d.isoformat() in seen:
            # Current-week == current-month in the final week of the
            # month; skip the duplicate.
            continue
        seen.add(expiry_d.isoformat())

        rows = get_option_chain(
            get_supabase_admin(),
            user_id=user.user_id,
            symbol=sym,
            expiry=expiry_d,
        )
        if rows is None:
            # No broker — can't compute term structure. Bail early
            # rather than return half-empty data.
            return {
                "symbol": sym, "spot": spot, "source": "unavailable",
                "expiries": [],
            }
        if not rows:
            continue

        # Find ATM IV: nearest strike to spot, average of CE + PE if both
        # solvable, else the available one.
        by_strike: Dict[float, Dict[str, Any]] = {}
        for r in rows:
            d = by_strike.setdefault(r.strike, {})
            d[r.option_type] = r
        atm_strike = min(by_strike.keys(), key=lambda k: abs(k - spot))
        atm_pair = by_strike[atm_strike]
        days = max((expiry_d - today).days, 1)
        ivs: List[float] = []
        for opt in ("CE", "PE"):
            r = atm_pair.get(opt)
            if not r or r.ltp <= 0:
                continue
            g = enrich_chain_row(
                spot=spot, strike=atm_strike, expiry_days=days,
                ltp=r.ltp, option_type=opt,
            )
            if g and g.iv > 0:
                ivs.append(g.iv)
        if not ivs:
            continue

        out.append({
            "anchor": anchor.value,
            "expiry": expiry_d.isoformat(),
            "days_to_expiry": days,
            "atm_strike": atm_strike,
            "atm_iv": round(sum(ivs) / len(ivs), 4),
            "ce_iv": round(ivs[0], 4) if "CE" in atm_pair and atm_pair["CE"].ltp > 0 else None,
            "pe_iv": round(ivs[-1], 4) if "PE" in atm_pair and atm_pair["PE"].ltp > 0 else None,
        })

    out.sort(key=lambda r: r["days_to_expiry"])

    # Curve shape classification
    shape = "flat"
    if len(out) >= 2:
        front = out[0]["atm_iv"]
        back = out[-1]["atm_iv"]
        spread_pts = (back - front) * 100  # convert to vol points
        if spread_pts < -1.0:
            shape = "backwardation"  # front > back by 1+ vol point
        elif spread_pts > 1.0:
            shape = "contango"

    return {
        "symbol": sym,
        "spot": spot,
        "source": "broker",
        "expiries": out,
        "shape": shape,
    }


__all__ = ["router"]
