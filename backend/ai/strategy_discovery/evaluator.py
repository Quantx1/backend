"""Candidate evaluator — runs the existing backtester, scores the result,
and decomposes performance by regime.

A "candidate" is a `Strategy` DSL document drawn from a search space.
For each candidate we:
  1. Pick the right backtester (equity vs F&O) by `instrument_segment`.
  2. Run it on the supplied bars (single symbol or first-symbol-of-universe
     for v1; universe sweep is PR-G2).
  3. Compute a composite score that rewards Sharpe + profit factor and
     penalises drawdown + low trade count + single-regime over-fit.
  4. Slice trades by entry-bar regime and compute per-regime sub-scores.

The composite score formula is deliberately explicit so the user can
inspect why one strategy beat another in the Discovered tab.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.ai.strategy.backtest import (
    DSLBacktestResult,
    run_dsl_backtest,
)
from backend.ai.strategy.dsl import InstrumentSegment, Strategy
from backend.ai.strategy.interpreter import EngineSignals
from backend.ai.strategy.options_backtest import (
    OptionsBacktestResult,
    run_options_backtest,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Scoring constants — tune carefully; surface in the Discovered UI so
# users see how strategies were ranked.
# ─────────────────────────────────────────────────────────────────────

MIN_TRADES_FOR_SCORING = 8         # below this, the strategy is undercooked
SHARPE_WEIGHT = 1.0                # primary Sharpe contribution
PROFIT_FACTOR_WEIGHT = 0.4         # secondary — caps at ln(PF) so PF=10 → +0.92
DRAWDOWN_PENALTY_WEIGHT = 0.5      # subtract |DD%| / 100 * weight
TRADE_COUNT_BONUS_CAP = 0.3        # +0.3 max for active strategies (n=50+)
REGIME_CONCENTRATION_PENALTY = 0.4  # subtract when >70% of trades hit in one regime


@dataclass
class RegimeBreakdown:
    """Per-regime slice of strategy performance.

    `score` here is the same composite formula re-applied to just the
    trades that entered in that regime. Lets the UI flag "this only works
    in bull markets — be cautious".
    """
    bull: float = 0.0
    sideways: float = 0.0
    bear: float = 0.0
    bull_trades: int = 0
    sideways_trades: int = 0
    bear_trades: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bull": round(self.bull, 4),
            "sideways": round(self.sideways, 4),
            "bear": round(self.bear, 4),
            "bull_trades": self.bull_trades,
            "sideways_trades": self.sideways_trades,
            "bear_trades": self.bear_trades,
        }


@dataclass
class CandidateScore:
    """Output of evaluating one Strategy candidate against one symbol."""
    strategy: Strategy
    score: float                           # composite — higher is better
    sharpe: float
    calmar: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_return_pct: float
    trade_count: int
    avg_hold_days: float
    regime_breakdown: RegimeBreakdown = field(default_factory=RegimeBreakdown)
    walk_forward: List[Dict[str, Any]] = field(default_factory=list)
    viable: bool = False                   # passed min_trades + sharpe>0 gate
    error: Optional[str] = None            # set when backtest itself failed
    label: str = ""                        # human-readable summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "score": round(self.score, 6),
            "sharpe": round(self.sharpe, 4),
            "calmar": round(self.calmar, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "total_return_pct": round(self.total_return_pct, 4),
            "trade_count": self.trade_count,
            "avg_hold_days": round(self.avg_hold_days, 2),
            "regime_breakdown": self.regime_breakdown.to_dict(),
            "walk_forward": self.walk_forward,
            "viable": self.viable,
            "error": self.error,
        }


# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────


def _composite_score(
    *,
    sharpe: float,
    profit_factor: float,
    max_drawdown_pct: float,
    trade_count: int,
    regime_breakdown: RegimeBreakdown,
) -> float:
    """Blend the components into a single comparable number.

    Formula:
        score = SHARPE_WEIGHT * sharpe
              + PROFIT_FACTOR_WEIGHT * ln(max(profit_factor, 0.01))
              - DRAWDOWN_PENALTY_WEIGHT * |max_drawdown_pct| / 100
              + min(TRADE_COUNT_BONUS_CAP, log(trade_count)/log(200) * cap)
              - REGIME_CONCENTRATION_PENALTY * concentration

    where concentration is 0 if trades are spread across regimes and 1
    if all trades hit a single regime.
    """
    pf_term = PROFIT_FACTOR_WEIGHT * math.log(max(profit_factor, 0.01))
    dd_penalty = DRAWDOWN_PENALTY_WEIGHT * abs(max_drawdown_pct) / 100.0
    # Trade count bonus — saturates around n=200
    if trade_count <= 1:
        count_bonus = 0.0
    else:
        count_bonus = min(
            TRADE_COUNT_BONUS_CAP,
            math.log(trade_count) / math.log(200) * TRADE_COUNT_BONUS_CAP,
        )

    # Regime concentration penalty: 0 = perfectly diversified, 1 = one regime
    total_regime_trades = (
        regime_breakdown.bull_trades
        + regime_breakdown.sideways_trades
        + regime_breakdown.bear_trades
    )
    if total_regime_trades > 0:
        max_share = max(
            regime_breakdown.bull_trades,
            regime_breakdown.sideways_trades,
            regime_breakdown.bear_trades,
        ) / total_regime_trades
        # Penalty kicks in over 70% concentration; max penalty at 100%.
        concentration = max(0.0, (max_share - 0.7) / 0.3)
    else:
        concentration = 0.0
    regime_penalty = REGIME_CONCENTRATION_PENALTY * concentration

    return (
        SHARPE_WEIGHT * sharpe
        + pf_term
        - dd_penalty
        + count_bonus
        - regime_penalty
    )


def _calmar(total_return_pct: float, max_drawdown_pct: float) -> float:
    """Calmar = annualized return / |max DD|. We don't annualize here
    (caller may, given window length); just total-return/DD."""
    dd = abs(max_drawdown_pct)
    if dd < 0.01:
        return 0.0
    return total_return_pct / dd


def _regime_breakdown_for_equity(
    trades: List[Any],
    engine_signals_by_date: Optional[Dict[pd.Timestamp, EngineSignals]],
) -> RegimeBreakdown:
    """Slice trades by entry-bar regime."""
    rb = RegimeBreakdown()
    if not engine_signals_by_date:
        return rb

    # Map regime → bucket of pnl%s
    bucket: Dict[str, List[float]] = {"bull": [], "sideways": [], "bear": []}
    for t in trades:
        try:
            entry_ts = pd.Timestamp(t.entry_date)
            # Find nearest engine signal at or before entry
            es = engine_signals_by_date.get(entry_ts)
            if es is None:
                # Try the date-only key (signals indexed by date, trades
                # may have datetime); pandas timestamp equality is exact,
                # so fall back to a linear scan over the date.
                for ts, sig in engine_signals_by_date.items():
                    if pd.Timestamp(ts).normalize() == entry_ts.normalize():
                        es = sig
                        break
            if es is None or es.regime_label is None:
                continue
            r = es.regime_label.lower()
            if r in bucket:
                bucket[r].append(t.net_pnl_pct)
        except Exception:
            continue

    rb.bull_trades = len(bucket["bull"])
    rb.sideways_trades = len(bucket["sideways"])
    rb.bear_trades = len(bucket["bear"])
    # Bucket score = mean P&L% / volatility-of-P&L%. Simplified Sharpe.
    for regime, pnls in bucket.items():
        if len(pnls) < 2:
            continue
        mean = sum(pnls) / len(pnls)
        var = sum((p - mean) ** 2 for p in pnls) / len(pnls)
        std = math.sqrt(var) if var > 0 else 1.0
        setattr(rb, regime, mean / std)
    return rb


def _label_for(strategy: Strategy, sharpe: float) -> str:
    """Compose a one-liner that summarises what this strategy is."""
    segment = strategy.instrument_segment.value.lower()
    if segment == "options":
        legs = len(strategy.legs or [])
        return (
            f"{strategy.name}  ·  {legs}-leg "
            f"{'CR' if (strategy.legs and any(leg.side.value == 'sell' for leg in strategy.legs)) else 'DR'}"
            f"  ·  Sharpe {sharpe:.2f}"
        )
    return (
        f"{strategy.name}  ·  {strategy.timeframe.value}"
        f"  ·  Sharpe {sharpe:.2f}  ·  SL {strategy.stop_loss_pct}%"
    )


# ─────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────


def evaluate_candidate(
    strategy: Strategy,
    bars: pd.DataFrame,
    *,
    symbol: str,
    initial_capital: float = 100_000.0,
    engine_signals_by_date: Optional[Dict[pd.Timestamp, EngineSignals]] = None,
) -> CandidateScore:
    """Backtest a single candidate against the supplied bars and score it.

    Returns a `CandidateScore` even on backtest failure — failures are
    surfaced via the `error` field so the runner can persist them for
    debugging instead of silently dropping candidates.
    """
    try:
        if strategy.instrument_segment == InstrumentSegment.OPTIONS:
            result: OptionsBacktestResult = run_options_backtest(
                strategy, bars,
                symbol=symbol,
                initial_capital=initial_capital,
                engine_signals_by_date=engine_signals_by_date,
            )
            sharpe = result.sharpe_ratio
            pf = result.profit_factor
            max_dd = result.max_drawdown_pct
            win_rate = result.win_rate
            total_return = result.total_return_pct
            trade_count = result.total_trades
            # Options trades don't have hold_days as a field; approximate
            # from equity_curve length / trade_count.
            avg_hold = (
                len(result.equity_curve) / max(1, trade_count)
                if result.equity_curve else 0.0
            )
            rb = RegimeBreakdown()  # F&O regime breakdown is PR-G2 enhancement
        else:
            r: DSLBacktestResult = run_dsl_backtest(
                strategy, bars,
                symbol=symbol,
                initial_capital=initial_capital,
                engine_signals_by_date=engine_signals_by_date,
            )
            sharpe = r.sharpe_ratio
            pf = r.profit_factor
            max_dd = r.max_drawdown_pct
            win_rate = r.win_rate
            total_return = r.total_return_pct
            trade_count = r.total_trades
            avg_hold = r.avg_hold_days
            rb = _regime_breakdown_for_equity(r.trades, engine_signals_by_date)

        viable = trade_count >= MIN_TRADES_FOR_SCORING and sharpe > 0
        score = _composite_score(
            sharpe=sharpe,
            profit_factor=pf,
            max_drawdown_pct=max_dd,
            trade_count=trade_count,
            regime_breakdown=rb,
        )

        return CandidateScore(
            strategy=strategy,
            score=score,
            sharpe=sharpe,
            calmar=_calmar(total_return, max_dd),
            max_drawdown_pct=max_dd,
            win_rate=win_rate,
            profit_factor=pf,
            total_return_pct=total_return,
            trade_count=trade_count,
            avg_hold_days=avg_hold,
            regime_breakdown=rb,
            viable=viable,
            label=_label_for(strategy, sharpe),
        )

    except Exception as e:
        logger.warning(
            "evaluate_candidate failed for %s on %s: %s",
            strategy.name, symbol, e,
        )
        return CandidateScore(
            strategy=strategy,
            score=-1e9,                         # ranked dead last
            sharpe=0.0, calmar=0.0,
            max_drawdown_pct=0.0, win_rate=0.0, profit_factor=0.0,
            total_return_pct=0.0, trade_count=0, avg_hold_days=0.0,
            viable=False,
            error=str(e)[:500],
            label=f"{strategy.name}  ·  ERROR",
        )
