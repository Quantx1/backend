"""
DSL backtest harness — PR-G per v2 design spec §7.4.

Runs a Strategy DSL bar-by-bar against historical OHLCV. Returns a
deterministic summary the UI can render (equity curve + win rate + max
DD + Sharpe + profit factor + trade list).

Performance target: ≤10 sec for a 90-day daily backtest on a single
symbol. The bottleneck is per-bar indicator computation; we mitigate
via the interpreter's InterpreterContext._cache (one indicator value
per bar, shared across all Condition children).

Why a new harness instead of reusing ml/backtest/engine.py:
  - The legacy engine takes a BaseStrategy subclass (hand-coded
    Python) — not compatible with the JSON DSL. Bridging would be
    more work than a clean reimplementation.
  - Cost model + risk gates are simple enough (slippage + brokerage +
    STT + per-stock cap) that we re-implement here in 80 lines.
  - The legacy engine carries portfolio-level complexity (heat caps,
    sector rotation) we don't need for a per-strategy backtest.

Cost constants match the legacy engine + the institutional eval gate
in ml/eval/cost_model.py — DO NOT drift these without updating the
production AutoPilot + signal evaluation that share the same math.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .dsl import Strategy
from .indicators import MIN_LOOKBACK
from .interpreter import (
    EngineSignals,
    InterpreterContext,
    evaluate_entry,
    evaluate_exit,
)

logger = logging.getLogger(__name__)


# NSE-realistic cost constants (round-trip ~18 bps before market impact).
# Match ml/backtest/engine.py defaults so DSL backtests and legacy
# backtests are comparable apples-to-apples.
SLIPPAGE_PCT = 0.05      # 0.05% slippage on entry + exit (one side each)
BROKERAGE_PCT = 0.03     # 0.03% brokerage per side
STT_PCT = 0.10           # 0.10% STT on sell only
DEFAULT_INITIAL_CAPITAL = 500_000.0


def _fmt_ts(ts: Any) -> str:
    """Timeline label for a bar. Full timestamp for INTRADAY bars (so the
    walk-forward can segment them — many 5m bars share a calendar date), but
    date-only for daily so display stays clean + back-compatible. Both trades
    and the equity curve use this so they sort + match consistently.
    """
    try:
        if ts.hour or ts.minute or ts.second:
            return ts.strftime("%Y-%m-%d %H:%M:%S")
    except AttributeError:
        pass
    return ts.strftime("%Y-%m-%d")


@dataclass
class DSLTrade:
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    quantity: int
    direction: str          # "LONG" — short-selling not in v1 DSL
    hold_days: int
    gross_pnl_pct: float
    net_pnl_pct: float      # after slippage + brokerage + STT
    net_pnl_amount: float
    exit_reason: str        # "exit_condition" | "stop_loss" | "take_profit" | "end_of_data"


@dataclass
class DSLBacktestResult:
    symbol: str
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_hold_days: float = 0.0
    trades: List[DSLTrade] = field(default_factory=list)
    equity_curve: List[Dict[str, float]] = field(default_factory=list)  # [{date, equity}]

    def to_summary_dict(self) -> Dict[str, Any]:
        """Trimmed dict suitable for user_strategies.last_backtest JSONB."""
        return {
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "total_return_pct": round(self.total_return_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_hold_days": round(self.avg_hold_days, 2),
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Full payload for the /backtest API response — includes trades + curve."""
        return {
            **self.to_summary_dict(),
            "trades": [t.__dict__ for t in self.trades],
            "equity_curve": self.equity_curve,
        }


# ─────────────────────────────────────────────────────────────────────
# Cost model
# ─────────────────────────────────────────────────────────────────────


def _apply_costs(entry_price: float, exit_price: float, quantity: int) -> tuple[float, float]:
    """Return (gross_pnl_pct, net_pnl_amount) after slippage + brokerage + STT.

    Slippage hits both sides (worse entry + worse exit). Brokerage
    each side. STT only on the sell leg per SEBI 2026 rules.
    """
    # Slippage: pay more on entry, get less on exit
    entry_eff = entry_price * (1 + SLIPPAGE_PCT / 100)
    exit_eff = exit_price * (1 - SLIPPAGE_PCT / 100)
    gross_pnl_pct = (exit_eff / entry_eff - 1) * 100
    # Per-share fees
    entry_value = entry_eff * quantity
    exit_value = exit_eff * quantity
    fees = (
        entry_value * (BROKERAGE_PCT / 100)
        + exit_value * (BROKERAGE_PCT / 100)
        + exit_value * (STT_PCT / 100)
    )
    net_pnl_amount = (exit_value - entry_value) - fees
    return gross_pnl_pct, net_pnl_amount


# ─────────────────────────────────────────────────────────────────────
# Main backtest loop — single symbol, single position
# ─────────────────────────────────────────────────────────────────────


def run_dsl_backtest(
    strategy: Strategy,
    ohlcv: pd.DataFrame,
    *,
    symbol: str,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    engine_signals_by_date: Optional[Dict[pd.Timestamp, EngineSignals]] = None,
    periods_per_year: float = 252.0,
) -> DSLBacktestResult:
    """Bar-by-bar simulator.

    Args:
        strategy: validated Strategy DSL
        ohlcv: lowercase OHLCV DataFrame, datetime-indexed, oldest first.
               Must include at least MIN_LOOKBACK + 30 bars for stats to
               be meaningful.
        symbol: display label (NSE ticker)
        initial_capital: starting equity in INR
        engine_signals_by_date: optional pre-computed engine outputs per
               bar (e.g. backfilled regime, sentiment). If None, every
               engine_signal condition evaluates to False (defensive).

    Returns:
        DSLBacktestResult with trades + equity curve + stats.
    """
    if ohlcv is None or len(ohlcv) < MIN_LOOKBACK + 10:
        raise ValueError(
            f"insufficient bars for backtest: got {0 if ohlcv is None else len(ohlcv)}, "
            f"need >= {MIN_LOOKBACK + 10}",
        )

    # Validate OHLCV columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}")

    df = ohlcv.copy().sort_index()
    n = len(df)

    # Capital allocation per trade
    pct_of_cap = (
        strategy.position_size.value / 100
        if strategy.position_size.kind.value == "percent_of_capital"
        else 0.05  # fall back to 5% for fixed_qty / risk_based (simplified)
    )

    capital = float(initial_capital)
    peak_capital = capital
    max_drawdown_pct = 0.0
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    quantity = 0
    trades: List[DSLTrade] = []
    equity_curve: List[Dict[str, Any]] = []

    # Pre-cache empty engine signals so we don't allocate per bar
    empty_signals = EngineSignals()

    # Per-strategy stop/target
    sl_pct = strategy.stop_loss_pct
    tp_pct = strategy.take_profit_pct
    trail_pct = strategy.trailing_stop_pct
    trailing_high = 0.0  # for trailing-stop tracking

    for i in range(MIN_LOOKBACK, n):
        bar = df.iloc[i]
        bar_date = df.index[i]

        # Slice up to and including current bar — interpreter sees only
        # past data (no look-ahead).
        window = df.iloc[: i + 1]

        # Resolve engine signals at this bar
        if engine_signals_by_date is not None:
            es = engine_signals_by_date.get(bar_date, empty_signals)
        else:
            es = empty_signals

        ctx = InterpreterContext(bars=window, engines=es)

        # ────────────────────────────────────────────────────────────
        # IN POSITION → check exit conditions
        # ────────────────────────────────────────────────────────────
        if in_position:
            exit_reason = None
            exit_price = float(bar["close"])

            # 1. Hard stop loss
            if sl_pct is not None:
                stop = entry_price * (1 - sl_pct / 100)
                if float(bar["low"]) <= stop:
                    exit_reason = "stop_loss"
                    exit_price = stop  # fill at stop

            # 2. Take profit
            if exit_reason is None and tp_pct is not None:
                target = entry_price * (1 + tp_pct / 100)
                if float(bar["high"]) >= target:
                    exit_reason = "take_profit"
                    exit_price = target

            # 3. Trailing stop (high-water mark)
            if exit_reason is None and trail_pct is not None:
                trailing_high = max(trailing_high, float(bar["high"]))
                trail_stop = trailing_high * (1 - trail_pct / 100)
                if float(bar["low"]) <= trail_stop and trail_stop > entry_price:
                    exit_reason = "trailing_stop"
                    exit_price = trail_stop

            # 4. DSL exit condition
            if exit_reason is None and evaluate_exit(strategy, ctx):
                exit_reason = "exit_condition"
                exit_price = float(bar["close"])

            if exit_reason is not None:
                gross_pnl_pct, net_pnl_amount = _apply_costs(entry_price, exit_price, quantity)
                net_pnl_pct = (net_pnl_amount / (entry_price * quantity)) * 100
                capital += net_pnl_amount
                hold_days = (bar_date - df.index[entry_idx]).days

                trades.append(DSLTrade(
                    entry_date=_fmt_ts(df.index[entry_idx]),
                    entry_price=round(entry_price, 2),
                    exit_date=_fmt_ts(bar_date),
                    exit_price=round(exit_price, 2),
                    quantity=quantity,
                    direction="LONG",
                    hold_days=hold_days,
                    gross_pnl_pct=round(gross_pnl_pct, 4),
                    net_pnl_pct=round(net_pnl_pct, 4),
                    net_pnl_amount=round(net_pnl_amount, 2),
                    exit_reason=exit_reason,
                ))

                in_position = False
                trailing_high = 0.0
                peak_capital = max(peak_capital, capital)

        # ────────────────────────────────────────────────────────────
        # FLAT → check entry conditions
        # ────────────────────────────────────────────────────────────
        else:
            if evaluate_entry(strategy, ctx):
                entry_price = float(bar["close"])
                entry_idx = i
                trade_value = capital * pct_of_cap
                quantity = max(int(trade_value / entry_price), 1)
                in_position = True
                trailing_high = float(bar["high"])

        # Equity curve (mark-to-market unrealized)
        if in_position:
            mtm = capital + (float(bar["close"]) - entry_price) * quantity
        else:
            mtm = capital

        equity_curve.append({
            "date": _fmt_ts(bar_date),
            "equity": round(mtm, 2),
        })

        # Running max drawdown
        if mtm > peak_capital:
            peak_capital = mtm
        else:
            dd = (peak_capital - mtm) / peak_capital * 100
            if dd > max_drawdown_pct:
                max_drawdown_pct = dd

    # ────────────────────────────────────────────────────────────────
    # Close any still-open position at last close
    # ────────────────────────────────────────────────────────────────
    if in_position:
        last_bar = df.iloc[-1]
        exit_price = float(last_bar["close"])
        gross_pnl_pct, net_pnl_amount = _apply_costs(entry_price, exit_price, quantity)
        net_pnl_pct = (net_pnl_amount / (entry_price * quantity)) * 100
        capital += net_pnl_amount
        hold_days = (df.index[-1] - df.index[entry_idx]).days

        trades.append(DSLTrade(
            entry_date=_fmt_ts(df.index[entry_idx]),
            entry_price=round(entry_price, 2),
            exit_date=_fmt_ts(df.index[-1]),
            exit_price=round(exit_price, 2),
            quantity=quantity,
            direction="LONG",
            hold_days=hold_days,
            gross_pnl_pct=round(gross_pnl_pct, 4),
            net_pnl_pct=round(net_pnl_pct, 4),
            net_pnl_amount=round(net_pnl_amount, 2),
            exit_reason="end_of_data",
        ))

    # ────────────────────────────────────────────────────────────────
    # Aggregate stats
    # ────────────────────────────────────────────────────────────────
    return _summarize(
        strategy=strategy,
        symbol=symbol,
        df=df,
        initial_capital=initial_capital,
        final_capital=capital,
        trades=trades,
        equity_curve=equity_curve,
        max_drawdown_pct=max_drawdown_pct,
        periods_per_year=periods_per_year,
    )


def _summarize(
    *,
    strategy: Strategy,
    symbol: str,
    df: pd.DataFrame,
    initial_capital: float,
    final_capital: float,
    trades: List[DSLTrade],
    equity_curve: List[Dict[str, Any]],
    max_drawdown_pct: float,
    periods_per_year: float = 252.0,
) -> DSLBacktestResult:
    res = DSLBacktestResult(
        symbol=symbol,
        strategy_name=strategy.name,
        start_date=df.index[0].strftime("%Y-%m-%d"),
        end_date=df.index[-1].strftime("%Y-%m-%d"),
        initial_capital=initial_capital,
        final_capital=round(final_capital, 2),
        trades=trades,
        equity_curve=equity_curve,
        max_drawdown_pct=round(max_drawdown_pct, 4),
    )

    if trades:
        net_pcts = [t.net_pnl_pct for t in trades]
        wins = [p for p in net_pcts if p > 0]
        losses = [p for p in net_pcts if p <= 0]

        res.total_trades = len(trades)
        res.winning_trades = len(wins)
        res.losing_trades = len(losses)
        res.win_rate = round(len(wins) / len(trades), 4)
        res.avg_pnl_pct = round(float(np.mean(net_pcts)), 4)
        res.avg_hold_days = round(float(np.mean([t.hold_days for t in trades])), 2)

        # Profit factor = sum(gains) / |sum(losses)|
        gain_sum = sum(t.net_pnl_amount for t in trades if t.net_pnl_amount > 0)
        loss_sum = abs(sum(t.net_pnl_amount for t in trades if t.net_pnl_amount < 0))
        res.profit_factor = round(gain_sum / loss_sum, 4) if loss_sum > 0 else 0.0

        # Sharpe — per-bar equity returns annualized for THIS timeframe.
        # periods_per_year = bars/year (252 daily; ~18900 for 5m, etc.) so an
        # intraday strategy isn't mis-annualized with the daily factor.
        eq_series = pd.Series([p["equity"] for p in equity_curve]).pct_change().dropna()
        if len(eq_series) > 5 and eq_series.std() > 0:
            res.sharpe_ratio = round(
                float(eq_series.mean() / eq_series.std() * np.sqrt(periods_per_year)), 4,
            )

    res.total_return_pct = round((final_capital / initial_capital - 1) * 100, 4)
    return res


# ─────────────────────────────────────────────────────────────────────
# Walk-forward / out-of-sample evaluation
#
# A DSL strategy has no fitted parameters, so the overfit vector is
# *selection* (generate many, keep the best full-history backtest). The
# defence: run the single in-sample backtest once, then segment its trades
# + equity curve into K contiguous time windows and require the strategy to
# hold up across windows (consistency) AND on the most-recent holdout window.
# The promotion gate (evaluation.py) scores THIS, never the in-sample Sharpe.
# ─────────────────────────────────────────────────────────────────────


@dataclass
class FoldResult:
    index: int
    start_date: str
    end_date: str
    trades: int
    net_pnl_amount: float
    net_return_pct: float     # equity change across the window
    win_rate: float
    sharpe: float
    max_drawdown_pct: float
    profitable: bool          # positive realised trade P&L in this window


@dataclass
class WalkForwardResult:
    symbol: str
    strategy_name: str
    n_folds: int
    folds: List[FoldResult]
    in_sample: Any                        # DSLBacktestResult | OptionsBacktestResult
    # ── aggregate out-of-sample metrics (what the gate reads) ──
    oos_trades: int
    oos_folds_profitable: int
    oos_consistency: float                # folds_profitable / n_folds
    oos_mean_sharpe: float
    oos_worst_drawdown_pct: float
    holdout_return_pct: float             # most-recent window
    holdout_sharpe: float
    holdout_trades: int

    def _oos_dict(self) -> Dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "oos_trades": self.oos_trades,
            "oos_folds_profitable": self.oos_folds_profitable,
            "oos_consistency": round(self.oos_consistency, 4),
            "oos_mean_sharpe": round(self.oos_mean_sharpe, 4),
            "oos_worst_drawdown_pct": round(self.oos_worst_drawdown_pct, 4),
            "holdout_return_pct": round(self.holdout_return_pct, 4),
            "holdout_sharpe": round(self.holdout_sharpe, 4),
            "holdout_trades": self.holdout_trades,
            "folds": [
                {
                    "index": f.index,
                    "start_date": f.start_date,
                    "end_date": f.end_date,
                    "trades": f.trades,
                    "net_return_pct": round(f.net_return_pct, 4),
                    "win_rate": round(f.win_rate, 4),
                    "sharpe": round(f.sharpe, 4),
                    "max_drawdown_pct": round(f.max_drawdown_pct, 4),
                    "profitable": f.profitable,
                }
                for f in self.folds
            ],
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        """In-sample summary + an ``out_of_sample`` block — persisted to
        user_strategies.last_backtest. The gate reads ``out_of_sample``."""
        summary = self.in_sample.to_summary_dict()
        summary["out_of_sample"] = self._oos_dict()
        return summary

    def to_full_dict(self) -> Dict[str, Any]:
        """Full in-sample payload (trades + curve) + the OOS block."""
        full = self.in_sample.to_full_dict()
        full["out_of_sample"] = self._oos_dict()
        return full


def _fold_count(eval_bars: int, requested: int) -> int:
    """Pick a fold count that keeps each window ≥ ~20 bars; ≥1, ≤ requested."""
    if eval_bars < 20:
        return 1
    return max(1, min(requested, eval_bars // 20))


def _segment_sharpe(equities: List[float], periods_per_year: float = 252.0) -> float:
    if len(equities) < 6:
        return 0.0
    s = pd.Series(equities).pct_change().dropna()
    if len(s) < 5 or s.std() == 0:
        return 0.0
    return float(s.mean() / s.std() * np.sqrt(periods_per_year))


def _trade_pnl(trade: Any) -> float:
    """Realised P&L of a trade — works for DSLTrade (net_pnl_amount) and
    OptionsTrade (net_pnl_inr)."""
    val = getattr(trade, "net_pnl_amount", None)
    if val is None:
        val = getattr(trade, "net_pnl_inr", 0.0)
    return float(val or 0.0)


def _segment_max_dd(equities: List[float]) -> float:
    peak = equities[0]
    mdd = 0.0
    for e in equities:
        if e > peak:
            peak = e
        elif peak > 0:
            dd = (peak - e) / peak * 100
            if dd > mdd:
                mdd = dd
    return mdd


def _walk_forward_from_result(
    full: Any,
    *,
    symbol: str,
    folds: int,
    periods_per_year: float,
) -> WalkForwardResult:
    """Segment any completed backtest (DSL or options) into ``folds``
    contiguous time windows + holdout. ``full`` must expose ``trades`` (each
    with ``entry_date`` + a P&L attr), ``equity_curve`` and ``strategy_name``.

    Trades are assigned to a window by entry date, so every trade lands in
    exactly one fold (sum of per-fold trades == in-sample total).
    """
    curve = full.equity_curve  # post-warmup eval bars: [{date, equity}]
    m = len(curve)
    k = _fold_count(m, folds)
    dates = [c["date"] for c in curve]
    equities = [float(c["equity"]) for c in curve]
    bounds = [round(i * m / k) for i in range(k + 1)]

    fold_results: List[FoldResult] = []
    fi = 0
    for i in range(k):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        seg_dates = dates[lo:hi]
        seg_eq = equities[lo:hi]
        start_date, end_date = seg_dates[0], seg_dates[-1]
        # ISO date strings compare lexicographically — safe window match.
        seg_trades = [t for t in full.trades if start_date <= str(t.entry_date) <= end_date]
        pnl = sum(_trade_pnl(t) for t in seg_trades)
        wins = sum(1 for t in seg_trades if _trade_pnl(t) > 0)
        net_return_pct = (seg_eq[-1] / seg_eq[0] - 1) * 100 if seg_eq[0] else 0.0
        fold_results.append(FoldResult(
            index=fi,
            start_date=start_date,
            end_date=end_date,
            trades=len(seg_trades),
            net_pnl_amount=round(pnl, 2),
            net_return_pct=round(net_return_pct, 4),
            win_rate=round(wins / len(seg_trades), 4) if seg_trades else 0.0,
            sharpe=round(_segment_sharpe(seg_eq, periods_per_year), 4),
            max_drawdown_pct=round(_segment_max_dd(seg_eq), 4),
            profitable=pnl > 0,
        ))
        fi += 1

    n = len(fold_results)
    oos_trades = sum(f.trades for f in fold_results)
    folds_profitable = sum(1 for f in fold_results if f.profitable)
    consistency = folds_profitable / n if n else 0.0
    mean_sharpe = float(np.mean([f.sharpe for f in fold_results])) if fold_results else 0.0
    worst_dd = max((f.max_drawdown_pct for f in fold_results), default=0.0)
    holdout = fold_results[-1] if fold_results else None

    return WalkForwardResult(
        symbol=symbol,
        strategy_name=full.strategy_name,
        n_folds=n,
        folds=fold_results,
        in_sample=full,
        oos_trades=oos_trades,
        oos_folds_profitable=folds_profitable,
        oos_consistency=consistency,
        oos_mean_sharpe=mean_sharpe,
        oos_worst_drawdown_pct=worst_dd,
        holdout_return_pct=holdout.net_return_pct if holdout else 0.0,
        holdout_sharpe=holdout.sharpe if holdout else 0.0,
        holdout_trades=holdout.trades if holdout else 0,
    )


def run_walk_forward(
    strategy: Strategy,
    ohlcv: pd.DataFrame,
    *,
    symbol: str,
    folds: int = 4,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    engine_signals_by_date: Optional[Dict[pd.Timestamp, EngineSignals]] = None,
    periods_per_year: float = 252.0,
) -> WalkForwardResult:
    """Equity DSL walk-forward: run the in-sample backtest once, then segment
    it into ``folds`` time windows + holdout. ``periods_per_year`` annualizes
    the Sharpe for the strategy's timeframe (see timeframes.py)."""
    full = run_dsl_backtest(
        strategy,
        ohlcv,
        symbol=symbol,
        initial_capital=initial_capital,
        engine_signals_by_date=engine_signals_by_date,
        periods_per_year=periods_per_year,
    )
    return _walk_forward_from_result(
        full, symbol=symbol, folds=folds, periods_per_year=periods_per_year,
    )


def run_options_walk_forward(
    strategy: Strategy,
    ohlcv: pd.DataFrame,
    *,
    symbol: str,
    folds: int = 4,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    engine_signals_by_date: Optional[Dict[pd.Timestamp, EngineSignals]] = None,
    periods_per_year: float = 252.0,
) -> WalkForwardResult:
    """Options multi-leg walk-forward — same windowing over the synthetic
    options backtest so options strategies also get an out-of-sample gate
    (fixes the 'OPTIONS can never reach live' gap)."""
    from .options_backtest import run_options_backtest

    full = run_options_backtest(
        strategy,
        ohlcv,
        symbol=symbol,
        initial_capital=initial_capital,
        engine_signals_by_date=engine_signals_by_date,
    )
    return _walk_forward_from_result(
        full, symbol=symbol, folds=folds, periods_per_year=periods_per_year,
    )


# ─────────────────────────────────────────────────────────────────────
# Multi-symbol (universe) walk-forward — robustness ACROSS symbols.
#
# A strategy that only works on one cherry-picked symbol is overfit. The
# universe walk-forward runs the per-symbol walk-forward across the strategy's
# declared universe and adds a *breadth* metric (fraction of symbols
# profitable). The gate (evaluation.py) requires breadth ≥ a floor for
# universe strategies, so "great on RELIANCE, loses on the other 49" is blocked.
# ─────────────────────────────────────────────────────────────────────


@dataclass
class SymbolWFSummary:
    symbol: str
    oos_trades: int
    oos_mean_sharpe: float
    oos_consistency: float
    oos_worst_drawdown_pct: float
    holdout_return_pct: float
    total_net_pnl: float
    profitable: bool


@dataclass
class UniverseWalkForwardResult:
    universe: str
    strategy_name: str
    symbols_tested: int
    symbols_profitable: int
    breadth: float                         # symbols_profitable / symbols_tested
    per_symbol: List[SymbolWFSummary]
    # ── aggregate out-of-sample metrics (the gate reads these) ──
    oos_trades: int
    oos_mean_sharpe: float                 # mean across symbols
    oos_consistency: float                 # mean per-symbol time-consistency
    oos_worst_drawdown_pct: float          # worst across symbols
    holdout_return_pct: float              # mean across symbols
    holdout_trades: int
    # ── in-sample aggregate (display) ──
    total_trades: int
    mean_win_rate: float

    def _oos_dict(self) -> Dict[str, Any]:
        return {
            "oos_trades": self.oos_trades,
            "oos_mean_sharpe": round(self.oos_mean_sharpe, 4),
            "oos_consistency": round(self.oos_consistency, 4),
            "oos_worst_drawdown_pct": round(self.oos_worst_drawdown_pct, 4),
            "holdout_return_pct": round(self.holdout_return_pct, 4),
            "holdout_trades": self.holdout_trades,
            # multi-symbol breadth (drives the universe gate)
            "symbols_tested": self.symbols_tested,
            "symbols_profitable": self.symbols_profitable,
            "breadth": round(self.breadth, 4),
            "per_symbol": [
                {
                    "symbol": s.symbol,
                    "oos_trades": s.oos_trades,
                    "oos_mean_sharpe": round(s.oos_mean_sharpe, 4),
                    "oos_consistency": round(s.oos_consistency, 4),
                    "oos_worst_drawdown_pct": round(s.oos_worst_drawdown_pct, 4),
                    "holdout_return_pct": round(s.holdout_return_pct, 4),
                    "profitable": s.profitable,
                }
                for s in self.per_symbol
            ],
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "symbol": f"universe:{self.universe}" if self.universe else "universe",
            "strategy_name": self.strategy_name,
            "total_trades": self.total_trades,
            "win_rate": round(self.mean_win_rate, 4),
            "sharpe_ratio": round(self.oos_mean_sharpe, 4),  # OOS mean = headline
            "symbols_tested": self.symbols_tested,
            "out_of_sample": self._oos_dict(),
        }

    def to_full_dict(self) -> Dict[str, Any]:
        return self.to_summary_dict()


def run_universe_walk_forward(
    strategy: Strategy,
    ohlcv_by_symbol: Dict[str, pd.DataFrame],
    *,
    universe: str = "",
    folds: int = 4,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    periods_per_year: float = 252.0,
    engine_signals_by_symbol: Optional[Dict[str, Dict[pd.Timestamp, EngineSignals]]] = None,
) -> UniverseWalkForwardResult:
    """Walk-forward across every symbol in ``ohlcv_by_symbol``. Symbols with
    insufficient history are skipped. Aggregates OOS metrics + a breadth score.
    """
    per_symbol: List[SymbolWFSummary] = []
    sharpes: List[float] = []
    consistencies: List[float] = []
    worst_dds: List[float] = []
    holdouts: List[float] = []
    oos_trades_total = 0
    holdout_trades_total = 0
    in_sample_trades = 0
    win_rates: List[float] = []

    for sym, df in ohlcv_by_symbol.items():
        try:
            wf = run_walk_forward(
                strategy,
                df,
                symbol=sym,
                folds=folds,
                initial_capital=initial_capital,
                periods_per_year=periods_per_year,
                engine_signals_by_date=(engine_signals_by_symbol or {}).get(sym),
            )
        except ValueError:
            continue  # insufficient bars for this symbol — skip, don't fail the batch

        total_net = sum(f.net_pnl_amount for f in wf.folds)
        per_symbol.append(SymbolWFSummary(
            symbol=sym,
            oos_trades=wf.oos_trades,
            oos_mean_sharpe=wf.oos_mean_sharpe,
            oos_consistency=wf.oos_consistency,
            oos_worst_drawdown_pct=wf.oos_worst_drawdown_pct,
            holdout_return_pct=wf.holdout_return_pct,
            total_net_pnl=round(total_net, 2),
            profitable=total_net > 0,
        ))
        sharpes.append(wf.oos_mean_sharpe)
        consistencies.append(wf.oos_consistency)
        worst_dds.append(wf.oos_worst_drawdown_pct)
        holdouts.append(wf.holdout_return_pct)
        oos_trades_total += wf.oos_trades
        holdout_trades_total += wf.holdout_trades
        in_sample_trades += wf.in_sample.total_trades
        win_rates.append(wf.in_sample.win_rate)

    n = len(per_symbol)
    symbols_profitable = sum(1 for s in per_symbol if s.profitable)

    return UniverseWalkForwardResult(
        universe=universe,
        strategy_name=strategy.name,
        symbols_tested=n,
        symbols_profitable=symbols_profitable,
        breadth=(symbols_profitable / n) if n else 0.0,
        per_symbol=per_symbol,
        oos_trades=oos_trades_total,
        oos_mean_sharpe=float(np.mean(sharpes)) if sharpes else 0.0,
        oos_consistency=float(np.mean(consistencies)) if consistencies else 0.0,
        oos_worst_drawdown_pct=max(worst_dds, default=0.0),
        holdout_return_pct=float(np.mean(holdouts)) if holdouts else 0.0,
        holdout_trades=holdout_trades_total,
        total_trades=in_sample_trades,
        mean_win_rate=float(np.mean(win_rates)) if win_rates else 0.0,
    )


__all__ = [
    "DSLBacktestResult",
    "DSLTrade",
    "FoldResult",
    "WalkForwardResult",
    "run_dsl_backtest",
    "run_walk_forward",
    "run_options_walk_forward",
    "run_universe_walk_forward",
    "SymbolWFSummary",
    "UniverseWalkForwardResult",
    "DEFAULT_INITIAL_CAPITAL",
]
