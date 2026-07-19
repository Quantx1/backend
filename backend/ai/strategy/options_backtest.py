"""Multi-leg options backtest — PR-J2.

Synthetic-pricing engine. Given equity OHLCV history we:

1. Estimate realized-vol-based σ from the underlier's bars (proxy for IV)
2. On each ENTRY signal: resolve the legs at that bar's close (= spot),
   price each leg via Black-Scholes, record the entry premium.
3. On each subsequent bar: re-price each leg with the new spot + reduced
   time-to-expiry. Aggregate position value = Σ side · premium · qty.
4. Exit triggers: stop_loss_pct / take_profit_pct on aggregate value,
   DSL exit condition firing, OR expiry day reached (auto-close at
   intrinsic value).

Caveats — labelled clearly:
* IV is approximated from 20-bar realized vol on the underlier. Real
  options trade above realized due to vol risk premium → backtest P&L
  for short-vol strategies (Iron Condor, Short Strangle) is OPTIMISTIC.
  This is acknowledged; we record the σ used in the trade record so
  the UI can show "synthetic backtest" disclaimers.
* No bid-ask spread modeling; assumes mid-price fills.
* No IV smile/skew — single σ per bar regardless of strike. OTM strikes
  in reality price higher than ATM (vol smile).
* These approximations are good enough for relative comparison + dry-run
  sanity, NOT for live-money sizing. Forward paper-trading remains the
  ground-truth signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .dsl import OptionSide, OptionType, Strategy
from .indicators import MIN_LOOKBACK
from .interpreter import EngineSignals, InterpreterContext, evaluate_condition
from .options_resolver import ResolvedLeg, resolve_legs


RISK_FREE_RATE = 0.07  # India 10Y yield — close enough for short-dated weeklies
REALIZED_VOL_WINDOW = 20  # bars used to estimate σ


# Cost model — round-trip options costs per leg, in % of premium.
# Zerodha brokerage on options: ₹20 flat per order. STT 0.0625% on sell.
# Slippage: 0.5% of premium (options spreads are wider than equity).
# We model the round trip as one number per leg.
OPTIONS_SLIPPAGE_PCT = 0.5
OPTIONS_STT_PCT = 0.0625
OPTIONS_BROKERAGE_FLAT_INR = 20.0  # per leg per side


@dataclass
class LegSnapshot:
    """One leg's state at one bar — used for trade records."""
    side: str
    option_type: str
    strike: float
    expiry: str  # ISO date
    qty_lots: int
    premium: float
    iv: float


@dataclass
class OptionsTrade:
    """A multi-leg position open-to-close. ``net_pnl_inr`` includes costs."""
    entry_date: str
    exit_date: str
    spot_entry: float
    spot_exit: float
    legs_entry: List[LegSnapshot]
    legs_exit: List[LegSnapshot]
    lot_size: int
    qty_total: int          # lot_size × Σ leg.qty_lots
    gross_pnl_inr: float
    costs_inr: float
    net_pnl_inr: float
    net_pnl_pct: float       # vs initial position margin
    margin_used_inr: float
    exit_reason: str          # stop_loss | take_profit | exit_condition | expiry | end_of_data
    sigma_at_entry: float


@dataclass
class OptionsBacktestResult:
    """Full result envelope — mirrors DSLBacktestResult."""
    symbol: str
    strategy_name: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    win_rate: float
    avg_win_inr: float
    avg_loss_inr: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[OptionsTrade] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    sigma_assumptions: Dict[str, float] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "segment": "OPTIONS",
            "initial_capital": self.initial_capital,
            "final_capital": round(self.final_capital, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 3),
            "avg_win_inr": round(self.avg_win_inr, 2),
            "avg_loss_inr": round(self.avg_loss_inr, 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "synthetic_backtest": True,  # always — see module docstring
        }

    def to_full_dict(self) -> Dict[str, Any]:
        out = self.to_summary_dict()
        out["trades"] = [t.__dict__ for t in self.trades]
        out["equity_curve"] = self.equity_curve
        out["sigma_assumptions"] = self.sigma_assumptions
        return out


# ─────────────────────────────────────────────────────────────────────
# BS pricer (no scipy — uses math.erf-based normal CDF)
# ─────────────────────────────────────────────────────────────────────


def _ncdf(x: float) -> float:
    """Standard normal CDF — math.erf based, no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(
    S: float, K: float, T: float, r: float, sigma: float, *, is_call: bool,
) -> float:
    """Black-Scholes mid price. Returns intrinsic at expiry (T<=0)."""
    if T <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    if sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def _realized_vol(closes: pd.Series, window: int = REALIZED_VOL_WINDOW) -> float:
    """Annualized realized vol of log returns over the last ``window`` bars."""
    if len(closes) < window + 1:
        return 0.20  # 20% default — neither low nor high
    log_ret = np.log(closes.iloc[-(window + 1):] / closes.iloc[-(window + 1):].shift(1)).dropna()
    return float(log_ret.std() * math.sqrt(252))


# ─────────────────────────────────────────────────────────────────────
# Leg / position math
# ─────────────────────────────────────────────────────────────────────


def _price_position(
    resolved: List[ResolvedLeg],
    *,
    spot: float,
    today: date,
    sigma: float,
    r: float = RISK_FREE_RATE,
    lots: int = 1,
) -> tuple[float, List[LegSnapshot]]:
    """Price every leg at ``today`` against current ``spot`` + ``sigma``.

    Internally always computes premium per single lot (independent of
    LegSpec.qty_lots, which is the SHAPE ratio only). ``lots`` scales
    the snapshot field for display — the returned per-leg net price is
    always per-lot so entry-vs-MTM math stays consistent regardless of
    how many lots got deployed.

    Returns (net_premium_per_lot, leg_snapshots).
    ``net_premium_per_lot`` = Σ (side · premium · leg_ratio) per ONE lot
    of the whole multi-leg structure. Credit positive, debit negative.
    """
    snapshots: List[LegSnapshot] = []
    net = 0.0
    for leg in resolved:
        days_to_expiry = max((leg.expiry - today).days, 0)
        T = days_to_expiry / 365.25
        prem = _bs_price(
            spot, leg.strike, T, r, sigma,
            is_call=(leg.option_type == OptionType.CE),
        )
        sign = -1.0 if leg.side == OptionSide.BUY else 1.0
        # leg.qty_lots is the SHAPE ratio (e.g. 2:1 ratio spread). Don't
        # confuse with deployment lots. Keep ratio in net so position math
        # is correct for asymmetric structures.
        net += sign * prem * leg.qty_lots
        snapshots.append(LegSnapshot(
            side=leg.side.value,
            option_type=leg.option_type.value,
            strike=leg.strike,
            expiry=leg.expiry.isoformat(),
            qty_lots=lots * leg.qty_lots,
            premium=round(prem, 2),
            iv=round(sigma, 4),
        ))
    return net, snapshots


def _estimate_margin(net_credit_per_lot: float, lot_size: int, lots: int, spread_width: float) -> float:
    """Rough margin estimate for sizing.

    Defined-risk credit (Iron Condor / Iron Butterfly) → max_loss + small buffer.
    Undefined-risk credit (Short Strangle) → SPAN approx ≈ 15% of spot × lot.
    Debit positions → premium paid.
    """
    if net_credit_per_lot >= 0 and spread_width > 0:
        # Credit spread — margin = (spread_width − credit) × lot
        max_loss_per_lot = max(0.0, spread_width - net_credit_per_lot)
        return max_loss_per_lot * lot_size * lots + 5000.0  # ₹5k buffer
    if net_credit_per_lot >= 0:
        # Naked credit (Short Strangle) — approximate SPAN at 8% of notional
        return abs(net_credit_per_lot) * lot_size * lots * 4.0 + 50_000.0
    # Debit (Bull Call / Long Straddle) — pay premium
    return abs(net_credit_per_lot) * lot_size * lots


def _spread_width(resolved: List[ResolvedLeg]) -> float:
    """Max strike − min strike per option_type for vertical-spread margin sizing.
    Returns 0 if not a defined-risk spread (e.g. naked or straddle)."""
    calls = [r.strike for r in resolved if r.option_type == OptionType.CE]
    puts = [r.strike for r in resolved if r.option_type == OptionType.PE]
    width = 0.0
    if len(calls) >= 2:
        width = max(width, max(calls) - min(calls))
    if len(puts) >= 2:
        width = max(width, max(puts) - min(puts))
    return width


def _per_leg_cost(prem: float, lot_size: int, lots: int) -> float:
    """Round-trip cost for one leg, in ₹.

    Slippage: applied to entry + exit premium.
    STT: applied on the SELL side only (entry for sell-legs, exit for buy-legs).
    Brokerage: ₹20 flat per side per leg.
    """
    notional = prem * lot_size * lots
    slip = notional * (OPTIONS_SLIPPAGE_PCT / 100.0) * 2  # entry + exit
    stt = notional * (OPTIONS_STT_PCT / 100.0)
    brok = OPTIONS_BROKERAGE_FLAT_INR * 2  # in + out
    return slip + stt + brok


# ─────────────────────────────────────────────────────────────────────
# Entry/exit signal evaluation — reuses the equity DSL interpreter
# ─────────────────────────────────────────────────────────────────────


def _evaluate_entry_at(
    strategy: Strategy, bars: pd.DataFrame, idx: int, engine_signals: Optional[EngineSignals],
) -> bool:
    """Replay the equity interpreter on the bar at idx to fire entry."""
    sub = bars.iloc[: idx + 1]
    ctx = InterpreterContext(
        bars=sub,
        engines=engine_signals or EngineSignals(),
    )
    # Apply regime_filter gate first
    if strategy.regime_filter.value != "any" and engine_signals is not None:
        rg = engine_signals.regime
        expected = strategy.regime_filter.value.replace("_only", "")
        if rg is not None and rg != expected:
            return False
    return evaluate_condition(strategy.entry, ctx)


def _evaluate_exit_at(
    strategy: Strategy, bars: pd.DataFrame, idx: int, engine_signals: Optional[EngineSignals],
) -> bool:
    sub = bars.iloc[: idx + 1]
    ctx = InterpreterContext(
        bars=sub,
        engines=engine_signals or EngineSignals(),
    )
    return evaluate_condition(strategy.exit, ctx)


# ─────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────


DEFAULT_INITIAL_CAPITAL = 500_000.0


def run_options_backtest(
    strategy: Strategy,
    bars: pd.DataFrame,
    *,
    symbol: str,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    engine_signals_by_date: Optional[Dict[pd.Timestamp, EngineSignals]] = None,
) -> OptionsBacktestResult:
    """Run a multi-leg options strategy through bars of the underlier.

    ``bars``: OHLCV of the underlying (NIFTY / BANKNIFTY etc.) — daily.
    σ is estimated bar-by-bar from rolling realized vol.
    """
    if strategy.legs is None or len(strategy.legs) == 0:
        raise ValueError("run_options_backtest requires Strategy.legs (multi-leg)")
    if strategy.symbol is None:
        raise ValueError("OPTIONS strategy requires symbol (underlier)")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")
    if len(bars) < MIN_LOOKBACK + 10:
        raise ValueError(
            f"insufficient bars for options backtest: got {len(bars)}, "
            f"need >= {MIN_LOOKBACK + 10}"
        )

    from ...trading.fo.engine import NSE_LOT_SIZES
    lot_size = NSE_LOT_SIZES.get(symbol.upper(), 50)

    trades: List[OptionsTrade] = []
    equity_curve: List[Dict[str, Any]] = []
    capital = initial_capital

    in_position = False
    entry_idx = -1
    entry_resolved: List[ResolvedLeg] = []
    entry_net_per_lot = 0.0
    entry_snapshots: List[LegSnapshot] = []
    entry_margin = 0.0
    entry_sigma = 0.0
    entry_lots = 1
    sigma_samples: List[float] = []
    # PR-DEPTH stagnation-aware trailing — tracks per-position peak PnL
    # and bar index when peak was set. Lets options_backtest tighten SL
    # aggressively when the trade isn't advancing (theta decay protection).
    peak_pnl_per_lot = 0.0
    peak_bar_idx = -1
    trailing_active = False

    for i in range(MIN_LOOKBACK, len(bars)):
        bar = bars.iloc[i]
        bar_date = bars.index[i].date() if hasattr(bars.index[i], "date") else bars.index[i]
        spot = float(bar["close"])
        eng = (engine_signals_by_date or {}).get(bars.index[i])
        sigma = _realized_vol(bars["close"].iloc[: i + 1])
        sigma_samples.append(sigma)

        if not in_position:
            # Check entry
            if _evaluate_entry_at(strategy, bars, i, eng):
                entry_resolved = resolve_legs(
                    strategy.legs, spot=spot, symbol=symbol, today=bar_date, sigma=sigma,
                )
                # Price per single lot to compute margin / sizing.
                entry_net_per_lot, _ = _price_position(
                    entry_resolved, spot=spot, today=bar_date, sigma=sigma, lots=1,
                )
                spread_w = _spread_width(entry_resolved)
                margin_per_lot = _estimate_margin(entry_net_per_lot, lot_size, 1, spread_w)

                # Sizing: position_size says how much capital to deploy.
                ps_kind = strategy.position_size.kind.value
                ps_val = strategy.position_size.value
                if ps_kind == "percent_of_capital":
                    deploy = capital * ps_val / 100.0
                elif ps_kind == "fixed_qty":
                    deploy = margin_per_lot * ps_val
                else:  # risk_based
                    deploy = capital * ps_val / 100.0
                entry_lots = max(1, int(deploy // margin_per_lot)) if margin_per_lot > 0 else 1
                entry_margin = margin_per_lot * entry_lots

                # Now re-snapshot with the final lots count for display only.
                _, entry_snapshots = _price_position(
                    entry_resolved, spot=spot, today=bar_date, sigma=sigma, lots=entry_lots,
                )
                entry_idx = i
                entry_sigma = sigma
                in_position = True
                equity_curve.append({
                    "date": str(bar_date), "equity": round(capital, 2),
                    "spot": round(spot, 2), "event": "ENTRY",
                })
                continue

            equity_curve.append({
                "date": str(bar_date), "equity": round(capital, 2),
                "spot": round(spot, 2),
            })
            continue

        # In position: mark-to-market — always per-single-lot pricing.
        cur_net_per_lot, cur_snapshots = _price_position(
            entry_resolved, spot=spot, today=bar_date, sigma=sigma, lots=entry_lots,
        )
        position_pnl_per_lot = cur_net_per_lot - entry_net_per_lot
        position_pnl_inr = position_pnl_per_lot * lot_size * entry_lots
        mtm_equity = capital + position_pnl_inr

        # PR-DEPTH stagnation-aware trailing tracking — update peak before exit check
        if position_pnl_per_lot > peak_pnl_per_lot:
            peak_pnl_per_lot = position_pnl_per_lot
            peak_bar_idx = i

        # Exit triggers — priority: SL → trailing_sl (stagnation) → TP →
        # DSL exit → expiry → continue
        exit_reason: Optional[str] = None
        if strategy.stop_loss_pct and entry_margin > 0:
            if (-position_pnl_inr) / entry_margin * 100 >= strategy.stop_loss_pct:
                exit_reason = "stop_loss"
        # Stagnation-aware trailing: if peak gain was meaningful AND we
        # haven't made a new peak for N bars, tighten the effective SL.
        # Mirrors aaryansinha's intra-bar exit sequence (2026-04-16).
        if exit_reason is None and entry_margin > 0 and peak_pnl_per_lot > 0:
            peak_gain_pct = (peak_pnl_per_lot * lot_size * entry_lots) / entry_margin * 100
            bars_since_peak = i - peak_bar_idx
            if peak_gain_pct >= 15 and bars_since_peak >= 5:
                # Tier the give-back tolerance by stagnation length
                if bars_since_peak >= 20:
                    retention = 0.90        # near-peak lock
                elif bars_since_peak >= 10:
                    retention = 0.82
                else:
                    retention = 0.70
                # Trailing SL fires if current PnL fell below retention × peak
                current_gain_pct = position_pnl_inr / entry_margin * 100
                trailing_sl_pct = peak_gain_pct * retention
                if current_gain_pct <= trailing_sl_pct:
                    exit_reason = "trailing_sl"
                    trailing_active = True
        if exit_reason is None and strategy.take_profit_pct and entry_margin > 0:
            if position_pnl_inr / entry_margin * 100 >= strategy.take_profit_pct:
                exit_reason = "take_profit"
        if exit_reason is None and _evaluate_exit_at(strategy, bars, i, eng):
            exit_reason = "exit_condition"

        # ── RL exit agent: early-exit suggestion (PR-MODELS) ─────────
        # Hard SL/target/exit_condition above remain authoritative. RL
        # can only ADD an early EXIT pathway when it's confident. Loaded
        # lazily on first call; no-op when ENABLE_RL_EXIT=false or
        # Q-table not trained for this state.
        if exit_reason is None and entry_margin > 0:
            try:
                from ..exit_engine.rl_exit_scaffold import (
                    get_rl_exit_agent, compute_rl_state,
                )
                rl_agent = get_rl_exit_agent()
                if rl_agent.is_enabled and rl_agent.is_loaded:
                    # Build state from current position metrics (per-lot scale)
                    current_value_per_lot = entry_net_per_lot + position_pnl_per_lot
                    state = compute_rl_state(
                        entry_price=abs(entry_net_per_lot) if entry_net_per_lot != 0 else 1.0,
                        current_price=abs(current_value_per_lot) if current_value_per_lot != 0 else 1.0,
                        bars_held=i - entry_idx,
                        max_hold_bars=40,
                        sl=entry_margin * (1 - (strategy.stop_loss_pct or 50) / 100),
                        target=entry_margin * (1 + (strategy.take_profit_pct or 40) / 100),
                        trailing_active=trailing_active,
                        peak_price=peak_pnl_per_lot if peak_pnl_per_lot > 0 else 0,
                        price_history=[entry_net_per_lot, current_value_per_lot],
                    )
                    rl_action = rl_agent.decide(state)
                    if rl_action == "EXIT":
                        # Sanity: only fire RL exit when we're in modest profit
                        # or shallow loss. Don't override a deep-loss SL path.
                        pnl_pct_check = position_pnl_inr / entry_margin * 100
                        if -strategy.stop_loss_pct * 0.7 <= pnl_pct_check < (strategy.take_profit_pct or 999):
                            exit_reason = "rl_exit"
            except Exception:
                pass  # fail-open — RL is advisory only
        # Expiry day check
        if exit_reason is None:
            min_exp = min(leg.expiry for leg in entry_resolved)
            if bar_date >= min_exp:
                exit_reason = "expiry"

        if exit_reason is None and i == len(bars) - 1:
            exit_reason = "end_of_data"

        if exit_reason is not None:
            gross = position_pnl_inr
            # Costs: per leg round-trip — use leg's shape ratio × deployment lots
            costs = 0.0
            for snap, e_snap, leg in zip(cur_snapshots, entry_snapshots, entry_resolved):
                avg = (snap.premium + e_snap.premium) / 2.0
                costs += _per_leg_cost(avg, lot_size, entry_lots * leg.qty_lots)
            net = gross - costs
            capital += net

            entry_date = bars.index[entry_idx].date() if hasattr(
                bars.index[entry_idx], "date") else bars.index[entry_idx]
            trades.append(OptionsTrade(
                entry_date=str(entry_date),
                exit_date=str(bar_date),
                spot_entry=round(float(bars.iloc[entry_idx]["close"]), 2),
                spot_exit=round(spot, 2),
                legs_entry=entry_snapshots,
                legs_exit=cur_snapshots,
                lot_size=lot_size,
                qty_total=lot_size * entry_lots,
                gross_pnl_inr=round(gross, 2),
                costs_inr=round(costs, 2),
                net_pnl_inr=round(net, 2),
                net_pnl_pct=round(net / entry_margin * 100 if entry_margin > 0 else 0.0, 2),
                margin_used_inr=round(entry_margin, 2),
                exit_reason=exit_reason,
                sigma_at_entry=round(entry_sigma, 4),
            ))
            equity_curve.append({
                "date": str(bar_date), "equity": round(capital, 2),
                "spot": round(spot, 2), "event": f"EXIT:{exit_reason}",
            })
            in_position = False
            entry_resolved = []
            entry_snapshots = []
            entry_margin = 0.0
            peak_pnl_per_lot = 0.0
            peak_bar_idx = -1
            trailing_active = False
        else:
            equity_curve.append({
                "date": str(bar_date), "equity": round(mtm_equity, 2),
                "spot": round(spot, 2),
            })

    # ── stats ─────────────────────────────────────────────────────
    total = len(trades)
    wins = [t for t in trades if t.net_pnl_inr > 0]
    losses = [t for t in trades if t.net_pnl_inr <= 0]
    win_rate = len(wins) / total if total else 0.0
    avg_win = float(np.mean([t.net_pnl_inr for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.net_pnl_inr for t in losses])) if losses else 0.0
    gross_win = sum(t.net_pnl_inr for t in wins)
    gross_loss = -sum(t.net_pnl_inr for t in losses)
    pf = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    eq_vals = [e["equity"] for e in equity_curve]
    if eq_vals:
        peak = eq_vals[0]
        max_dd = 0.0
        for v in eq_vals:
            peak = max(peak, v)
            if peak > 0:
                dd = (peak - v) / peak * 100
                max_dd = max(max_dd, dd)
    else:
        max_dd = 0.0

    # Sharpe — daily-equity-return based, annualized √252
    sharpe = 0.0
    if len(eq_vals) > 1:
        rets = np.diff(eq_vals) / np.array(eq_vals[:-1])
        rets = rets[np.isfinite(rets)]
        if len(rets) > 1 and rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))

    total_return_pct = (capital / initial_capital - 1) * 100

    return OptionsBacktestResult(
        symbol=symbol,
        strategy_name=strategy.name,
        initial_capital=initial_capital,
        final_capital=capital,
        total_return_pct=total_return_pct,
        total_trades=total,
        win_rate=win_rate,
        avg_win_inr=avg_win,
        avg_loss_inr=avg_loss,
        profit_factor=pf if math.isfinite(pf) else 999.99,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        trades=trades,
        equity_curve=equity_curve,
        sigma_assumptions={
            "mean_sigma": round(float(np.mean(sigma_samples)), 4) if sigma_samples else 0.0,
            "min_sigma": round(float(np.min(sigma_samples)), 4) if sigma_samples else 0.0,
            "max_sigma": round(float(np.max(sigma_samples)), 4) if sigma_samples else 0.0,
            "window_bars": REALIZED_VOL_WINDOW,
        },
    )


__all__ = [
    "DEFAULT_INITIAL_CAPITAL",
    "OptionsBacktestResult",
    "OptionsTrade",
    "LegSnapshot",
    "run_options_backtest",
]
