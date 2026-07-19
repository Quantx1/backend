"""Live track record aggregation for AutoPilot.

CRITICAL #2 (2026-05-31) — closes the audit gap "we have zero days of
live track record." Aggregates the realised outcomes from paper_trades +
trades + auto_trader_runs into a single dashboard view so users can
verify the bot is actually making them money.

Output is brand-safe per memory `project_greek_branding_2026_04_19`:
shows OUTCOMES (Sharpe, win-rate, drawdown, R-multiple) but NEVER
exposes the underlying model names (Qlib, HMM, TFT, FinBERT) to the
user. Per-model decomposition stays admin-only.

Per memory `project_no_fallbacks_no_refunds_2026_04_19` — when no live
trade history exists, returns honest zeros + "0 trades yet" tag rather
than showing synthetic backtest numbers as if they were live.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrackRecordSummary:
    """30/60/90-day rolling track record for one user (or system-wide)."""
    user_id: Optional[str]
    window_days: int
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    median_return_pct: float = 0.0
    total_pnl_inr: float = 0.0
    realised_sharpe: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    best_trade_pct: Optional[float] = None
    worst_trade_pct: Optional[float] = None
    avg_holding_days: Optional[float] = None
    profit_factor: Optional[float] = None
    last_trade_at: Optional[str] = None
    first_trade_at: Optional[str] = None
    source: str = "paper"        # 'paper' | 'live'
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "window_days": self.window_days,
            "trades_count": self.trades_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_return_pct": round(self.avg_return_pct, 4),
            "median_return_pct": round(self.median_return_pct, 4),
            "total_pnl_inr": round(self.total_pnl_inr, 2),
            "realised_sharpe": (
                round(self.realised_sharpe, 3) if self.realised_sharpe is not None else None
            ),
            "max_drawdown_pct": (
                round(self.max_drawdown_pct, 4) if self.max_drawdown_pct is not None else None
            ),
            "best_trade_pct": (
                round(self.best_trade_pct, 4) if self.best_trade_pct is not None else None
            ),
            "worst_trade_pct": (
                round(self.worst_trade_pct, 4) if self.worst_trade_pct is not None else None
            ),
            "avg_holding_days": (
                round(self.avg_holding_days, 2) if self.avg_holding_days is not None else None
            ),
            "profit_factor": (
                round(self.profit_factor, 3) if self.profit_factor is not None else None
            ),
            "last_trade_at": self.last_trade_at,
            "first_trade_at": self.first_trade_at,
            "source": self.source,
            "notes": self.notes,
        }


def _compute_sharpe(returns_pct: List[float], periods_per_year: int = 252) -> Optional[float]:
    """Annualised Sharpe ratio assuming `returns_pct` is per-trade %."""
    if len(returns_pct) < 5:
        return None
    avg = sum(returns_pct) / len(returns_pct)
    variance = sum((r - avg) ** 2 for r in returns_pct) / len(returns_pct)
    std = math.sqrt(variance)
    if std == 0:
        return None
    # Convert per-trade return → annualised by trade frequency proxy
    # (use 252 since most trades close within a week — caller can override)
    return (avg / std) * math.sqrt(periods_per_year)


def _max_drawdown_pct(equity_curve: List[float]) -> Optional[float]:
    """Max peak-to-trough drawdown as a negative percentage."""
    if not equity_curve:
        return None
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd
    return max_dd if max_dd < 0 else None


def _profit_factor(trade_pnls: List[float]) -> Optional[float]:
    """gross_profits / abs(gross_losses) — >1 means winners outweigh losers."""
    wins = sum(p for p in trade_pnls if p > 0)
    losses = abs(sum(p for p in trade_pnls if p < 0))
    if losses == 0:
        return None if wins == 0 else float("inf")
    return wins / losses


def aggregate_track_record(
    supabase: Any,
    *,
    user_id: Optional[str] = None,
    window_days: int = 30,
    source: str = "paper",
) -> TrackRecordSummary:
    """Compute realised track record over `window_days`.

    user_id=None → system-wide aggregate (admin view).
    source='paper' aggregates paper_trades. source='live' aggregates
    closed `trades` table rows. Per memory locks, paper is the primary
    safety-net feed; live aggregation only kicks in once users actually
    place live trades.
    """
    out = TrackRecordSummary(user_id=user_id, window_days=window_days, source=source)
    since = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()

    if source == "paper":
        table = "paper_trades"
        pnl_col = "pnl"
        pct_col = "pnl_pct"
    elif source == "live":
        table = "trades"
        pnl_col = "net_pnl"
        pct_col = "pnl_percent"
    else:
        out.notes.append(f"unknown source: {source}")
        return out

    try:
        q = supabase.table(table).select(
            f"id, {pnl_col}, {pct_col}, executed_at, action"
        ).gte("executed_at", since).not_.is_(pnl_col, "null")
        if user_id:
            q = q.eq("user_id", user_id)
        if source == "paper":
            # Only the SELL leg has realised pnl populated
            q = q.eq("action", "sell")
        else:
            q = q.eq("status", "closed")
        rows = q.execute().data or []
    except Exception as e:
        out.notes.append(f"query_failed: {str(e)[:200]}")
        return out

    if not rows:
        out.notes.append(f"no_{source}_trades_in_window")
        return out

    pnls = []
    pcts = []
    timestamps = []
    for r in rows:
        try:
            p = float(r.get(pnl_col) or 0)
            pct = float(r.get(pct_col) or 0)
            ts = r.get("executed_at")
            pnls.append(p)
            pcts.append(pct)
            if ts:
                timestamps.append(ts)
        except Exception:
            continue

    if not pcts:
        out.notes.append("zero_valid_pnl_rows")
        return out

    pcts_sorted = sorted(pcts)
    median = (
        pcts_sorted[len(pcts_sorted) // 2] if len(pcts_sorted) % 2
        else (pcts_sorted[len(pcts_sorted) // 2 - 1] + pcts_sorted[len(pcts_sorted) // 2]) / 2
    )

    # Build running equity curve from cumulative pnl for drawdown calc
    running = 0.0
    equity_curve = []
    for p in sorted(zip(timestamps, pnls)):
        running += p[1]
        equity_curve.append(running)

    out.trades_count = len(pcts)
    out.winning_trades = sum(1 for p in pcts if p > 0)
    out.losing_trades = sum(1 for p in pcts if p < 0)
    out.win_rate = out.winning_trades / out.trades_count if out.trades_count else 0.0
    out.avg_return_pct = sum(pcts) / len(pcts)
    out.median_return_pct = median
    out.total_pnl_inr = sum(pnls)
    out.realised_sharpe = _compute_sharpe(pcts)
    out.max_drawdown_pct = _max_drawdown_pct(equity_curve)
    out.best_trade_pct = max(pcts)
    out.worst_trade_pct = min(pcts)
    out.profit_factor = _profit_factor(pnls)
    out.first_trade_at = min(timestamps) if timestamps else None
    out.last_trade_at = max(timestamps) if timestamps else None

    return out


def daily_aggregate_and_persist(supabase_admin: Any) -> Dict[str, Any]:
    """Cron entry — runs daily after market close.

    Aggregates system-wide AutoPilot performance for 30/60/90-day
    windows and persists into `autopilot_track_record_daily` so the
    /autopilot/track-record page reads from a single fast table
    instead of recomputing from paper_trades on every request.

    Returns a small summary dict for the cron logger.
    """
    today = date.today().isoformat()
    out = {"date": today, "snapshots_written": 0, "errors": []}
    for w in (30, 60, 90):
        for src in ("paper", "live"):
            try:
                summary = aggregate_track_record(
                    supabase_admin, user_id=None, window_days=w, source=src,
                )
                row = summary.to_dict()
                row["snapshot_date"] = today
                supabase_admin.table("autopilot_track_record_daily").upsert(
                    row, on_conflict="snapshot_date,window_days,source"
                ).execute()
                out["snapshots_written"] += 1
            except Exception as e:
                out["errors"].append(f"{src}_{w}d: {str(e)[:200]}")
    return out
