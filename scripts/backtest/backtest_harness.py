#!/usr/bin/env python3
"""
Simple backtest harness for confluence-only strategy ranking.
- Data source: yfinance (daily)
- Entry: next-day open
- Exit: stop or target1 hit, else time exit

Usage:
  python scripts/backtest/backtest_harness.py --symbol RELIANCE.NS --period 5y --min-confluence 0.6 --hold-days 10
  python scripts/backtest/backtest_harness.py --universe-file data/alpha_universe.txt --period 5y
  python scripts/backtest/backtest_harness.py --symbols RELIANCE,TCS,INFY --period 5y
  python scripts/backtest/backtest_harness.py --universe-file data/alpha_universe.txt --data-source nsepy --period 5y
  python scripts/backtest/backtest_harness.py --symbol RELIANCE --data-source nsepy --start 2021-01-01 --end 2026-02-06
"""

from __future__ import annotations

import argparse
import sys
import time
import ssl
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.features.indicators import compute_all_indicators as calculate_all_70_features
from ml.scanner import get_all_strategies


def select_best_strategy(feat_df: pd.DataFrame, idx: int):
    """Select best strategy signal at given bar index.
    Returns (strategy, confluence) or (None, 0) if no signal fires."""
    strategies = get_all_strategies()
    best = None
    best_conf = 0.0
    for strategy in strategies:
        try:
            signal = strategy.check_entry(feat_df, idx)
            if signal and signal.confidence > best_conf:
                best = strategy
                best_conf = signal.confidence / 100.0  # normalize to 0-1
        except Exception:
            continue
    return best, best_conf


@dataclass
class Trade:
    symbol: str
    strategy: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    confluence: float
    regime: str
    hold_days: int


def _normalize_symbol(symbol: str, data_source: str) -> str:
    if data_source == "yfinance":
        if "." not in symbol:
            return f"{symbol}.NS"
        return symbol
    # nsepy expects NSE symbol without .NS suffix
    if symbol.endswith(".NS"):
        return symbol[:-3]
    return symbol


def _build_yf_session(user_agent: Optional[str], proxy: Optional[str], impersonate: Optional[str]):
    """
    Build a session compatible with yfinance (prefers curl_cffi if available).
    """
    try:
        from curl_cffi import requests as creq
        if impersonate:
            session = creq.Session(impersonate=impersonate)
        else:
            session = creq.Session()
    except Exception:
        # If curl_cffi is unavailable, let yfinance manage its own session
        return None

    if user_agent:
        try:
            session.headers.update({"User-Agent": user_agent})
        except Exception:
            pass
    if proxy:
        if hasattr(session, "proxies"):
            session.proxies.update({
                "http": proxy,
                "https": proxy,
            })
        elif hasattr(session, "set_proxy"):
            try:
                session.set_proxy(proxy)
            except Exception:
                pass
    return session


def fetch_ohlcv_yfinance(
    symbol: str,
    period: str,
    start: Optional[date],
    end: Optional[date],
    timeout: int,
    proxy: Optional[str],
    user_agent: Optional[str],
    impersonate: Optional[str],
    threads: bool,
) -> pd.DataFrame:
    session = _build_yf_session(user_agent, proxy, impersonate) if (user_agent or proxy or impersonate) else None
    if start and end:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            threads=threads,
            timeout=timeout,
            session=session,
        )
    else:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            progress=False,
            threads=threads,
            timeout=timeout,
            session=session,
        )
    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df


def _configure_nsepy_session(
    insecure: bool,
    force_tls12: bool,
    skip_symbolcount: bool,
    http_only: bool,
    no_redirect: bool,
    use_archives: bool,
) -> None:
    import requests
    from functools import partial
    from nsepy import urls, commons, history as nsepy_history

    class _NoRedirectSession(requests.Session):
        def request(self, *args, **kwargs):
            kwargs.setdefault("allow_redirects", False)
            return super().request(*args, **kwargs)

    session = _NoRedirectSession() if no_redirect else requests.Session()
    session.headers.update(urls.headers)
    session.verify = not insecure

    if force_tls12:
        try:
            from urllib3.util.ssl_ import create_urllib3_context
            ctx = create_urllib3_context()
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            adapter = requests.adapters.HTTPAdapter(ssl_context=ctx)
            session.mount("https://", adapter)
        except Exception:
            pass

    urls.session = session
    urls.URLFetchSession = partial(commons.URLFetch, session=session, headers=urls.headers)

    # Update existing URLFetch objects with new session/headers
    try:
        urls.symbol_count_url.set_session(session)
        urls.symbol_count_url.update_headers(urls.headers)
    except Exception:
        pass
    try:
        urls.equity_history_url_full.set_session(session)
        urls.equity_history_url_full.update_headers(urls.headers)
    except Exception:
        pass

    if use_archives:
        try:
            urls.symbol_count_url.url = "https://archives.nseindia.com/marketinfo/sym_map/symbolCount.jsp"
        except Exception:
            pass
        try:
            urls.equity_history_url_full.url = "https://archives.nseindia.com/products/dynaContent/common/productsSymbolMapping.jsp"
        except Exception:
            pass
    elif http_only:
        try:
            urls.symbol_count_url.url = "http://www1.nseindia.com/marketinfo/sym_map/symbolCount.jsp"
        except Exception:
            pass
        try:
            urls.equity_history_url_full.url = "http://www1.nseindia.com/products/dynaContent/common/productsSymbolMapping.jsp"
        except Exception:
            pass

    if skip_symbolcount:
        def _safe_symbol_count(symbol: str) -> str:
            return urls.symbol_count.get(symbol, "1")
        urls.get_symbol_count = _safe_symbol_count
        # ensure history.validate_params uses the patched function
        nsepy_history.get_symbol_count = _safe_symbol_count


def fetch_ohlcv_nsepy(
    symbol: str,
    start: date,
    end: date,
    chunk_days: int = 120,
    insecure: bool = False,
    force_tls12: bool = False,
    skip_symbolcount: bool = True,
    http_only: bool = False,
    no_redirect: bool = False,
    use_archives: bool = False,
) -> pd.DataFrame:
    try:
        from nsepy.history import get_history_quanta
    except ImportError as exc:
        raise ImportError("nsepy is required for NSE data. Install with `pip install nsepy`.") from exc
    if start > end:
        raise ValueError("start date must be <= end date")

    _configure_nsepy_session(
        insecure=insecure,
        force_tls12=force_tls12,
        skip_symbolcount=skip_symbolcount,
        http_only=http_only,
        no_redirect=no_redirect,
        use_archives=use_archives,
    )

    chunks: List[pd.DataFrame] = []
    current = start
    step = timedelta(days=chunk_days)  # avoid nsepy internal threading

    while current <= end:
        chunk_end = min(current + step, end)
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                df = get_history_quanta(
                    symbol=symbol,
                    start=current,
                    end=chunk_end,
                    index=False,
                    futures=False,
                    option_type="",
                    expiry_date=None,
                    strike_price="",
                    series="EQ",
                )
                if df is not None and not df.empty:
                    chunks.append(df)
                else:
                    # If empty, try a smaller window once
                    if (chunk_end - current).days > 30:
                        small_end = min(current + timedelta(days=30), chunk_end)
                        df_small = get_history_quanta(
                            symbol=symbol,
                            start=current,
                            end=small_end,
                            index=False,
                            futures=False,
                            option_type="",
                            expiry_date=None,
                            strike_price="",
                            series="EQ",
                        )
                        if df_small is not None and not df_small.empty:
                            chunks.append(df_small)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        if last_error is not None:
            raise last_error
        current = chunk_end + timedelta(days=1)

    if not chunks:
        raise ValueError(f"No data returned for {symbol} from nsepy")

    df = pd.concat(chunks, ignore_index=False)
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df = df.set_index("date")
    df = df.sort_index()
    return df


def fetch_ohlcv(
    symbol: str,
    period: str,
    data_source: str,
    start: Optional[date],
    end: Optional[date],
    yf_timeout: int,
    yf_proxy: Optional[str],
    yf_user_agent: Optional[str],
    yf_impersonate: Optional[str],
    yf_threads: bool,
) -> pd.DataFrame:
    if data_source == "nsepy":
        if not start or not end:
            raise ValueError("nsepy requires --start and --end dates")
        return fetch_ohlcv_nsepy(symbol, start, end)

    return fetch_ohlcv_yfinance(
        symbol,
        period,
        start,
        end,
        yf_timeout,
        yf_proxy,
        yf_user_agent,
        yf_impersonate,
        yf_threads,
    )


def fetch_index_history(
    index_symbol: str,
    period: str,
    start: Optional[date],
    end: Optional[date],
    yf_timeout: int,
    yf_proxy: Optional[str],
    yf_user_agent: Optional[str],
    yf_impersonate: Optional[str],
    yf_threads: bool,
) -> Optional[pd.DataFrame]:
    if not index_symbol:
        return None
    try:
        return fetch_ohlcv_yfinance(
            index_symbol,
            period,
            start,
            end,
            yf_timeout,
            yf_proxy,
            yf_user_agent,
            yf_impersonate,
            yf_threads,
        )
    except Exception as exc:
        print(f"Warning: failed to fetch index history for {index_symbol}: {exc}", file=sys.stderr)
        return None


def simulate_trades_from_features(
    symbol: str,
    feat_df: pd.DataFrame,
    min_confluence: float,
    hold_days: int,
    fee: float,
    stop_mult: float,
    target_mult: float,
    entry_start: Optional[pd.Timestamp] = None,
    entry_end: Optional[pd.Timestamp] = None,
) -> List[Trade]:
    trades: List[Trade] = []

    in_position = False
    exit_idx = None

    # Start after we have enough bars for long-history strategies
    start_idx = 200
    if len(feat_df) < start_idx + 2:
        return trades

    for idx in range(start_idx, len(feat_df) - 2):
        if in_position and exit_idx is not None and idx <= exit_idx:
            continue
        in_position = False
        exit_idx = None

        current_date = feat_df.index[idx]
        if entry_start is not None and current_date < entry_start:
            continue
        if entry_end is not None and current_date > entry_end:
            continue

        best_strategy, confluence = select_best_strategy(feat_df, idx)
        if not best_strategy:
            continue
        if confluence < min_confluence:
            continue
        regime = "NONE"

        # Entry at next-day open
        entry_idx = idx + 1
        entry_price_raw = float(feat_df["open"].iloc[entry_idx])
        entry_price = entry_price_raw * (1 + fee)

        levels = best_strategy.calculate_entry_stop_targets(feat_df, idx)
        risk = float(levels.get("risk", 0.0))
        if risk <= 0:
            continue

        stop = entry_price - (risk * stop_mult)
        target1 = entry_price + (risk * target_mult)

        exit_price = None
        exit_reason = None
        last_idx = min(entry_idx + hold_days, len(feat_df) - 1)

        for j in range(entry_idx, last_idx + 1):
            low = float(feat_df["low"].iloc[j])
            high = float(feat_df["high"].iloc[j])
            if low <= stop:
                exit_price = stop * (1 - fee)
                exit_reason = "stop"
                exit_idx = j
                break
            if high >= target1:
                exit_price = target1 * (1 - fee)
                exit_reason = "target1"
                exit_idx = j
                break

        if exit_price is None:
            exit_idx = last_idx
            exit_price = float(feat_df["close"].iloc[last_idx]) * (1 - fee)
            exit_reason = "time"

        pnl = exit_price - entry_price
        pnl_pct = (exit_price / entry_price - 1.0) * 100

        trades.append(
            Trade(
                symbol=symbol,
                strategy=best_strategy.name,
                entry_date=feat_df.index[entry_idx],
                exit_date=feat_df.index[exit_idx],
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason=exit_reason,
                confluence=confluence,
                regime=regime,
                hold_days=(exit_idx - entry_idx),
            )
        )

        in_position = True

    return trades


def simulate_trades(
    symbol: str,
    df: pd.DataFrame,
    min_confluence: float,
    hold_days: int,
    fee: float,
    stop_mult: float,
    target_mult: float,
    index_close: Optional[pd.Series] = None,
) -> List[Trade]:
    # Precompute strategy features
    feat_df = calculate_all_70_features(df, index_close=index_close)
    return simulate_trades_from_features(
        symbol,
        feat_df,
        min_confluence,
        hold_days,
        fee,
        stop_mult,
        target_mult,
    )


def summarize(trades: List[Trade]) -> Dict[str, float]:
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "total_pnl_pct": 0.0,
            "avg_hold_days": 0.0,
        }

    wins = [t for t in trades if t.pnl > 0]
    win_rate = (len(wins) / len(trades)) * 100
    avg_pnl_pct = float(np.mean([t.pnl_pct for t in trades]))
    total_pnl_pct = float(np.sum([t.pnl_pct for t in trades]))

    avg_hold_days = float(np.mean([t.hold_days for t in trades]))

    return {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "avg_pnl_pct": avg_pnl_pct,
        "total_pnl_pct": total_pnl_pct,
        "avg_hold_days": avg_hold_days,
    }


def summarize_by_symbol(trades: List[Trade]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if not trades:
        return summary

    symbols = sorted({t.symbol for t in trades})
    for sym in symbols:
        sym_trades = [t for t in trades if t.symbol == sym]
        summary[sym] = summarize(sym_trades)
    return summary


def summarize_by_strategy(trades: List[Trade]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if not trades:
        return summary

    strategies = sorted({t.strategy for t in trades})
    for name in strategies:
        strat_trades = [t for t in trades if t.strategy == name]
        summary[name] = summarize(strat_trades)
    return summary


def _parse_int_list(value: Optional[str], default: List[int]) -> List[int]:
    if not value:
        return default
    value = value.strip()
    if ":" in value:
        parts = [p.strip() for p in value.split(":")]
        if len(parts) != 3:
            raise ValueError("Expected range format start:end:step for integer list")
        start, end, step = (int(parts[0]), int(parts[1]), int(parts[2]))
        if step <= 0:
            raise ValueError("Step must be > 0 for integer list")
        return list(range(start, end + 1, step))
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_float_list(value: Optional[str], default: List[float]) -> List[float]:
    if not value:
        return default
    value = value.strip()
    if ":" in value:
        parts = [p.strip() for p in value.split(":")]
        if len(parts) != 3:
            raise ValueError("Expected range format start:end:step for float list")
        start, end, step = (float(parts[0]), float(parts[1]), float(parts[2]))
        if step <= 0:
            raise ValueError("Step must be > 0 for float list")
        vals: List[float] = []
        v = start
        # include end within tolerance
        while v <= end + 1e-9:
            vals.append(round(v, 6))
            v += step
        return vals
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _generate_sweep_configs(
    base_hold: int,
    base_stop: float,
    base_target: float,
    sweep_pct: float,
) -> List[Tuple[int, float, float]]:
    factors = [1.0 - sweep_pct, 1.0, 1.0 + sweep_pct]
    hold_days = sorted({max(1, int(round(base_hold * f))) for f in factors})
    stop_mults = [round(base_stop * f, 4) for f in factors]
    target_mults = [round(base_target * f, 4) for f in factors]
    configs: List[Tuple[int, float, float]] = []
    for hold in hold_days:
        for stop in stop_mults:
            for target in target_mults:
                configs.append((hold, stop, target))
    return configs


def _build_walk_forward_folds(
    index_df: Optional[pd.DataFrame],
    train_years: int,
    test_years: int,
    step_months: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if index_df is None or index_df.empty:
        return []
    if step_months <= 0:
        raise ValueError("wf_step_months must be > 0")
    dates = pd.to_datetime(index_df.index)
    start = dates.min().normalize()
    end = dates.max().normalize()
    folds: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)
        if test_end > end:
            break
        folds.append((train_start, train_end, test_start, test_end))
        cursor = cursor + pd.DateOffset(months=step_months)

    return folds

def _equity_metrics(trades: List[Trade], initial_capital: float, position_size: float) -> Dict[str, float]:
    if not trades:
        return {"final_equity": initial_capital, "total_return_pct": 0.0, "max_drawdown_pct": 0.0}
    curve = build_equity_curve(trades, initial_capital, position_size)
    if curve.empty:
        return {"final_equity": initial_capital, "total_return_pct": 0.0, "max_drawdown_pct": 0.0}
    max_dd = float(curve["drawdown_pct"].min())
    final_equity = float(curve["equity"].iloc[-1])
    total_return = (final_equity / initial_capital - 1.0) * 100
    return {
        "final_equity": final_equity,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd,
    }


def build_equity_curve(
    trades: List[Trade],
    initial_capital: float,
    position_size: float,
) -> pd.DataFrame:
    """
    Build a simple equity curve using trade exit dates.
    Assumes no overlap and uses current equity for position sizing.
    """
    if not trades:
        return pd.DataFrame(columns=["equity", "drawdown_pct"])

    trades_sorted = sorted(trades, key=lambda t: t.exit_date)
    dates = pd.date_range(trades_sorted[0].exit_date, trades_sorted[-1].exit_date, freq="D")
    equity = pd.Series(index=dates, dtype=float)

    current_equity = initial_capital
    trade_idx = 0

    for d in dates:
        # apply all trades exiting on this date
        while trade_idx < len(trades_sorted) and trades_sorted[trade_idx].exit_date.date() <= d.date():
            t = trades_sorted[trade_idx]
            pnl_cash = current_equity * position_size * (t.pnl_pct / 100.0)
            current_equity += pnl_cash
            trade_idx += 1
        equity.loc[d] = current_equity

    peak = equity.cummax()
    drawdown_pct = (equity - peak) / peak * 100
    curve = pd.DataFrame({"equity": equity, "drawdown_pct": drawdown_pct})
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy backtest harness")
    parser.add_argument("--symbol", help="Symbol (e.g., RELIANCE.NS)")
    parser.add_argument("--symbols", help="Comma-separated symbols (e.g., RELIANCE,TCS,INFY)")
    parser.add_argument("--universe-file", help="Path to symbol list file")
    parser.add_argument("--period", default="5y", help="History period (e.g., 2y, 5y)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (optional)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (optional)")
    parser.add_argument("--data-source", default="yfinance", choices=["yfinance", "nsepy"], help="OHLCV source")
    parser.add_argument("--index-symbol", default="^NSEI", help="Index symbol for RS benchmark (default ^NSEI)")
    parser.add_argument("--index-period", help="Index history period (defaults to --period)")
    parser.add_argument("--yf-timeout", type=int, default=30, help="yfinance request timeout (seconds)")
    parser.add_argument("--yf-proxy", help="Proxy URL for yfinance (e.g., http://user:pass@host:port)")
    parser.add_argument("--yf-user-agent", help="Custom User-Agent for yfinance requests")
    parser.add_argument("--yf-impersonate", default="chrome120", help="curl_cffi impersonate profile (e.g., chrome120)")
    parser.add_argument("--yf-no-threads", action="store_true", help="Disable yfinance threading")
    parser.add_argument("--nsepy-chunk-days", type=int, default=120, help="Chunk size (days) for nsepy requests")
    parser.add_argument("--nsepy-insecure", action="store_true", help="Disable SSL verification for nsepy")
    parser.add_argument("--nsepy-force-tls12", action="store_true", help="Force TLS1.2 for nsepy https calls")
    parser.add_argument("--nsepy-skip-symbolcount", action="store_true", help="Use local symbol_count cache only")
    parser.add_argument("--nsepy-http-only", action="store_true", help="Force HTTP endpoints for nsepy")
    parser.add_argument("--nsepy-no-redirect", action="store_true", help="Disable redirects for nsepy requests")
    parser.add_argument("--nsepy-archives", action="store_true", help="Use archives.nseindia.com endpoints")
    parser.add_argument("--min-confluence", type=float, default=0.6, help="Min confluence (0-1)")
    parser.add_argument("--hold-days", type=int, default=15, help="Time exit in days")
    parser.add_argument("--stop-mult", type=float, default=1.5, help="Stop multiplier vs base risk (1.0 = baseline)")
    parser.add_argument("--target-mult", type=float, default=2.5, help="Target multiplier vs base risk (2.0 = baseline)")
    parser.add_argument("--fee", type=float, default=0.0001, help="Fee per side (e.g., 0.0001 = 0.01%%)")
    parser.add_argument("--position-size", type=float, default=0.1, help="Position size as fraction of capital")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Initial capital for PnL estimate")
    parser.add_argument("--tune", action="store_true", help="Grid search hold-days/SL/TP multipliers")
    parser.add_argument("--tune-hold-days", default="5,7,10,15", help="Hold-days list (csv or start:end:step)")
    parser.add_argument("--tune-stop-mults", default="0.8,1.0,1.2,1.5", help="Stop multipliers list (csv or start:end:step)")
    parser.add_argument("--tune-target-mults", default="1.5,2.0,2.5,3.0", help="Target multipliers list (csv or start:end:step)")
    parser.add_argument("--tune-top", type=int, default=5, help="Show top N tuning configs")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward robustness test")
    parser.add_argument("--robustness", action="store_true", help="Alias for --walk-forward")
    parser.add_argument("--wf-train-years", type=int, default=3, help="Walk-forward train window (years)")
    parser.add_argument("--wf-test-years", type=int, default=1, help="Walk-forward test window (years)")
    parser.add_argument("--wf-step-months", type=int, default=6, help="Walk-forward step size (months)")
    parser.add_argument("--sweep-pct", type=float, default=0.2, help="Sweep percentage around base config (e.g., 0.2)")
    args = parser.parse_args()
    if args.robustness:
        args.walk_forward = True

    symbols: List[str] = []
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.universe_file:
        universe_path = Path(args.universe_file)
        if not universe_path.exists():
            raise FileNotFoundError(f"Universe file not found: {args.universe_file}")
        symbols = [line.strip() for line in universe_path.read_text().splitlines() if line.strip()]
    else:
        raise ValueError("Provide --symbol, --symbols, or --universe-file")

    all_trades: List[Trade] = []
    errors: List[Tuple[str, str]] = []

    start_date: Optional[date] = None
    end_date: Optional[date] = None
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    elif args.data_source == "nsepy":
        # derive start/end from period
        period = args.period.strip().lower()
        end_date = date.today()
        if period.endswith("y"):
            years = int(period[:-1])
            start_date = end_date - timedelta(days=365 * years)
        elif period.endswith("mo"):
            months = int(period[:-2])
            start_date = end_date - timedelta(days=30 * months)
        else:
            raise ValueError("For nsepy, provide --start/--end or use period like 1y/2y/5y.")

    index_period = args.index_period or args.period
    index_df = fetch_index_history(
        args.index_symbol,
        index_period,
        start_date,
        end_date,
        args.yf_timeout,
        args.yf_proxy,
        args.yf_user_agent,
        args.yf_impersonate,
        not args.yf_no_threads,
    )

    features_cache: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        try:
            normalized = _normalize_symbol(sym, args.data_source)
            if args.data_source == "nsepy":
                df = fetch_ohlcv_nsepy(
                    normalized,
                    start_date,
                    end_date,
                    chunk_days=args.nsepy_chunk_days,
                    insecure=args.nsepy_insecure,
                    force_tls12=args.nsepy_force_tls12,
                    skip_symbolcount=args.nsepy_skip_symbolcount,
                    http_only=args.nsepy_http_only,
                    no_redirect=args.nsepy_no_redirect,
                    use_archives=args.nsepy_archives,
                )
            else:
                df = fetch_ohlcv(
                    normalized,
                    args.period,
                    args.data_source,
                    start_date,
                    end_date,
                    args.yf_timeout,
                    args.yf_proxy,
                    args.yf_user_agent,
                    args.yf_impersonate,
                    not args.yf_no_threads,
                )
            aligned_index = None
            if index_df is not None and not index_df.empty:
                if "close" in index_df.columns:
                    idx_close = index_df["close"]
                elif "Close" in index_df.columns:
                    idx_close = index_df["Close"]
                else:
                    idx_close = index_df.iloc[:, 0]
                aligned_index = idx_close.reindex(df.index).ffill()
            feat_df = calculate_all_70_features(df, index_close=aligned_index)
            features_cache[normalized] = feat_df
        except Exception as exc:
            errors.append((sym, str(exc)))

    if args.walk_forward:
        if index_df is None or index_df.empty:
            raise SystemExit("Walk-forward requires index data. Ensure --index-symbol is available.")

        folds = _build_walk_forward_folds(
            index_df=index_df,
            train_years=args.wf_train_years,
            test_years=args.wf_test_years,
            step_months=args.wf_step_months,
        )
        if not folds:
            raise SystemExit("No walk-forward folds could be built from index history.")

        sweep_configs = _generate_sweep_configs(
            base_hold=args.hold_days,
            base_stop=args.stop_mult,
            base_target=args.target_mult,
            sweep_pct=args.sweep_pct,
        )

        warmup_bars = 260
        config_results: Dict[Tuple[int, float, float], List[Dict[str, float]]] = {
            cfg: [] for cfg in sweep_configs
        }

        for fold_id, (train_start, train_end, test_start, test_end) in enumerate(folds, start=1):
            print(
                f"\nWalk-Forward Fold {fold_id}: "
                f"train [{train_start.date()} - {train_end.date()}] "
                f"test [{test_start.date()} - {test_end.date()}]"
            )
            for hold_days, stop_mult, target_mult in sweep_configs:
                fold_trades: List[Trade] = []
                for sym, feat_df in features_cache.items():
                    if feat_df.index.max() < test_start or feat_df.index.min() > test_end:
                        continue
                    idx_start = feat_df.index.searchsorted(test_start)
                    if idx_start < warmup_bars:
                        continue
                    fold_trades.extend(
                        simulate_trades_from_features(
                            sym,
                            feat_df,
                            args.min_confluence,
                            hold_days,
                            args.fee,
                            stop_mult,
                            target_mult,
                            entry_start=test_start,
                            entry_end=test_end,
                        )
                    )

                summary = summarize(fold_trades)
                equity = _equity_metrics(fold_trades, args.initial_capital, args.position_size)
                config_results[(hold_days, stop_mult, target_mult)].append({**summary, **equity})

        print("\nWalk-Forward Robustness Summary")
        print("-" * 60)
        aggregated: List[Dict[str, float]] = []
        for (hold_days, stop_mult, target_mult), folds_metrics in config_results.items():
            if not folds_metrics:
                continue
            returns = np.array([m["total_return_pct"] for m in folds_metrics])
            dds = np.array([m["max_drawdown_pct"] for m in folds_metrics])
            win_rates = np.array([m["win_rate"] for m in folds_metrics])
            aggregated.append({
                "hold_days": hold_days,
                "stop_mult": stop_mult,
                "target_mult": target_mult,
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)),
                "mean_drawdown": float(np.mean(dds)),
                "worst_drawdown": float(np.min(dds)),
                "median_win_rate": float(np.median(win_rates)),
            })

        aggregated.sort(key=lambda r: (r["mean_return"], r["median_win_rate"]), reverse=True)
        for row in aggregated:
            print(
                f"hold={row['hold_days']:>2} stop={row['stop_mult']:.2f} target={row['target_mult']:.2f} | "
                f"mean_return {row['mean_return']:.2f}% (std {row['std_return']:.2f}) | "
                f"worst_dd {row['worst_drawdown']:.2f}% | median_win {row['median_win_rate']:.1f}%"
            )

        if errors:
            print("\nErrors (symbols skipped)")
            print("-" * 60)
            for sym, err in errors:
                print(f"{sym}: {err}")
        return

    if args.tune and features_cache:
        hold_days_list = _parse_int_list(args.tune_hold_days, [args.hold_days])
        stop_mults = _parse_float_list(args.tune_stop_mults, [args.stop_mult])
        target_mults = _parse_float_list(args.tune_target_mults, [args.target_mult])

        tuning_results: List[Dict[str, float]] = []
        for hold in hold_days_list:
            for stop_mult in stop_mults:
                for target_mult in target_mults:
                    trades: List[Trade] = []
                    for sym, feat_df in features_cache.items():
                        trades.extend(
                            simulate_trades_from_features(
                                sym,
                                feat_df,
                                args.min_confluence,
                                hold,
                                args.fee,
                                stop_mult,
                                target_mult,
                            )
                        )
                    summary = summarize(trades)
                    equity = _equity_metrics(trades, args.initial_capital, args.position_size)
                    tuning_results.append(
                        {
                            "hold_days": hold,
                            "stop_mult": stop_mult,
                            "target_mult": target_mult,
                            **summary,
                            **equity,
                        }
                    )

        tuning_results.sort(key=lambda r: (r["total_return_pct"], r["avg_pnl_pct"]), reverse=True)

        print("\nTuning Results (Top Configs by total_return_pct)")
        print("-" * 60)
        for row in tuning_results[: args.tune_top]:
            print(
                f"hold={int(row['hold_days'])} stop={row['stop_mult']:.2f} target={row['target_mult']:.2f} | "
                f"return {row['total_return_pct']:.2f}% | avg_pnl {row['avg_pnl_pct']:.2f}% | "
                f"win {row['win_rate']:.1f}% | trades {row['total_trades']}"
            )

        if tuning_results:
            best = tuning_results[0]
            args.hold_days = int(best["hold_days"])
            args.stop_mult = float(best["stop_mult"])
            args.target_mult = float(best["target_mult"])

    if features_cache:
        for sym, feat_df in features_cache.items():
            all_trades.extend(
                simulate_trades_from_features(
                    sym,
                    feat_df,
                    args.min_confluence,
                    args.hold_days,
                    args.fee,
                    args.stop_mult,
                    args.target_mult,
                )
            )

    summary = summarize(all_trades)
    sym_summary = summarize_by_symbol(all_trades)
    strat_summary = summarize_by_strategy(all_trades)

    print("\nBacktest Summary (All Symbols)")
    print("-" * 60)
    print(f"config: hold_days={args.hold_days}, stop_mult={args.stop_mult:.2f}, target_mult={args.target_mult:.2f}")
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"{key}: {val:.2f}")
        else:
            print(f"{key}: {val}")

    if all_trades:
        pnl_cash = sum((t.pnl_pct / 100.0) * args.initial_capital * args.position_size for t in all_trades)
        print(f"estimated_pnl_cash: {pnl_cash:.2f}")

        curve = build_equity_curve(all_trades, args.initial_capital, args.position_size)
        if not curve.empty:
            max_dd = curve["drawdown_pct"].min()
            final_equity = curve["equity"].iloc[-1]
            total_return = (final_equity / args.initial_capital - 1.0) * 100

            print("\nEquity Curve Summary")
            print("-" * 60)
            print(f"final_equity: {final_equity:.2f}")
            print(f"total_return_pct: {total_return:.2f}")
            print(f"max_drawdown_pct: {max_dd:.2f}")

        print("\nStrategy Breakdown (min 5 trades)")
        print("-" * 60)
        ranked_strats = []
        for name, stats in strat_summary.items():
            if stats["total_trades"] >= 5:
                ranked_strats.append((name, stats["avg_pnl_pct"], stats["total_trades"], stats["win_rate"]))
        ranked_strats.sort(key=lambda x: x[1], reverse=True)
        for name, avg_pnl, total_trades, win_rate in ranked_strats[:15]:
            print(f"{name:30s} | avg_pnl_pct {avg_pnl:6.2f} | trades {total_trades:4d} | win_rate {win_rate:5.1f}%")

        print("\nTop Symbols by Avg PnL% (min 5 trades)")
        print("-" * 60)
        ranked = []
        for sym, stats in sym_summary.items():
            if stats["total_trades"] >= 5:
                ranked.append((sym, stats["avg_pnl_pct"], stats["total_trades"], stats["win_rate"]))
        ranked.sort(key=lambda x: x[1], reverse=True)
        for sym, avg_pnl, total_trades, win_rate in ranked[:10]:
            print(f"{sym:12s} | avg_pnl_pct {avg_pnl:6.2f} | trades {total_trades:4d} | win_rate {win_rate:5.1f}%")

        print("\nSample Trades (first 5)")
        print("-" * 60)
        for t in all_trades[:5]:
            print(
                f"{t.symbol:10s} {t.entry_date.date()} -> {t.exit_date.date()} | {t.strategy} | "
                f"entry {t.entry_price:.2f} exit {t.exit_price:.2f} | "
                f"pnl {t.pnl_pct:.2f}% | {t.exit_reason} | conf {t.confluence:.2f}"
            )

    if errors:
        print("\nErrors (symbols skipped)")
        print("-" * 60)
        for sym, err in errors:
            print(f"{sym}: {err}")


if __name__ == "__main__":
    main()
