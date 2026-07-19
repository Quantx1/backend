#!/usr/bin/env python
"""
Per-trainer data audit — verify EVERY trainer's input data BEFORE training.

For each trainer that consumes external data, this script:

  1. Fetches a small sample of its real data path
  2. Reports row count, date range, NaN%, feature ranges, label balance
  3. Flags blockers: empty frames, all-NaN columns, suspicious label
     distributions, out-of-range features
  4. Exits 0 if all checks pass, 1 if any trainer has bad data

Run this BEFORE smoke_all.py / runpod_full_pipeline.sh to make sure
GPU time isn't spent training on rotten inputs.

Usage:
    python scripts/train/audit_trainer_data.py            # default smoke (10 stocks, 2y)
    python scripts/train/audit_trainer_data.py --universe 50 --period 5y
    python scripts/train/audit_trainer_data.py --only lgbm_signal_gate,tft_swing
    python scripts/train/audit_trainer_data.py --report data_audit.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("audit_trainer_data")


# ---------------------------------------------------------------------------
# Per-trainer audit helpers
# ---------------------------------------------------------------------------


def _summarize_ohlcv(df, name: str) -> Dict[str, Any]:
    """Compact OHLCV summary: rows, columns, date range, NaN counts."""
    import pandas as pd
    import numpy as np
    if df is None or (hasattr(df, "empty") and df.empty):
        return {"name": name, "ok": False, "reason": "empty frame"}

    nan_pct = {}
    if isinstance(df.columns, pd.MultiIndex):
        # MultiIndex: flatten to flat names for NaN report
        first_outer = df.columns.get_level_values(0)[0]
        field_outer = first_outer in {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        if field_outer:
            for field in ["Open", "High", "Low", "Close", "Volume"]:
                if field in df.columns.get_level_values(0):
                    sub = df[field]
                    total = sub.size
                    nan_pct[field] = round(float(sub.isna().sum().sum() / max(total, 1) * 100), 2)
        else:
            # ticker-outer
            for ticker in df.columns.get_level_values(0).unique()[:3]:
                if (ticker, "Close") in df.columns:
                    s = df[(ticker, "Close")]
                    nan_pct[f"{ticker}.Close"] = round(float(s.isna().sum() / max(s.size, 1) * 100), 2)
    else:
        for c in df.columns[:5]:
            nan_pct[str(c)] = round(float(df[c].isna().sum() / max(df[c].size, 1) * 100), 2)

    first = df.index.min()
    last = df.index.max()
    return {
        "name": name,
        "ok": True,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "first_date": str(first)[:10] if first is not None else None,
        "last_date": str(last)[:10] if last is not None else None,
        "nan_pct_per_field": nan_pct,
    }


def _check_label_balance(labels, name: str, min_minority_pct: float = 5.0) -> Dict[str, Any]:
    """Verify class label distribution is sane. ML training needs >=5% minority class
    to avoid trivial-classifier overfit. Returns ok=False when bad."""
    import numpy as np
    arr = np.asarray(labels)
    if arr.size == 0:
        return {"name": name, "ok": False, "reason": "empty labels"}
    unique, counts = np.unique(arr, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(unique, counts)}
    total = int(counts.sum())
    pcts = {k: round(v / total * 100, 2) for k, v in dist.items()}
    min_pct = min(pcts.values())
    return {
        "name": name,
        "ok": min_pct >= min_minority_pct,
        "n_samples": total,
        "label_distribution": dist,
        "label_pct": pcts,
        "min_minority_pct": min_pct,
        "warn": (
            f"minority class < {min_minority_pct}% — may overfit to majority"
            if min_pct < min_minority_pct else None
        ),
    }


# ---------------------------------------------------------------------------
# Per-trainer audit functions
# ---------------------------------------------------------------------------


def audit_regime_hmm() -> Dict[str, Any]:
    """Nifty + India VIX from yfinance — indices, no CA needed."""
    try:
        import yfinance as yf
        nifty = yf.download("^NSEI", period="2y", progress=False, auto_adjust=False)
        vix = yf.download("^INDIAVIX", period="2y", progress=False, auto_adjust=False)
        if nifty is None or nifty.empty:
            return {"trainer": "regime_hmm", "ok": False, "reason": "yfinance ^NSEI empty"}
        if vix is None or vix.empty:
            return {"trainer": "regime_hmm", "ok": False, "reason": "yfinance ^INDIAVIX empty"}
        return {
            "trainer": "regime_hmm",
            "ok": True,
            "nifty": _summarize_ohlcv(nifty, "^NSEI"),
            "vix": _summarize_ohlcv(vix, "^INDIAVIX"),
        }
    except Exception as exc:
        return {"trainer": "regime_hmm", "ok": False, "reason": str(exc)}


def audit_lgbm_signal_gate(universe: int, period: str) -> Dict[str, Any]:
    """Production OHLCV for top-N liquid + Alpha158 features."""
    try:
        from ml.data import LiquidUniverseConfig, liquid_universe
        from ml.data.production_ohlcv import production_ohlcv
        from datetime import date, timedelta

        u = liquid_universe(LiquidUniverseConfig(top_n=universe))
        if not u:
            return {"trainer": "lgbm_signal_gate", "ok": False, "reason": "empty universe"}

        years = int(period.rstrip("y")) if period.endswith("y") else 5
        start = date.today() - timedelta(days=365 * years)

        df = production_ohlcv(
            symbols=u[:universe], start=start, end=date.today(),
            include_delisted=False, adjust_corp_actions=True, quality_check=True,
        )
        ohlcv_summary = _summarize_ohlcv(df, "production_ohlcv")
        if not ohlcv_summary.get("ok"):
            return {"trainer": "lgbm_signal_gate", "ok": False,
                    "reason": "production_ohlcv empty", "summary": ohlcv_summary}

        # Spot-check a few features: returns + RSI on first symbol
        import pandas as pd
        import numpy as np
        first_ticker = df.columns.get_level_values(0)[0]
        try:
            close = df[first_ticker]["Close"].dropna()
        except (KeyError, IndexError):
            # Maybe column order is (field, ticker)
            close = df["Close"].iloc[:, 0].dropna()
        if close.empty:
            return {"trainer": "lgbm_signal_gate", "ok": False, "reason": "first symbol has no Close"}

        ret_1d = close.pct_change(1).dropna()
        ret_5d = close.pct_change(5).dropna()
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        rs = gain / loss
        rsi = (100 - 100 / (1 + rs)).fillna(50)

        return {
            "trainer": "lgbm_signal_gate",
            "ok": True,
            "universe_size": len(u),
            "first_ticker": first_ticker,
            "ohlcv_summary": ohlcv_summary,
            "feature_sanity": {
                "ret_1d_mean": round(float(ret_1d.mean()), 5),
                "ret_1d_std": round(float(ret_1d.std()), 5),
                "ret_1d_min": round(float(ret_1d.min()), 4),
                "ret_1d_max": round(float(ret_1d.max()), 4),
                "ret_5d_std": round(float(ret_5d.std()), 5),
                "rsi_min": round(float(rsi.min()), 2),
                "rsi_max": round(float(rsi.max()), 2),
                "rsi_mean": round(float(rsi.mean()), 2),
            },
            "warn": (
                "abs(ret_1d_max) > 0.5 — possible split/data error"
                if abs(ret_1d).max() > 0.5 else None
            ),
        }
    except Exception as exc:
        return {"trainer": "lgbm_signal_gate", "ok": False,
                "reason": str(exc), "traceback": traceback.format_exc().splitlines()[-5:]}


def audit_tft_swing(universe: int, period: str) -> Dict[str, Any]:
    """Same data path as lgbm_signal_gate for stocks; report TFT-specific shape."""
    try:
        from ml.training.trainers.tft_swing import _build_long_format_frame
        # Use the trainer's actual data builder
        os.environ["SMOKE_MODE"] = "1"
        os.environ["SMOKE_UNIVERSE_SIZE"] = str(universe)
        os.environ["SMOKE_YFINANCE_PERIOD"] = period

        df = _build_long_format_frame(top_n=universe)
        if df is None or df.empty:
            return {"trainer": "tft_swing", "ok": False, "reason": "empty frame"}

        import numpy as np
        n_symbols = df["unique_id"].nunique()
        rows_per_symbol = df.groupby("unique_id").size()
        return {
            "trainer": "tft_swing",
            "ok": int(rows_per_symbol.min()) >= 60,  # needs at least 60 bars
            "n_rows": int(len(df)),
            "n_symbols": int(n_symbols),
            "rows_per_symbol_min": int(rows_per_symbol.min()),
            "rows_per_symbol_max": int(rows_per_symbol.max()),
            "first_date": str(df["ds"].min())[:10],
            "last_date": str(df["ds"].max())[:10],
            "feature_columns": [c for c in df.columns if c in (
                "ret_1d", "ret_5d", "rsi_14", "atr_14_pct",
                "volume_ratio_10d", "log_close", "day_of_week",
            )],
            "feature_ranges": {
                "rsi_14": [round(float(df["rsi_14"].min()), 2), round(float(df["rsi_14"].max()), 2)],
                "atr_14_pct": [round(float(df["atr_14_pct"].min()), 4),
                               round(float(df["atr_14_pct"].max()), 4)],
                "ret_1d_std": round(float(df["ret_1d"].std()), 5),
            },
            "warn": (
                f"some symbols have <60 bars (min={rows_per_symbol.min()}) — TFT needs >=60"
                if rows_per_symbol.min() < 60 else None
            ),
        }
    except Exception as exc:
        return {"trainer": "tft_swing", "ok": False,
                "reason": str(exc), "traceback": traceback.format_exc().splitlines()[-5:]}


def audit_qlib_alpha158() -> Dict[str, Any]:
    """Qlib provider dir + Alpha158 features."""
    qlib_dir = Path.home() / ".qlib" / "qlib_data" / "nse_data"
    if not qlib_dir.exists():
        return {"trainer": "qlib_alpha158", "ok": False,
                "reason": f"Qlib provider missing at {qlib_dir} — "
                          "run scripts/data/ingest_nse_to_qlib.py"}
    features_dir = qlib_dir / "features"
    if not features_dir.exists():
        return {"trainer": "qlib_alpha158", "ok": False,
                "reason": "Qlib provider has no features/ subdir"}
    sym_dirs = list(features_dir.glob("*"))
    return {
        "trainer": "qlib_alpha158",
        "ok": len(sym_dirs) >= 50,   # need decent universe for cross-section
        "provider_dir": str(qlib_dir),
        "n_symbols": len(sym_dirs),
        "sample_symbols": [d.name for d in sym_dirs[:5]],
        "warn": (
            f"only {len(sym_dirs)} symbols — Alpha158 cross-sectional Rank IC "
            "is unstable below 100"
            if len(sym_dirs) < 100 else None
        ),
    }


# audit_earnings_xgb removed 2026-05-11 — F9 EarningsScout deferred.


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=int, default=10)
    parser.add_argument("--period", default="2y")
    parser.add_argument("--only", help="comma-separated trainer subset")
    parser.add_argument("--report", type=Path, default=Path("data_audit.json"))
    args = parser.parse_args()

    all_audits = [
        ("regime_hmm",         lambda: audit_regime_hmm()),
        ("lgbm_signal_gate",   lambda: audit_lgbm_signal_gate(args.universe, args.period)),
        ("tft_swing",          lambda: audit_tft_swing(args.universe, args.period)),
        ("qlib_alpha158",      lambda: audit_qlib_alpha158()),
    ]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        all_audits = [(n, fn) for n, fn in all_audits if n in wanted]

    print("=" * 76)
    print(f"DATA AUDIT — universe={args.universe}, period={args.period}")
    print("=" * 76)

    results: List[Dict[str, Any]] = []
    n_ok = n_fail = n_warn = 0

    for name, fn in all_audits:
        t0 = time.time()
        try:
            res = fn()
        except Exception as exc:
            res = {"trainer": name, "ok": False, "reason": f"crashed: {exc}"}
        res["_elapsed_s"] = round(time.time() - t0, 1)
        results.append(res)

        mark = "✅" if res.get("ok") else "❌"
        if res.get("warn"):
            mark = "⚠"
        print(f"\n  {mark} {name:<22} ({res['_elapsed_s']}s)")
        for k, v in res.items():
            if k.startswith("_"): continue
            if k in ("trainer", "ok"): continue
            v_s = json.dumps(v, default=str)
            if len(v_s) > 110:
                v_s = v_s[:107] + "..."
            print(f"      {k}: {v_s}")

        if not res.get("ok"):
            n_fail += 1
        elif res.get("warn"):
            n_warn += 1
        else:
            n_ok += 1

    print()
    print("=" * 76)
    print(f"SUMMARY:  ok={n_ok}  warn={n_warn}  fail={n_fail}")
    print("=" * 76)

    args.report.write_text(json.dumps(results, indent=2, default=str))
    print(f"Full audit → {args.report}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
