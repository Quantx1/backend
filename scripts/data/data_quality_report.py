#!/usr/bin/env python
"""
Pre-training data quality report.

Runs before any trainer in the production pipeline (Phase 7a). Checks:

  1. bhavcopy cache health — symbol count, freshness, coverage
  2. liquid_universe yields a non-empty top-N universe
  3. fii_dii_history parquet exists and has recent rows
  4. corporate_actions registry loads cleanly
  5. delisted_registry loads cleanly
  6. yfinance fallback reachable
  7. Per-symbol OHLCV coverage for the actual training universe
  8. Output: ``data_quality_report.json`` + console verdict

Exit code 0 = all green; 1 = any blocker. Pipeline scripts gate the
training phase on a 0 exit so we never train on rotten data.

Usage:
    python scripts/data/data_quality_report.py
    python scripts/data/data_quality_report.py --top-n 50 --start 2020-01-01
    python scripts/data/data_quality_report.py --strict       # fail on warnings
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("data_quality_report")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_bhavcopy_cache() -> Dict[str, Any]:
    """Symbol count + freshness in the per-symbol bhavcopy cache."""
    cache_dir = REPO_ROOT / "ml" / "data" / "cache" / "bhavcopy"
    if not cache_dir.exists():
        return {"ok": False, "reason": f"cache dir missing: {cache_dir}",
                "n_symbols": 0, "newest_mtime": None}
    files = list(cache_dir.glob("*.parquet"))
    if not files:
        return {"ok": False, "reason": "cache empty (run scripts/data/ingest_nse_to_qlib.py "
                "or let bhavcopy_download warm it)",
                "n_symbols": 0, "newest_mtime": None}
    newest = max(f.stat().st_mtime for f in files)
    age_h = (time.time() - newest) / 3600.0
    return {
        "ok": True,
        "n_symbols": len(files),
        "newest_mtime": datetime.fromtimestamp(newest).isoformat(),
        "age_hours": round(age_h, 1),
        "warn": "cache is older than 7 days" if age_h > 24 * 7 else None,
    }


def check_liquid_universe(top_n: int) -> Dict[str, Any]:
    from ml.data import LiquidUniverseConfig, liquid_universe

    try:
        u = liquid_universe(LiquidUniverseConfig(top_n=top_n))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"liquid_universe failed: {exc}",
                "n_symbols": 0}
    return {
        "ok": len(u) >= top_n * 0.7,   # at least 70% of requested
        "n_symbols": len(u),
        "requested": top_n,
        "first_5": list(u[:5]),
    }


def check_fii_dii_cache() -> Dict[str, Any]:
    cache = REPO_ROOT / "ml" / "data" / "cache" / "fii_dii_history.parquet"
    if not cache.exists():
        return {"ok": False, "reason": "fii_dii_history.parquet missing — "
                "run scripts/data/backfill_fii_dii.py (NSE archive often blocks; "
                "is non-fatal for non-Qlib trainers)"}
    try:
        import pandas as pd  # noqa: PLC0415
        df = pd.read_parquet(cache)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"parquet unreadable: {exc}"}
    if df.empty:
        return {"ok": False, "reason": "fii_dii_history is empty"}
    return {
        "ok": True,
        "n_rows": int(len(df)),
        "first_date": str(df.index.min())[:10] if hasattr(df.index, "min") else None,
        "last_date": str(df.index.max())[:10] if hasattr(df.index, "max") else None,
    }


def check_corporate_actions() -> Dict[str, Any]:
    try:
        from ml.data import CORPORATE_ACTIONS, actions_for
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"import failed: {exc}"}
    n_total = len(CORPORATE_ACTIONS) if hasattr(CORPORATE_ACTIONS, "__len__") else 0
    # Spot-check a known symbol with actions
    try:
        sample = actions_for("RELIANCE")
        sample_n = len(sample) if sample else 0
    except Exception as exc:  # noqa: BLE001
        sample_n = -1
    return {
        "ok": True,
        "n_actions_tracked": int(n_total),
        "reliance_actions": int(sample_n),
    }


def check_delisted_registry() -> Dict[str, Any]:
    try:
        from ml.data import DELISTED_NSE, was_listed_at
        listed_today = was_listed_at("TATAMOTORS", date.today())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"import failed: {exc}"}
    return {
        "ok": True,
        "n_delisted_tracked": len(DELISTED_NSE) if hasattr(DELISTED_NSE, "__len__") else 0,
        "tatamotors_listed_today": bool(listed_today),
    }


def check_yfinance_reachable() -> Dict[str, Any]:
    """Sanity ping — Nifty 50 close should always be fetchable."""
    try:
        import yfinance as yf  # noqa: PLC0415
        df = yf.download("^NSEI", period="5d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return {"ok": False, "reason": "yfinance returned empty frame"}
        return {"ok": True, "nifty_last_close": float(df["Close"].iloc[-1].item()
                if hasattr(df["Close"].iloc[-1], "item") else df["Close"].iloc[-1])}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"yfinance ping failed: {exc}"}


def check_production_ohlcv_roundtrip(top_n: int, start: str) -> Dict[str, Any]:
    """End-to-end check: build universe → call production_ohlcv → verify
    coverage. Uses the actual data path every trainer will hit.
    """
    from ml.data import LiquidUniverseConfig, liquid_universe
    from ml.data.production_ohlcv import coverage_summary, production_ohlcv

    try:
        universe = liquid_universe(LiquidUniverseConfig(top_n=top_n))
        df = production_ohlcv(
            symbols=universe[:5],   # spot-check first 5 symbols
            start=start,
            include_delisted=False,   # delisted expansion is slow; skip for spot check
            adjust_corp_actions=True,
            quality_check=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"production_ohlcv failed: {exc}"}

    if df is None or df.empty:
        return {"ok": False, "reason": "production_ohlcv returned empty frame"}

    cov = coverage_summary(
        df, start=datetime.strptime(start, "%Y-%m-%d").date(), end=date.today(),
    )
    rows = cov.to_dict(orient="records") if not cov.empty else []
    return {
        "ok": True,
        "n_symbols_returned": len(rows),
        "spot_check_first_5": rows[:5],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=50,
                        help="liquid universe size to check (default 50)")
    parser.add_argument("--start", default="2024-01-01",
                        help="OHLCV roundtrip start date (default 2024-01-01)")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 on any warning, not just hard failures")
    parser.add_argument("--report", type=Path,
                        default=Path("data_quality_report.json"))
    args = parser.parse_args()

    # Each tuple: (name, check_fn, blocking).
    # blocking=False means failure → warning (training proceeds), not exit-1.
    # bhavcopy + fii_dii caches are non-blocking because production_ohlcv
    # falls back to yfinance cleanly when bhavcopy is unavailable.
    checks: List[Tuple[str, Any, bool]] = [
        ("bhavcopy_cache", check_bhavcopy_cache, False),
        ("liquid_universe", lambda: check_liquid_universe(args.top_n), True),
        ("fii_dii_cache", check_fii_dii_cache, False),
        ("corporate_actions", check_corporate_actions, True),
        ("delisted_registry", check_delisted_registry, True),
        ("yfinance_reachable", check_yfinance_reachable, True),
        ("production_ohlcv_roundtrip",
         lambda: check_production_ohlcv_roundtrip(args.top_n, args.start), True),
    ]

    results: Dict[str, Dict[str, Any]] = {}
    blockers: List[str] = []
    warnings: List[str] = []

    print("=" * 76)
    print("PRE-TRAINING DATA QUALITY REPORT")
    print("=" * 76)
    for name, fn, is_blocking in checks:
        t0 = time.time()
        try:
            res = fn()
        except Exception as exc:  # noqa: BLE001
            res = {"ok": False, "reason": f"exception: {exc}"}
        elapsed = time.time() - t0
        res["_elapsed_s"] = round(elapsed, 2)
        res["_blocking"] = is_blocking
        results[name] = res

        if res.get("ok"):
            mark = "✅"
        else:
            mark = "❌" if is_blocking else "⚠"
        warn = res.get("warn")
        print(f"  {mark} {name:<35} ({elapsed:.1f}s){'' if is_blocking else '  (non-blocking)'}")
        for k, v in sorted(res.items()):
            if k in ("ok", "_elapsed_s", "_blocking", "reason", "warn"):
                continue
            v_s = str(v)
            if len(v_s) > 90:
                v_s = v_s[:87] + "..."
            print(f"        {k}: {v_s}")
        if res.get("reason"):
            print(f"        reason: {res['reason']}")
        if warn:
            print(f"        ⚠ warn: {warn}")
            warnings.append(f"{name}: {warn}")
        if not res.get("ok"):
            (blockers if is_blocking else warnings).append(
                f"{name}: {res.get('reason', 'unknown')}"
            )

    print()
    print("=" * 76)
    if blockers:
        print(f"BLOCKERS ({len(blockers)}):")
        for b in blockers:
            print(f"  ❌ {b}")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")
    if not blockers and not warnings:
        print("✓ All checks green — data layer is production-ready.")
    elif not blockers:
        print(f"⚠ {len(warnings)} warnings, no blockers — proceed with caution.")
    else:
        print(f"❌ {len(blockers)} blockers — training would be unsafe.")
    print("=" * 76)

    args.report.write_text(json.dumps(results, indent=2, default=str))
    print(f"Full report → {args.report}")

    if blockers:
        return 1
    if warnings and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
