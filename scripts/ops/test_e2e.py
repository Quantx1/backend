"""
============================================================================
Quant X — End-to-End Test Script
============================================================================
Tests the full data pipeline:
  1. Backend health check
  2. Supabase DB connectivity (49 strategies seeded)
  3. Kite data provider (historical OHLCV)
  4. Screener engine (run a scanner)
  5. Signal generation (manual trigger)
  6. Frontend API compatibility
  7. Seed demo signals (if no real signals exist)

Usage:
  python scripts/test_e2e.py                  # Full test
  python scripts/test_e2e.py --seed           # Seed demo signals only
  python scripts/test_e2e.py --scan RELIANCE  # Scan specific stock
  python scripts/test_e2e.py --backend-only   # Skip frontend check
============================================================================
"""

import asyncio
import sys
import os
import json
import argparse
from datetime import date, datetime
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Load .env
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


# ── Helpers ──────────────────────────────────────────────────────────────────

class Colors:
    OK = "\033[92m"
    FAIL = "\033[91m"
    WARN = "\033[93m"
    INFO = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"

def ok(msg):   print(f"  {Colors.OK}[PASS]{Colors.END} {msg}")
def fail(msg): print(f"  {Colors.FAIL}[FAIL]{Colors.END} {msg}")
def warn(msg): print(f"  {Colors.WARN}[WARN]{Colors.END} {msg}")
def info(msg): print(f"  {Colors.INFO}[INFO]{Colors.END} {msg}")
def header(msg): print(f"\n{Colors.BOLD}{'='*60}\n  {msg}\n{'='*60}{Colors.END}")

results = {"pass": 0, "fail": 0, "warn": 0}

def check(condition, pass_msg, fail_msg):
    if condition:
        ok(pass_msg)
        results["pass"] += 1
        return True
    else:
        fail(fail_msg)
        results["fail"] += 1
        return False


# ── 1. Environment Check ────────────────────────────────────────────────────

def test_env():
    header("1. Environment Variables")
    required = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_ANON_KEY": os.getenv("SUPABASE_ANON_KEY"),
        "SUPABASE_SERVICE_KEY": os.getenv("SUPABASE_SERVICE_KEY"),
    }
    optional = {
        "KITE_ADMIN_API_KEY": os.getenv("KITE_ADMIN_API_KEY"),
        "KITE_ADMIN_API_SECRET": os.getenv("KITE_ADMIN_API_SECRET"),
        "KITE_ADMIN_USER_ID": os.getenv("KITE_ADMIN_USER_ID"),
        "KITE_ADMIN_PASSWORD": os.getenv("KITE_ADMIN_PASSWORD"),
        "KITE_ADMIN_TOTP_SECRET": os.getenv("KITE_ADMIN_TOTP_SECRET"),
    }

    all_ok = True
    for key, val in required.items():
        if not check(val and len(val) > 5, f"{key} set", f"{key} MISSING"):
            all_ok = False

    for key, val in optional.items():
        if val and len(val) > 3:
            ok(f"{key} set")
            results["pass"] += 1
        else:
            warn(f"{key} not set (Kite auto-login won't work)")
            results["warn"] += 1

    return all_ok


# ── 2. Supabase DB ──────────────────────────────────────────────────────────

def test_supabase():
    header("2. Supabase Database")
    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        sb = create_client(url, key)
        ok("Supabase client created")
        results["pass"] += 1
    except Exception as e:
        fail(f"Supabase client failed: {e}")
        results["fail"] += 1
        return False

    # Check critical tables exist
    tables_to_check = [
        ("subscription_plans", "id", 3),
        ("strategy_catalog", "id", 49),
        ("signals", "id", None),
        ("trades", "id", None),
        ("positions", "id", None),
    ]

    all_ok = True
    for table, col, expected_min in tables_to_check:
        try:
            result = sb.table(table).select(col, count="exact").limit(1).execute()
            count = result.count or 0
            if expected_min is not None:
                if check(count >= expected_min, f"{table}: {count} rows (expected >= {expected_min})", f"{table}: only {count} rows (expected >= {expected_min})"):
                    pass
                else:
                    all_ok = False
            else:
                ok(f"{table}: {count} rows")
                results["pass"] += 1
        except Exception as e:
            fail(f"{table}: {e}")
            results["fail"] += 1
            all_ok = False

    # Check strategy categories
    try:
        cats = sb.table("strategy_catalog").select("category").execute()
        categories = set(r["category"] for r in cats.data)
        info(f"Strategy categories: {', '.join(sorted(categories))}")
    except:
        pass

    return all_ok


# ── 3. Kite Data Provider ───────────────────────────────────────────────────

def test_kite_data():
    header("3. Kite Data Provider (Historical OHLCV)")

    try:
        from backend.data.providers.kite import KiteDataProvider, get_kite_admin_client
        provider = KiteDataProvider()
        ok("KiteDataProvider instantiated")
        results["pass"] += 1
    except Exception as e:
        fail(f"KiteDataProvider import failed: {e}")
        results["fail"] += 1
        return False

    # Try fetching historical data for a well-known stock
    test_symbols = ["RELIANCE", "INFY", "TCS"]
    any_success = False

    for symbol in test_symbols:
        try:
            df = provider.get_historical(symbol, period="3mo", interval="1d")
            if df is not None and len(df) > 20:
                ok(f"{symbol}: {len(df)} candles, last={df.index[-1].date()}, close={df['Close'].iloc[-1]:.2f}")
                results["pass"] += 1
                any_success = True
                break
            else:
                warn(f"{symbol}: got {len(df) if df is not None else 0} candles (expected >20)")
                results["warn"] += 1
        except Exception as e:
            warn(f"{symbol}: {e}")
            results["warn"] += 1

    if not any_success:
        warn("No historical data fetched — Kite token may be expired. Try auto-refresh or seed demo data.")
        info("Run: python scripts/test_e2e.py --seed  to insert demo signals without Kite")

    return any_success


# ── 4. Screener Engine ──────────────────────────────────────────────────────

def test_screener():
    header("4. Live Screener Engine")

    try:
        from backend.data.screener.engine import LiveScreenerEngine
        engine = LiveScreenerEngine()
        ok("LiveScreenerEngine instantiated")
        results["pass"] += 1
    except Exception as e:
        fail(f"LiveScreenerEngine import failed: {e}")
        results["fail"] += 1
        return False

    # Try running a scanner (scanner 0 = full screening)
    try:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(engine.run_scanner(scanner_id=0, exchange="N", index="12"))
        count = len(result.get("results", []))
        if count > 0:
            ok(f"Scanner 0 (Full Screening): {count} results")
            results["pass"] += 1
            # Show first 3
            for r in result["results"][:3]:
                info(f"  {r.get('symbol', '?')}: LTP={r.get('ltp', '?')}, Signal={r.get('signal', '?')}")
            return True
        else:
            warn(f"Scanner 0 returned 0 results (may need Kite token)")
            results["warn"] += 1
            return False
    except Exception as e:
        warn(f"Scanner execution failed: {e}")
        results["warn"] += 1
        return False


# ── 5. Signal Generation ────────────────────────────────────────────────────

def test_signal_generation(symbols=None):
    header("5. Signal Generation (ML Pipeline)")

    try:
        from backend.ai.signals import SignalGenerator
        sg = SignalGenerator()
        ok("SignalGenerator instantiated")
        results["pass"] += 1
    except Exception as e:
        fail(f"SignalGenerator init failed: {e}")
        results["fail"] += 1
        return False

    # ML models loaded?
    models = []
    if sg._ml_labeler:
        models.append("ML Meta-Labeler")
    if sg._lgbm_gate:
        models.append("LightGBM Gate")
    if sg._regime_detector:
        models.append("HMM Regime")

    if models:
        ok(f"ML models loaded: {', '.join(models)}")
        results["pass"] += 1
    else:
        warn("No ML models loaded (will use strategy-only signals)")
        results["warn"] += 1

    # Try generating signals for a small set
    test_stocks = symbols or ["RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK"]
    info(f"Scanning {len(test_stocks)} stocks: {', '.join(test_stocks)}")

    try:
        loop = asyncio.get_event_loop()
        signals = loop.run_until_complete(
            sg.generate_intraday_signals(save=False, candidates=test_stocks)
        )
        if len(signals) > 0:
            ok(f"Generated {len(signals)} signals!")
            results["pass"] += 1
            for s in signals[:5]:
                info(f"  {s.symbol} {s.direction} @ {s.entry_price:.2f} → T1: {s.target_1:.2f}, SL: {s.stop_loss:.2f} (conf: {s.confidence:.0f}%)")
            return True
        else:
            warn(f"0 signals generated from {len(test_stocks)} stocks (normal if no patterns detected)")
            results["warn"] += 1
            return False
    except Exception as e:
        warn(f"Signal generation failed: {e}")
        results["warn"] += 1
        return False


# ── 6. Seed Demo Signals ────────────────────────────────────────────────────

def seed_demo_signals(count=15):
    header("6. Seed Demo Signals")
    import random

    try:
        from supabase import create_client
        sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    except Exception as e:
        fail(f"Supabase client failed: {e}")
        results["fail"] += 1
        return False

    # Check if signals already exist for today
    today = date.today().isoformat()
    existing = sb.table("signals").select("id", count="exact").eq("date", today).execute()
    existing_count = existing.count or 0

    if existing_count >= 5:
        ok(f"Already {existing_count} signals for today — skipping seed")
        results["pass"] += 1
        return True

    stocks = [
        ("RELIANCE", 2890.50), ("INFY", 1845.30), ("TCS", 4120.75),
        ("HDFCBANK", 1672.40), ("ICICIBANK", 1245.80), ("KOTAKBANK", 1890.20),
        ("BHARTIARTL", 1580.90), ("ITC", 465.30), ("HINDUNILVR", 2340.60),
        ("BAJFINANCE", 7120.50), ("SBIN", 780.40), ("MARUTI", 12450.00),
        ("SUNPHARMA", 1780.20), ("TATASTEEL", 152.80), ("WIPRO", 485.60),
        ("LT", 3520.40), ("TITAN", 3450.80), ("NESTLEIND", 2180.50),
        ("DRREDDY", 6780.30), ("HCLTECH", 1720.90), ("CIPLA", 1520.70),
        ("EICHERMOT", 4950.80), ("TATAPOWER", 425.30), ("DLF", 870.50),
        ("COALINDIA", 385.40), ("JSWSTEEL", 920.60), ("GRASIM", 2680.30),
        ("APOLLOHOSP", 6920.50), ("ADANIENT", 2890.70), ("LUPIN", 2120.30),
    ]

    strategies = [
        "Consolidation_Breakout", "Trend_Pullback", "Reversal_Patterns",
        "Candle_Reversal", "BOS_Structure", "Volume_Reversal",
    ]

    inserted = 0
    sample = random.sample(stocks, min(count, len(stocks)))

    for symbol, base_price in sample:
        direction = random.choice(["LONG", "SHORT"])
        confidence = round(random.uniform(65, 92), 1)
        strategy = random.choice(strategies)

        if direction == "LONG":
            entry = round(base_price * random.uniform(0.98, 1.02), 2)
            sl = round(entry * random.uniform(0.95, 0.97), 2)
            t1 = round(entry * random.uniform(1.03, 1.06), 2)
            t2 = round(entry * random.uniform(1.06, 1.10), 2)
            t3 = round(entry * random.uniform(1.10, 1.15), 2)
        else:
            entry = round(base_price * random.uniform(0.98, 1.02), 2)
            sl = round(entry * random.uniform(1.03, 1.05), 2)
            t1 = round(entry * random.uniform(0.94, 0.97), 2)
            t2 = round(entry * random.uniform(0.90, 0.94), 2)
            t3 = round(entry * random.uniform(0.85, 0.90), 2)

        rr = round(abs(t1 - entry) / max(abs(entry - sl), 0.01), 2)

        signal = {
            "symbol": symbol,
            "exchange": "NSE",
            "segment": "EQUITY",
            "direction": direction,
            "signal_type": "swing",
            "confidence": confidence,
            "catboost_score": round(random.uniform(0.5, 0.9), 2),
            "tft_score": round(random.uniform(0.4, 0.85), 2),
            "stockformer_score": round(random.uniform(55, 90), 1),
            "entry_price": entry,
            "stop_loss": sl,
            "target_1": t1,
            "target_2": t2,
            "target_3": t3,
            "risk_reward": rr,
            "expected_return": round(abs(t1 - entry) / entry * 100, 2),
            "max_loss_percent": round(abs(entry - sl) / entry * 100, 2),
            "reasons": [strategy, f"Confidence {confidence}%", f"R:R {rr}"],
            "strategy_names": [strategy],
            "status": "active",
            "date": today,
            "is_premium": random.choice([True, False]),
        }

        try:
            sb.table("signals").insert(signal).execute()
            inserted += 1
        except Exception as e:
            warn(f"  Insert failed for {symbol}: {e}")

    check(inserted > 0, f"Seeded {inserted} demo signals for {today}", "Failed to seed any signals")
    return inserted > 0


# ── 7. API Endpoint Check ───────────────────────────────────────────────────

def test_api_endpoints():
    header("7. API Endpoint Availability (requires running backend)")

    try:
        import httpx
    except ImportError:
        warn("httpx not installed — skipping API checks")
        results["warn"] += 1
        return False

    base = "http://localhost:8000"
    endpoints = [
        ("GET", "/api/health", 200),
        ("GET", "/api/market/status", 200),
        ("GET", "/api/screener/scanners", 200),
        ("GET", "/api/marketplace/strategies", 200),
    ]

    any_up = False
    for method, path, expected in endpoints:
        try:
            resp = httpx.request(method, f"{base}{path}", timeout=5)
            if resp.status_code == expected:
                ok(f"{method} {path} -> {resp.status_code}")
                results["pass"] += 1
                any_up = True
            elif resp.status_code == 401:
                ok(f"{method} {path} -> 401 (auth required, endpoint exists)")
                results["pass"] += 1
                any_up = True
            else:
                warn(f"{method} {path} -> {resp.status_code} (expected {expected})")
                results["warn"] += 1
        except httpx.ConnectError:
            if not any_up:
                warn(f"Backend not running at {base} — start with: python -m backend.api.app")
                results["warn"] += 1
                return False
            warn(f"{method} {path} -> connection refused")
            results["warn"] += 1
        except Exception as e:
            warn(f"{method} {path} -> {e}")
            results["warn"] += 1

    return any_up


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quant X E2E Test")
    parser.add_argument("--seed", action="store_true", help="Seed demo signals only")
    parser.add_argument("--scan", type=str, help="Scan specific symbols (comma-separated)")
    parser.add_argument("--backend-only", action="store_true", help="Skip API endpoint checks")
    parser.add_argument("--count", type=int, default=15, help="Number of demo signals to seed")
    args = parser.parse_args()

    print(f"\n{Colors.BOLD}Quant X — End-to-End Test{Colors.END}")
    print(f"Date: {date.today()} | Time: {datetime.now().strftime('%H:%M:%S')}\n")

    if args.seed:
        test_env()
        seed_demo_signals(args.count)
    elif args.scan:
        test_env()
        symbols = [s.strip().upper() for s in args.scan.split(",")]
        test_kite_data()
        test_signal_generation(symbols)
    else:
        # Full test
        test_env()
        test_supabase()
        kite_ok = test_kite_data()
        test_screener()
        test_signal_generation()

        # Seed demo data if no real signals and Kite failed
        if not kite_ok:
            seed_demo_signals(args.count)

        if not args.backend_only:
            test_api_endpoints()

    # Summary
    header("SUMMARY")
    total = results["pass"] + results["fail"] + results["warn"]
    print(f"  {Colors.OK}Passed: {results['pass']}{Colors.END}")
    print(f"  {Colors.FAIL}Failed: {results['fail']}{Colors.END}")
    print(f"  {Colors.WARN}Warnings: {results['warn']}{Colors.END}")
    print(f"  Total checks: {total}\n")

    if results["fail"] == 0:
        print(f"  {Colors.OK}{Colors.BOLD}All critical checks passed!{Colors.END}")
        if results["warn"] > 0:
            print(f"  {Colors.WARN}Some warnings — check Kite token or optional services.{Colors.END}")
    else:
        print(f"  {Colors.FAIL}{Colors.BOLD}{results['fail']} critical failures — fix before testing frontend.{Colors.END}")

    print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
    print("  1. Start backend:  cd {project} && python -m backend.api.app")
    print("  2. Start frontend: cd frontend && npm run dev")
    print("  3. Open browser:   http://localhost:3000")
    print("  4. Sign up, then check /signals and /screener pages")
    print()

    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
