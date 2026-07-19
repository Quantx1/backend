"""
Run REAL signal scan using production SignalGenerator (all 4 PROD models).

Bypasses the broad universe screener (which tries 2092 NSE symbols and
hits yfinance rate limits) — passes a focused Nifty 50 candidate list
directly to generate_intraday_signals().
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()


NIFTY_50_DEMO = [
    "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK",
    "BHARTIARTL", "SBIN", "ITC", "LT", "KOTAKBANK",
    "BAJFINANCE", "ASIANPAINT", "MARUTI", "WIPRO", "HCLTECH",
    "ULTRACEMCO", "TITAN", "SUNPHARMA", "POWERGRID", "NTPC",
    "ADANIENT", "ONGC", "JSWSTEEL", "GRASIM", "DIVISLAB",
    "TECHM", "DRREDDY", "M&M", "BAJAJFINSV", "TATASTEEL",
    "AXISBANK", "BAJAJ-AUTO", "HINDALCO", "EICHERMOT", "HEROMOTOCO",
    "INDUSINDBK", "CIPLA", "BEL", "COALINDIA", "TATACONSUM",
    "HINDUNILVR", "NESTLEIND", "BPCL", "BRITANNIA", "APOLLOHOSP",
    "TATAMOTORS", "TRENT", "HDFCLIFE", "ADANIPORTS", "SHRIRAMFIN",
]


async def main():
    print("═" * 70)
    print("Quant X — REAL Signal Scan (Production SignalGenerator)")
    print("Models: regime_hmm v21 + qlib_alpha158 v4 + tft_swing v3 + finbert_india v1")
    print(f"Universe: Nifty 50 ({len(NIFTY_50_DEMO)} symbols, focused demo)")
    print("═" * 70)

    from backend.core.database import get_supabase_admin
    sb = get_supabase_admin()

    from backend.ai.signals import SignalGenerator
    sg = SignalGenerator(sb)
    print()
    print("All 4 models loaded. Running model-first signal generation...")
    print("(bypassing broad universe screener — passing Nifty 50 directly)")
    print()

    signals = await sg.generate_intraday_signals(
        save=True,
        candidates=NIFTY_50_DEMO,
    )

    print()
    print("═" * 70)
    print(f"SCAN COMPLETE: {len(signals)} signals emitted from {len(NIFTY_50_DEMO)} candidates")
    print("═" * 70)
    if signals:
        print(f"{'Symbol':<14} {'Dir':<5} {'Conf':>5} {'Entry':>9} {'Stop':>9} {'Target':>9} {'R:R':>5} {'Regime':<10}")
        print("─" * 84)
        for s in signals[:20]:
            sym = (getattr(s, 'symbol', '') or '')[:13]
            d = getattr(s, 'direction', '?') or '?'
            conf = (getattr(s, 'confidence', 0) or 0) * 100
            entry = getattr(s, 'entry_price', 0) or 0
            stop = getattr(s, 'stop_loss', 0) or 0
            tgt = getattr(s, 'target', None) or getattr(s, 'target_1', 0) or 0
            try:
                rr = ((tgt - entry) / (entry - stop)) if entry != stop else 0
            except Exception:
                rr = 0
            reg = getattr(s, 'regime_at_signal', '?') or '?'
            print(
                f"{sym:<14} {d:<5} {conf:>4.0f}% {entry:>9.2f} {stop:>9.2f} "
                f"{tgt:>9.2f} {rr:>5.1f} {reg:<10}"
            )
    else:
        print()
        print("  No signals emitted. The ensemble gates rejected all candidates.")
        print("  Common reasons:")
        print("    - bear regime (HMM v21 detected — gates halve confidence)")
        print("    - TFT 5-day forecast has wide quantile spread (low R:R)")
        print("    - LGBM gate verdict != BUY or below confidence threshold")
        print("    - <N voters concur on BUY")
        print()
        print("  This is BY DESIGN — gates are tight so we don't ship noise.")

asyncio.run(main())
