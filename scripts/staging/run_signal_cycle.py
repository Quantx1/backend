#!/usr/bin/env python3
"""Run one intraday/EOD signal cycle in a staging environment."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

from supabase import create_client


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def parse_candidates(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    parsed = [token.strip().upper() for token in raw.split(",") if token.strip()]
    return parsed or None


def parse_signal_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    return date.fromisoformat(raw)


async def run() -> int:
    parser = argparse.ArgumentParser(description="Run staging signal generation cycle.")
    parser.add_argument(
        "--mode",
        choices=["intraday", "eod"],
        default="intraday",
        help="Signal generation mode",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist generated signals to DB (default: dry-run, no save)",
    )
    parser.add_argument(
        "--candidates",
        help="Optional comma-separated symbol list (e.g. RELIANCE,TCS,INFY)",
    )
    parser.add_argument(
        "--signal-date",
        help="Signal date for EOD mode (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of generated signals to print",
    )
    parser.add_argument(
        "--require-signals",
        action="store_true",
        help="Fail if zero signals are generated",
    )
    args = parser.parse_args()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_key:
        print("SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_ANON_KEY) are required.")
        return 2

    from backend.core.config import settings
    from backend.ai.signals import SignalGenerator

    supabase = create_client(supabase_url, supabase_key)
    generator = SignalGenerator(
        supabase_client=supabase,
        modal_endpoint=settings.ML_INFERENCE_URL,
    )

    candidates = parse_candidates(args.candidates)
    signal_date = parse_signal_date(args.signal_date)

    if args.mode == "intraday":
        signals = await generator.generate_intraday_signals(
            save=args.save,
            candidates=candidates,
            signal_date=signal_date,
        )
    else:
        result = await generator.run_eod_scan(signal_date=signal_date)
        signals = result.get("signals", [])

    print(f"Mode: {args.mode}")
    print(f"Save: {args.save}")
    print(f"Signals generated: {len(signals)}")

    for signal in signals[: max(args.preview, 0)]:
        print(
            f"{signal.symbol:12s} | {signal.direction:5s} | {signal.segment:7s} | "
            f"conf={signal.confidence:6.2f} | rr={signal.risk_reward}"
        )

    if args.require_signals and not signals:
        print("No signals generated and --require-signals is set.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
