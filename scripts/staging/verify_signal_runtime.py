#!/usr/bin/env python3
"""Verify staging signal records follow confluence-only long/equity contract."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

from supabase import create_client


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


REQUIRED_FIELDS = [
    "target_1",
    "target_2",
    "risk_reward",
    "model_agreement",
    "generated_at",
    "strategy_names",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify staging signal contract.")
    parser.add_argument(
        "--date",
        dest="trade_date",
        default=date.today().isoformat(),
        help="Trade date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--status",
        default="active",
        help="Signal status filter (default: active)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum number of signals required for pass",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of records to inspect",
    )
    return parser.parse_args()


def make_client():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_ANON_KEY) are required.")
    return create_client(supabase_url, supabase_key)


def validate_row(row: Dict) -> List[str]:
    violations: List[str] = []
    signal_id = row.get("id", "<unknown>")

    if row.get("direction") != "LONG":
        violations.append(f"{signal_id}: direction is not LONG ({row.get('direction')})")
    if row.get("segment") != "EQUITY":
        violations.append(f"{signal_id}: segment is not EQUITY ({row.get('segment')})")

    reasons = row.get("reasons") or []
    if any("regime" in str(reason).lower() for reason in reasons):
        violations.append(f"{signal_id}: reasons contains regime text ({reasons})")

    for field in REQUIRED_FIELDS:
        value = row.get(field)
        if value is None:
            violations.append(f"{signal_id}: missing required field {field}")
        if field == "strategy_names" and (not isinstance(value, list) or not value):
            violations.append(f"{signal_id}: strategy_names must be a non-empty list")

    return violations


def main() -> int:
    args = parse_args()
    client = make_client()

    result = (
        client.table("signals")
        .select(
            "id,date,status,symbol,direction,segment,reasons,target_1,target_2,risk_reward,"
            "model_agreement,generated_at,strategy_names,confidence"
        )
        .eq("date", args.trade_date)
        .eq("status", args.status)
        .order("confidence", desc=True)
        .limit(args.limit)
        .execute()
    )
    rows = result.data or []

    print(f"Date: {args.trade_date}")
    print(f"Status filter: {args.status}")
    print(f"Rows inspected: {len(rows)}")

    if len(rows) < args.min_count:
        print(f"FAILED: expected at least {args.min_count} signals, found {len(rows)}")
        return 1

    violations: List[str] = []
    for row in rows:
        violations.extend(validate_row(row))

    if violations:
        print("FAILED: contract violations found:")
        for violation in violations:
            print(f"- {violation}")
        return 1

    print("PASSED: all inspected signals are LONG/EQUITY with required fields and no regime text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
