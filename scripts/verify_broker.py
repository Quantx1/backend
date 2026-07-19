#!/usr/bin/env python3
"""
Broker adapter verification harness — proves a BETA broker against a LIVE account.

Run this locally with your own broker token. It exercises every adapter method
(login → quote → positions → holdings → margin, and optionally a tiny
UNFILLABLE test order) and prints a ✓/✗ report. It NEVER prints your token.

Send the printed report back to the developer — NOT your token — and any ✗ gets
fixed, then the broker is flipped BETA → stable.

Usage (from the repo root):
    PYTHONPATH=. python scripts/verify_broker.py --broker fyers
    PYTHONPATH=. python scripts/verify_broker.py --broker dhan --symbol TCS
    PYTHONPATH=. python scripts/verify_broker.py --broker kotakneo --place-test-order --yes

Secrets are read via a hidden prompt (getpass) so they never touch your shell
history. Read-only by default; --place-test-order is strictly opt-in and places
a deliberately-unfillable LIMIT buy (₹1) that it cancels immediately.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import traceback

# Make `backend...` importable when run from the repo root.
sys.path.insert(0, os.getcwd())

from backend.data.brokers.integration import (  # noqa: E402
    BrokerFactory, Order, OrderType, TransactionType, ProductType,
)

# Which credential fields each broker needs, and whether each is a secret.
BROKER_FIELDS = {
    "zerodha":  [("api_key", False), ("access_token", True)],
    "upstox":   [("api_key", False), ("api_secret", True), ("access_token", True)],
    "angelone": [("api_key", False), ("client_id", False), ("access_token", True)],
    "fyers":    [("app_id", False), ("access_token", True)],
    "dhan":     [("client_id", False), ("access_token", True)],
    "kotakneo": [("client_id", False), ("access_token", True), ("session_token", True)],
    "aliceblue":[("client_id", False), ("access_token", True)],
}


def _mask(v: str) -> str:
    if not v:
        return "<empty>"
    return v[:3] + "…" + v[-2:] if len(v) > 6 else "***"


def _collect_creds(broker: str) -> dict:
    creds = {}
    print(f"\nEnter credentials for {broker} (input hidden for secrets; "
          f"leave blank to skip an optional field):")
    for field, secret in BROKER_FIELDS[broker]:
        env_key = f"{broker.upper()}_{field.upper()}"
        val = os.getenv(env_key)
        if val:
            print(f"  {field}: (from ${env_key})")
        elif secret:
            val = getpass.getpass(f"  {field} (hidden): ").strip()
        else:
            val = input(f"  {field}: ").strip()
        if val:
            creds[field] = val
    return creds


def _row(name: str, ok: bool, detail: str = "") -> None:
    mark = "✓" if ok else "✗"
    print(f"  [{mark}] {name:<24} {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", required=True, choices=sorted(BROKER_FIELDS))
    ap.add_argument("--symbol", default="RELIANCE")
    ap.add_argument("--exchange", default="NSE")
    ap.add_argument("--place-test-order", action="store_true",
                    help="Place an UNFILLABLE ₹1 limit buy of 1 share and cancel it (real order path test).")
    ap.add_argument("--yes", action="store_true", help="Skip the test-order confirmation prompt.")
    args = ap.parse_args()

    broker = args.broker.lower()
    creds = _collect_creds(broker)

    print(f"\n=== Verifying {broker} — creds present: "
          f"{', '.join(f'{k}={_mask(v)}' for k, v in creds.items())} ===\n")

    try:
        adapter = BrokerFactory.create(broker, creds)
    except Exception as e:
        print(f"FATAL: could not construct adapter: {e}")
        return 2

    results = {}

    # 1) login — everything hinges on this
    try:
        ok = bool(adapter.login())
        results["login"] = ok
        _row("login", ok, "" if ok else "→ token invalid/expired OR endpoint/field mismatch")
    except Exception as e:
        results["login"] = False
        _row("login", False, f"raised: {e}")

    if not results.get("login"):
        print("\nLogin failed — fix this first. If your token is fresh and correct, "
              "the login endpoint/fields need adjusting. Send this report back.")
        return 1

    # 2) quote
    try:
        q = adapter.get_quote(args.symbol, args.exchange)
        has = bool(q) and (q.get("last_price") or q.get("ltp") or q.get("lp"))
        results["get_quote"] = bool(has)
        _row("get_quote", bool(has),
             f"keys={list(q.keys()) if isinstance(q, dict) else type(q).__name__}"
             + ("" if has else " → empty: symbol/instrument-master mapping needs wiring"))
    except Exception as e:
        results["get_quote"] = False
        _row("get_quote", False, f"raised: {e}")

    # 3) historical (only some adapters implement it)
    if hasattr(adapter, "get_historical"):
        try:
            h = adapter.get_historical(args.symbol, "1mo", "1d")
            n = len(h) if h else 0
            results["get_historical"] = n > 0
            _row("get_historical", n > 0, f"{n} candles")
        except Exception as e:
            results["get_historical"] = False
            _row("get_historical", False, f"raised: {e}")

    # 4) positions / holdings / margin (read-only)
    for name, fn in (("get_positions", adapter.get_positions),
                     ("get_holdings", adapter.get_holdings),
                     ("get_available_margin", adapter.get_available_margin)):
        try:
            val = fn()
            results[name] = True
            shape = f"{len(val)} rows" if isinstance(val, list) else f"{val}"
            _row(name, True, shape)
        except Exception as e:
            results[name] = False
            _row(name, False, f"raised: {e}")

    # 5) optional test order — real order path, deliberately unfillable
    if args.place_test_order:
        if not args.yes:
            ans = input("\nPlace a REAL (unfillable ₹1 limit) test order on your "
                        "account and cancel it? [y/N] ").strip().lower()
            if ans != "y":
                print("Skipped test order.")
                args.place_test_order = False
        if args.place_test_order:
            try:
                order = Order(
                    symbol=args.symbol, exchange=args.exchange,
                    transaction_type=TransactionType.BUY, quantity=1,
                    product=ProductType.CNC, order_type=OrderType.LIMIT, price=1.0,
                )
                placed = adapter.place_order(order)
                ok = bool(placed.order_id) and str(placed.status) != "OrderStatus.REJECTED"
                results["place_order"] = ok
                _row("place_order", ok, f"id={placed.order_id} status={placed.status} {placed.message}")
                if placed.order_id:
                    cancelled = adapter.cancel_order(placed.order_id)
                    results["cancel_order"] = bool(cancelled)
                    _row("cancel_order", bool(cancelled),
                         "cancelled" if cancelled else "→ CANCEL MANUALLY in your broker app!")
            except Exception as e:
                results["place_order"] = False
                _row("place_order", False, f"raised: {e}")
                print(traceback.format_exc())

    passed = sum(1 for v in results.values() if v)
    print(f"\n=== {broker}: {passed}/{len(results)} checks passed ===")
    print("Copy everything above (no tokens are shown) and send it back.\n")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
