#!/usr/bin/env python3
"""
Swing AI — Load Testing Script
================================
Simulates concurrent API requests and WebSocket connections against
the backend to validate performance under load.

Usage:
    python scripts/ops/load_test.py --url http://localhost:8000 --users 50

Requirements:
    pip install httpx websockets
"""

import argparse
import asyncio
import json
import time
import statistics
from typing import List

import httpx

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore

# ============================================================================
# Configuration
# ============================================================================

ENDPOINTS = [
    ("GET", "/health"),
    ("GET", "/api/health"),
    ("GET", "/ready"),
    ("GET", "/api/ready"),
]

AUTHENTICATED_ENDPOINTS = [
    ("GET", "/api/signals/today"),
    ("GET", "/api/portfolio/summary"),
    ("GET", "/api/watchlist"),
]


# ============================================================================
# HTTP Load Test
# ============================================================================

async def hit_endpoint(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict | None = None,
) -> tuple[int, float]:
    """Make a single request and return (status_code, latency_ms)."""
    start = time.perf_counter()
    try:
        resp = await client.request(method, url, headers=headers or {})
        latency = (time.perf_counter() - start) * 1000
        return resp.status_code, latency
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return 0, latency


async def run_http_load(base_url: str, num_users: int, rounds: int = 3):
    """Run concurrent HTTP requests against public endpoints."""
    print(f"\n{'='*60}")
    print(f"HTTP Load Test — {num_users} concurrent users x {rounds} rounds")
    print(f"Target: {base_url}")
    print(f"{'='*60}")

    results: dict[str, List[tuple[int, float]]] = {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        for method, path in ENDPOINTS:
            url = f"{base_url}{path}"
            endpoint_results: List[tuple[int, float]] = []

            for _ in range(rounds):
                tasks = [
                    hit_endpoint(client, method, url)
                    for _ in range(num_users)
                ]
                batch = await asyncio.gather(*tasks)
                endpoint_results.extend(batch)

            results[f"{method} {path}"] = endpoint_results

    # Print results
    print(f"\n{'Endpoint':<30} {'Reqs':>5} {'OK':>5} {'Err':>5} {'p50':>8} {'p95':>8} {'p99':>8} {'Max':>8}")
    print("-" * 95)

    for endpoint, res in results.items():
        total = len(res)
        ok = sum(1 for s, _ in res if 200 <= s < 500)
        err = total - ok
        latencies = [l for _, l in res]
        latencies.sort()

        p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
        p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0
        mx = max(latencies) if latencies else 0

        print(
            f"{endpoint:<30} {total:>5} {ok:>5} {err:>5} "
            f"{p50:>7.1f}ms {p95:>7.1f}ms {p99:>7.1f}ms {mx:>7.1f}ms"
        )


# ============================================================================
# WebSocket Load Test
# ============================================================================

async def ws_client(url: str, duration: float, client_id: int) -> dict:
    """Connect a single WebSocket client and hold the connection."""
    result = {"id": client_id, "connected": False, "messages_received": 0, "error": None}

    if websockets is None:
        result["error"] = "websockets library not installed"
        return result

    try:
        async with websockets.connect(url, close_timeout=5) as ws:
            result["connected"] = True

            # Send a ping
            await ws.send(json.dumps({"type": "ping"}))

            # Hold connection and count messages
            end_time = time.time() + duration
            while time.time() < end_time:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    result["messages_received"] += 1
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    break

    except Exception as e:
        result["error"] = str(e)[:80]

    return result


async def run_ws_load(base_url: str, num_connections: int, duration: float = 10.0):
    """Simulate concurrent WebSocket connections."""
    print(f"\n{'='*60}")
    print(f"WebSocket Load Test — {num_connections} connections for {duration}s")
    print(f"{'='*60}")

    if websockets is None:
        print("  SKIPPED: 'websockets' package not installed (pip install websockets)")
        return

    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws/live?token=load_test"

    tasks = [
        ws_client(ws_url, duration, i)
        for i in range(num_connections)
    ]

    results = await asyncio.gather(*tasks)

    connected = sum(1 for r in results if r["connected"])
    failed = sum(1 for r in results if not r["connected"])
    total_msgs = sum(r["messages_received"] for r in results)

    print(f"\n  Connected:  {connected}/{num_connections}")
    print(f"  Failed:     {failed}")
    print(f"  Total msgs: {total_msgs}")

    if failed > 0:
        errors = [r["error"] for r in results if r["error"]]
        unique_errors = set(errors[:5])
        print(f"  Errors:     {', '.join(str(e) for e in unique_errors)}")


# ============================================================================
# Summary
# ============================================================================

async def run_all(base_url: str, num_users: int):
    """Run both HTTP and WebSocket load tests."""
    print(f"\nSwing AI Load Test")
    print(f"Target: {base_url}")
    print(f"Simulated users: {num_users}")

    # 1. Quick connectivity check
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(f"{base_url}/health")
            print(f"Server health: {r.status_code} — {r.json().get('status', 'unknown')}")
        except Exception as e:
            print(f"Server unreachable: {e}")
            return

    # 2. HTTP load
    await run_http_load(base_url, num_users)

    # 3. WebSocket load (fewer connections — more resource-intensive)
    ws_count = min(num_users, 50)
    await run_ws_load(base_url, ws_count, duration=10.0)

    print(f"\n{'='*60}")
    print("Load test complete.")
    print(f"{'='*60}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Swing AI Load Test")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--users", type=int, default=50, help="Number of concurrent users")
    args = parser.parse_args()

    asyncio.run(run_all(args.url, args.users))


if __name__ == "__main__":
    main()
