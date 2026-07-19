#!/usr/bin/env python
"""
Smoke-test every trainer with 10 stocks, GPU when available.

Runs each model end-to-end in 10-stock smoke mode (universe=10, period=2y,
timesteps=50k, epochs=2) so we catch broken trainers BEFORE any expensive
GPU run. Order is cheapest-first; failures don't abort the batch — every
trainer gets a chance so we see the full picture.

Final output: a table with PASS/FAIL/ERROR/SKIPPED for every trainer +
exit 0 / 1 based on whether any HARD failures occurred. Skipped trainers
don't count as failures (rare since earnings_xgb removed; the zero-shot
trainers can self-skip if their HF model can't download).

Usage:
    python scripts/ops/smoke_all.py
    python scripts/ops/smoke_all.py --only regime_hmm,lgbm_signal_gate
    python scripts/ops/smoke_all.py --skip tft_swing,qlib_alpha158

Cost / time (on RTX 4090):
    Each trainer in 10-stock smoke ≈ 2-15 min. Full batch ≈ 1.5-2 hours.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("smoke_all")


# Cheapest-first order.
# earnings_xgb removed 2026-05-11 — F9 deferred.
DEFAULT_ORDER = [
    "regime_hmm",
    "lgbm_signal_gate",
    "qlib_alpha158",
    "tft_swing",
]


def run_one(trainer: str) -> Dict[str, Any]:
    """Subprocess into validate_trainer.py so each run is isolated.

    Isolation matters because some trainers leak global state (HF model
    caches, torch CUDA mem, Qlib provider init). One trainer's mess
    shouldn't corrupt the next.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train" / "validate_trainer.py"),
        trainer,
        "--report", str(REPO_ROOT / f"smoke_{trainer}.json"),
    ]
    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=str(REPO_ROOT),
        capture_output=True, text=True, env=None,
    )
    elapsed = time.time() - t0

    report_path = REPO_ROOT / f"smoke_{trainer}.json"
    report: Dict[str, Any] = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            pass

    verdict_line = ""
    for line in (result.stdout + result.stderr).splitlines()[-12:]:
        if "VERDICT" in line:
            verdict_line = line.strip()
            break

    # exit code: 0=PASS, 1=FAIL/MISSING, other=crash
    if result.returncode == 0:
        if report.get("metrics", {}).get("skipped"):
            status = "SKIPPED"
        else:
            status = "PASS"
    elif result.returncode == 1:
        status = "FAIL"
    else:
        status = f"CRASH({result.returncode})"

    return {
        "trainer": trainer,
        "status": status,
        "elapsed_s": round(elapsed, 1),
        "verdict_line": verdict_line,
        "stderr_tail": "\n".join(result.stderr.splitlines()[-3:]),
        "report_path": str(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated trainer subset")
    parser.add_argument("--skip", help="comma-separated trainers to skip")
    parser.add_argument("--report", type=Path, default=Path("smoke_all_report.json"))
    args = parser.parse_args()

    trainers = list(DEFAULT_ORDER)
    if args.only:
        wanted = {t.strip() for t in args.only.split(",")}
        trainers = [t for t in trainers if t in wanted]
    if args.skip:
        unwanted = {t.strip() for t in args.skip.split(",")}
        trainers = [t for t in trainers if t not in unwanted]

    print("=" * 76)
    print(f"SMOKE BATCH — {len(trainers)} trainers, 10-stock universe, GPU when available")
    print("=" * 76)

    results: List[Dict[str, Any]] = []
    for i, name in enumerate(trainers, 1):
        print(f"\n[{i}/{len(trainers)}] → {name}")
        res = run_one(name)
        results.append(res)
        emoji = {"PASS": "✅", "SKIPPED": "⏭", "FAIL": "❌"}.get(res["status"], "💥")
        print(f"    {emoji} {res['status']:<10} {res['elapsed_s']:>6}s  {res['verdict_line']}")
        if res["status"] not in ("PASS", "SKIPPED"):
            print(f"    stderr: {res['stderr_tail'][:200]}")

    print()
    print("=" * 76)
    print("SMOKE BATCH SUMMARY")
    print("=" * 76)
    by_status: Dict[str, List[str]] = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r["trainer"])
    for status in ["PASS", "SKIPPED", "FAIL"]:
        names = by_status.get(status, [])
        if names:
            mark = {"PASS": "✅", "SKIPPED": "⏭", "FAIL": "❌"}[status]
            print(f"  {mark} {status:<10} ({len(names)})  {', '.join(names)}")
    # Any other status (CRASH etc) are hard failures
    crashes = [r for r in results if r["status"] not in ("PASS", "SKIPPED", "FAIL")]
    if crashes:
        print(f"  💥 CRASHES ({len(crashes)})")
        for c in crashes:
            print(f"      {c['trainer']}: {c['status']}")

    args.report.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull batch report → {args.report}")

    hard_failures = sum(1 for r in results if r["status"] not in ("PASS", "SKIPPED"))
    return 1 if hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
