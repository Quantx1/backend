#!/usr/bin/env python3
"""
Unified end-to-end ML training orchestrator.

Runs every engine trainer in canonical dependency order. Meant to be
fired once — after the PR flow is closed — on a GPU box (Modal / RunPod
/ Lambda / similar) with a populated Supabase connection.

Usage:

    # Dry-run all trainers (reports which are ready, trains none):
    python scripts/train/train_all_models.py --dry-run

    # Real run — trains only engines whose ``is_ready()`` returns True;
    # uploads artifacts to B2 as shadow versions; writes a metrics
    # summary JSON to ``reports/train_<timestamp>/``:
    python scripts/train/train_all_models.py

    # Train only specific engines:
    python scripts/train/train_all_models.py --only earnings_scout tickpulse

    # Skip upload (artifacts stay local):
    python scripts/train/train_all_models.py --no-upload

    # Auto-promote shadow → prod for engines that clear the gates:
    python scripts/train/train_all_models.py --promote

Environment:
    * Supabase creds: ``SUPABASE_URL`` + ``SUPABASE_SERVICE_ROLE_KEY``
    * B2 creds (only for --upload): ``B2_KEY_ID`` + ``B2_APPLICATION_KEY`` + ``B2_BUCKET``
    * GPU detection: uses ``torch.cuda.is_available()``; falls back to False.

No fallback / no simulated output: engines that don't have their
prerequisites in place report ``skipped`` with a reason. No synthetic
metrics are written.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List


def _setup_path():
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _supabase():
    try:
        from backend.core.database import get_supabase_admin
        return get_supabase_admin()
    except Exception as exc:
        logging.warning("Supabase unavailable: %s", exc)
        return None


def main() -> int:
    _setup_path()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report which trainers are ready but train none.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Train only the given model names (space-separated).")
    ap.add_argument("--no-upload", action="store_true",
                    help="Skip B2 upload + registry insert.")
    ap.add_argument("--promote", action="store_true",
                    help="Auto-promote shadow → prod if a trainer's metrics "
                         "clear its own gate (EarningsScout: acc ≥ 0.60 "
                         "& n_test ≥ 50; others: disabled by default).")
    ap.add_argument("--reports-root", default="reports",
                    help="Where to write the run summary JSON.")
    ap.add_argument("--models-root", default="artifacts/models",
                    help="Where to write trained artifacts locally.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip any trainer whose primary artifact already "
                         "exists in --models-root (matching by model name). "
                         "Use after a pod crash to pick up where you left "
                         "off without re-running successful trainers.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("train_all_models")

    from backend.ai.training import all_trainers, TrainingContext, TrainReport

    # --- Context -----------------------------------------------------
    repo_root = Path(__file__).resolve().parents[2]
    out_root = repo_root / args.models_root
    out_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_label = f"run-{ts}"

    ctx = TrainingContext(
        out_root=out_root,
        version_label=version_label,
        supabase_client=_supabase(),
        gpu_available=_gpu_available(),
        dry_run=args.dry_run,
    )

    log.info("Unified training run %s", version_label)
    log.info("  models_root = %s", ctx.out_root)
    log.info("  gpu         = %s", ctx.gpu_available)
    log.info("  supabase    = %s", "ok" if ctx.supabase_client else "unavailable")
    log.info("  dry_run     = %s", ctx.dry_run)

    # --- Select trainers --------------------------------------------
    trainers = all_trainers()
    if args.only:
        wanted = set(args.only)
        trainers = [t for t in trainers if t.model_name in wanted]
    if not trainers:
        log.error("No trainers to run (check --only arg)")
        return 2

    log.info("Trainers in run: %s", [t.model_name for t in trainers])

    # --- Resume: skip trainers whose artifact already exists --------
    # Canonical artifact filenames per trainer. Kept in sync with the
    # parallel map in tests/ml/test_artifact_smoke.py (EXPECTED_ARTIFACTS).
    _RESUME_ARTIFACT_BY_MODEL = {
        "regime_hmm":        "regime_hmm.pkl",
        "lgbm_signal_gate":  "lgbm_signal_gate.txt",
        "qlib_alpha158":     "qlib_alpha158.txt",
        "tft_swing":         "tft_model.ckpt",
    }
    skipped_by_resume: List[str] = []
    if args.resume:
        remaining = []
        for t in trainers:
            rel = _RESUME_ARTIFACT_BY_MODEL.get(t.model_name)
            if rel and (out_root / rel).exists() and (out_root / rel).stat().st_size > 100:
                skipped_by_resume.append(t.model_name)
                log.info("Resume: skipping %s — artifact %s already on disk",
                         t.model_name, rel)
            else:
                remaining.append(t)
        trainers = remaining
        if skipped_by_resume:
            log.info("Resume: skipped %d/%d trainers; running %d",
                     len(skipped_by_resume),
                     len(skipped_by_resume) + len(trainers),
                     len(trainers))
        if not trainers:
            log.info("Resume: nothing to do — all artifacts present")
            return 0

    # --- Run each ---------------------------------------------------
    reports: List[TrainReport] = []
    for trainer in trainers:
        log.info("=" * 60)
        log.info(">>> %s", trainer.model_name)
        try:
            report = trainer.run(ctx)
        except Exception as exc:
            log.exception("Trainer crashed: %s", trainer.model_name)
            report = TrainReport(
                model_name=trainer.model_name,
                status="failed",
                reason=f"exception: {exc}",
            )
        log.info("    status=%s", report.status)
        if report.reason:
            log.info("    reason=%s", report.reason)
        if report.metrics:
            log.info("    metrics=%s",
                     {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in report.metrics.items()})
        reports.append(report)

    # --- Upload + promote -------------------------------------------
    if args.no_upload:
        log.info("--no-upload set — skipping registry upload")
    else:
        _upload_all(reports, trainers, args.promote, log)

    # --- Write run summary ------------------------------------------
    reports_dir = repo_root / args.reports_root / f"train_{ts}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "summary.json"
    summary = {
        "version_label": version_label,
        "started_at_utc": ts,
        "gpu_available": ctx.gpu_available,
        "supabase_available": ctx.supabase_client is not None,
        "dry_run": ctx.dry_run,
        "no_upload": args.no_upload,
        "promote": args.promote,
        "resume": args.resume,
        "skipped_by_resume": skipped_by_resume,
        "reports": [asdict(r) for r in reports],
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Summary written: %s", summary_path)

    # --- Exit code reflects failures --------------------------------
    failed = [r for r in reports if r.status == "failed"]
    ok = [r for r in reports if r.status == "ok"]
    log.info("Run complete: %d ok, %d skipped, %d failed",
             len(ok), len(reports) - len(ok) - len(failed), len(failed))
    return 1 if failed else 0


def _upload_all(reports, trainers, promote: bool, log) -> None:
    try:
        from backend.ai.registry.model_registry import get_registry
    except Exception as exc:
        log.error("Registry unavailable — cannot upload: %s", exc)
        return
    reg = get_registry()
    if reg is None:
        log.error("Registry not configured — skipping upload")
        return

    report_by_name = {r.model_name: r for r in reports}
    for trainer in trainers:
        r = report_by_name.get(trainer.model_name)
        if r is None or r.status != "ok":
            continue
        files = [Path(p) for p in trainer.artifacts() if Path(p).exists()]
        if not files:
            log.warning("%s: no artifacts to upload", trainer.model_name)
            continue
        try:
            row = reg.register(
                trainer.model_name, files,
                metrics=r.metrics,
                trained_by=os.getenv("USER", "unified-pipeline"),
                notes=f"unified run {r.version}",
                is_shadow=True,
            )
            log.info("%s v%s registered (shadow)", trainer.model_name, row.get("version"))
            if promote and _passes_promote_gate(trainer.model_name, r.metrics):
                reg.promote(trainer.model_name, row["version"])
                log.info("%s v%s promoted to prod", trainer.model_name, row["version"])
                _invalidate_engine_cache(trainer.model_name)
        except Exception as exc:
            log.exception("%s upload failed: %s", trainer.model_name, exc)


def _passes_promote_gate(model_name: str, metrics: dict) -> bool:
    """Per-engine promote gate. Conservative by default so a weak model
    never auto-ships. Add a row per engine as its gate is validated."""
    if model_name == "earnings_scout":
        return (float(metrics.get("accuracy", 0)) >= 0.60
                and int(metrics.get("n_test", 0)) >= 50)
    # Other engines: no auto-promote. Admin promotes manually from the
    # admin panel after reviewing /admin/system drift + backtest runs.
    return False


def _invalidate_engine_cache(model_name: str) -> None:
    """Drop the in-memory inference cache for the given engine so
    workers pick up the new weights on their next call."""
    try:
        if model_name == "earnings_scout":
            from backend.ai.earnings.training.trainer import invalidate_cache
            invalidate_cache()
        elif model_name == "tickpulse":
            from backend.ai.intraday import invalidate_cache
            invalidate_cache()
        elif model_name == "autopilot":
            from backend.ai.autopilot import invalidate_cache
            invalidate_cache()
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
