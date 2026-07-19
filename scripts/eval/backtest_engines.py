#!/usr/bin/env python3
"""Walk-forward portfolio backtest CLI for the style engines.

Runs ``ml.eval.walkforward_backtest.walkforward_portfolio_backtest`` for each
requested engine (momentum / swing / positional), prints a comparison table
and writes a JSON report. The "does this predict real alpha, net of costs?"
verdict tool.

Usage:
    python3 scripts/eval/backtest_engines.py --engines momentum,swing \
        --top-n 20 --cost-bps 30 --limit 40 --out artifacts/eval/backtests.json

Per engine it:
  * enables ``with_forecasts=True`` only when the engine's forecast caches
    exist locally (artifacts/forecast_cache/ or $FORECAST_CACHE_DIR) —
    otherwise warns and runs price-only features;
  * loads tuned hyperparameters from artifacts/models/<params_dir>/metrics.json
    (``hpo.best_params``, falling back to ``best_params``) — missing file =>
    trainer defaults with a loud note;
  * skips an engine gracefully when its trainer module doesn't exist yet
    (positional is being built in parallel).
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# LightGBM/PyTorch both vendor libomp on macOS; without this the second
# import aborts the interpreter. Harmless elsewhere. setdefault so an
# explicit env choice always wins.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_ROOT = Path(__file__).resolve().parents[2]  # scripts/eval/ -> repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("backtest_engines")


# Per-engine wiring: trainer module/classes, tuned-params artifact dir and the
# forecast caches the engine needs before with_forecasts is honest to enable.
ENGINES: Dict[str, Dict[str, Any]] = {
    "momentum": {
        "module": "ml.training.trainers.momentum_lambdarank",
        "trainer": "MomentumTrainer",
        "config": "MomentumConfig",
        "params_dir": "momentum_lambdarank_v2",
        "forecast_files": ("momentum_tsfm.parquet", "momentum_kronos.parquet"),
    },
    "swing": {
        "module": "ml.training.trainers.swing_lambdarank",
        "trainer": "SwingTrainer",
        "config": "SwingConfig",
        "params_dir": "swing_lambdarank_v1",
        "forecast_files": ("momentum_tsfm.parquet", "momentum_kronos.parquet",
                           "swing_chronos.parquet"),
    },
    "positional": {
        "module": "ml.training.trainers.positional_lambdarank",
        "trainer": "PositionalTrainer",
        "config": "PositionalConfig",
        "params_dir": "positional_lambdarank_v1",
        "forecast_files": ("momentum_tsfm.parquet", "momentum_kronos.parquet",
                           "swing_chronos.parquet"),
    },
}


def _forecast_cache_dir() -> Path:
    return Path(os.environ.get("FORECAST_CACHE_DIR",
                               str(_ROOT / "artifacts" / "forecast_cache")))


def _tuned_params(params_dir: str) -> Tuple[Dict[str, Any], str]:
    """(params, source_note) from artifacts/models/<dir>/metrics.json."""
    path = _ROOT / "artifacts" / "models" / params_dir / "metrics.json"
    if not path.exists():
        note = (f"tuned params MISSING ({path.relative_to(_ROOT)} not found) — "
                f"falling back to trainer default hyperparameters")
        logger.warning(note)
        return {}, note
    try:
        metrics = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        note = f"tuned params UNREADABLE ({path}): {exc} — using trainer defaults"
        logger.warning(note)
        return {}, note
    params = (metrics.get("hpo") or {}).get("best_params") or metrics.get("best_params") or {}
    if not params:
        note = (f"metrics.json at {path.relative_to(_ROOT)} has no hpo.best_params/"
                f"best_params — using trainer defaults")
        logger.warning(note)
        return {}, note
    return dict(params), f"tuned params from {path.relative_to(_ROOT)}"


def _build_trainer(name: str, wiring: Dict[str, Any],
                   limit: Optional[int]) -> Tuple[Optional[Any], bool, List[str]]:
    """(trainer_or_None, with_forecasts, notes). None => skip this engine."""
    notes: List[str] = []
    try:
        mod = importlib.import_module(wiring["module"])
    except ModuleNotFoundError as exc:
        logger.warning("engine %s skipped: trainer module %s not importable (%s)",
                       name, wiring["module"], exc)
        return None, False, notes
    trainer_cls = getattr(mod, wiring["trainer"], None)
    config_cls = getattr(mod, wiring["config"], None)
    if trainer_cls is None or config_cls is None:
        logger.warning("engine %s skipped: %s must export %s + %s", name,
                       wiring["module"], wiring["trainer"], wiring["config"])
        return None, False, notes

    cache_dir = _forecast_cache_dir()
    missing = [f for f in wiring["forecast_files"] if not (cache_dir / f).exists()]
    with_forecasts = not missing
    if missing:
        note = (f"forecast caches missing in {cache_dir}: {', '.join(missing)} — "
                f"running with_forecasts=False (price-only features)")
        logger.warning("engine %s: %s", name, note)
        notes.append(note)

    cfg_kwargs: Dict[str, Any] = {}
    if any(f.name == "with_forecasts" for f in dataclass_fields(config_cls)):
        cfg_kwargs["with_forecasts"] = with_forecasts
    cfg = config_cls(**cfg_kwargs)

    from ml.training.trainers.momentum_lambdarank import cached_universe  # noqa: PLC0415
    symbols = cached_universe(limit=limit)
    if not symbols:
        logger.warning("engine %s skipped: empty universe (no data/cache CSVs "
                       "and no data/nse_tiers lists)", name)
        return None, with_forecasts, notes
    try:
        trainer = trainer_cls(cfg=cfg, symbols=symbols)
    except TypeError as exc:
        logger.warning("engine %s skipped: trainer signature unexpected "
                       "(%s(cfg=..., symbols=...) failed: %s)", name,
                       wiring["trainer"], exc)
        return None, with_forecasts, notes
    return trainer, with_forecasts, notes


def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:7.1f}%"
    except (TypeError, ValueError):
        return "      —"


def _print_table(results: Dict[str, dict]) -> None:
    cols = ("engine", "hor", "folds", "dates", "grossShp", "netShp", "netCAGR",
            "netMaxDD", "calmar", "excShp", "DSR(net)", "turn/day")
    header = (f"{cols[0]:<12} {cols[1]:>4} {cols[2]:>5} {cols[3]:>6} {cols[4]:>9} "
              f"{cols[5]:>7} {cols[6]:>8} {cols[7]:>8} {cols[8]:>7} {cols[9]:>7} "
              f"{cols[10]:>8} {cols[11]:>8}")
    print("\n" + "=" * len(header))
    print("WALK-FORWARD TOP-N PORTFOLIO BACKTEST (OOS, net of costs)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<12} SKIPPED/FAILED: {r['error']}")
            continue
        ni = r.get("net_iid", {})
        xi = r.get("excess_iid", {})
        print(f"{name:<12} {r['horizon']:>4} {r['n_folds']:>5} {r['n_test_dates']:>6} "
              f"{r['gross']['sharpe']:>9.2f} {r['net']['sharpe']:>7.2f} "
              f"{_fmt_pct(r['net']['cagr']):>8} {_fmt_pct(r['net']['max_drawdown_pct']):>8} "
              f"{r['net']['calmar']:>7.2f} {r['excess']['sharpe']:>7.2f} "
              f"{r['deflated_sharpe_net']:>8.2f} {r['avg_daily_turnover']:>8.2f}")
        print(f"{'  └ iid':<12} {'':>4} {'':>5} {ni.get('n_periods', 0):>6} "
              f"{'':>9} {ni.get('sharpe', 0.0):>7.2f} "
              f"{_fmt_pct(ni.get('cagr')):>8} {_fmt_pct(ni.get('max_drawdown')):>8} "
              f"{'':>7} {xi.get('sharpe', 0.0):>7.2f} "
              f"{r.get('deflated_sharpe_net_iid', 0.0):>8.2f} {'':>8}")
    print("-" * len(header))
    print("grossShp/netShp/excShp = annualized Sharpe of the gross / net-of-cost / "
          "net-minus-benchmark daily series (overlap-smoothed — flattering).")
    print("└ iid rows = NON-OVERLAPPING H-period evaluation: independent samples, "
          "the statistically defensible headline numbers.\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Walk-forward top-N portfolio backtest for the style engines")
    ap.add_argument("--engines", default=",".join(ENGINES),
                    help=f"comma-separated subset of: {', '.join(ENGINES)}")
    ap.add_argument("--top-n", type=int, default=20, help="names held per test date")
    ap.add_argument("--cost-bps", type=float, default=30.0,
                    help="one-side transaction cost in bps")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap universe size (quick sanity runs)")
    ap.add_argument("--out", default=str(_ROOT / "artifacts" / "eval" / "backtests.json"),
                    help="JSON report path")
    ap.add_argument("--dump-preds", action="store_true",
                    help="persist per-name OOS fold predictions to "
                         "artifacts/eval/fold_preds/{engine}_preds.parquet "
                         "(the meta-labeling training set)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    names = [e.strip().lower() for e in args.engines.split(",") if e.strip()]
    unknown = [e for e in names if e not in ENGINES]
    if unknown:
        ap.error(f"unknown engine(s): {unknown} (available: {list(ENGINES)})")

    results: Dict[str, dict] = {}
    for name in names:
        wiring = ENGINES[name]
        logger.info("=== engine %s ===", name)
        trainer, with_forecasts, notes = _build_trainer(name, wiring, args.limit)
        if trainer is None:
            results[name] = {"error": "trainer unavailable (module missing or "
                                      "universe empty) — skipped"}
            continue
        params, params_note = _tuned_params(wiring["params_dir"])
        dump_path = (_ROOT / "artifacts" / "eval" / "fold_preds" / f"{name}_preds.parquet"
                     if args.dump_preds else None)
        t0 = time.time()
        try:
            res = walkforward_portfolio_backtest_entry(
                trainer, top_n=args.top_n, cost_bps_side=args.cost_bps, params=params,
                dump_preds_path=dump_path)
        except Exception as exc:  # noqa: BLE001 — one engine failing must not kill the rest
            logger.exception("engine %s FAILED", name)
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        res["with_forecasts"] = bool(with_forecasts)
        res["notes"] = [*notes, params_note, *res.get("notes", [])]
        res["runtime_seconds"] = round(time.time() - t0, 1)
        if args.limit:
            res["notes"].append(f"universe capped at --limit {args.limit} — "
                                f"quick-run numbers, not the full-universe verdict")
        results[name] = res
        logger.info("engine %s done in %.1fs (net sharpe %.2f)", name,
                    res["runtime_seconds"], res["net"]["sharpe"])

    _print_table(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "top_n": args.top_n,
        "cost_bps_side": args.cost_bps,
        "universe_limit": args.limit,
        "engines": results,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"report written to {out_path}")
    return 0 if any("error" not in r for r in results.values()) else 1


def walkforward_portfolio_backtest_entry(trainer, **kwargs) -> dict:
    """Import indirection so `--engines positional` on a machine without the
    heavy ML deps still prints the skip message before any lightgbm import."""
    from ml.eval.walkforward_backtest import walkforward_portfolio_backtest  # noqa: PLC0415
    return walkforward_portfolio_backtest(trainer, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
