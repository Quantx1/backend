# Canonical 9-Stage ML/DL Training Pipeline Spine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the plan's canonical `run_pipeline(ctx)` template-method spine (`ml/training/pipeline.py` + `specs.py` + `report.py` + `baseline_drift.py`) that runs the 9 stages — data → EDA → quality → label → feature → purged-CV → fit → HPO → evaluation → results — in fixed order with fail-loud gates and a uniform metrics dict, then migrate the momentum trainer onto it so EDA, Optuna optimization, DSR/PBO evaluation, and report artifacts become part of the real momentum run.

**Architecture:** A `PipelineTrainer(Trainer)` base implements `train()` by delegating to `run_pipeline(self, out_dir)`. The spine owns the shared stages (panel load, EDA, quality, purged-CV fold loop, evaluation, report); each engine overrides declarative hooks (`engine_spec`, `load_panel`, `build_features`, `build_labels`, `make_model`, `fit_args`, `search_space`). Stages reuse existing M0 primitives verbatim (`ml/preprocessing/eda.py`, `ml/data/quality_check.py`, `ml/training/purged_cv.py`, `ml/training/optuna_search.py`, `ml/eval/overfitting.py`). The unified `runner.py` is unchanged — it still discovers trainers and applies the promote/serve-smoke/register gate one level above the spine. `momentum_lambdarank` becomes a thin `PipelineTrainer` with hooks instead of a bespoke `train_momentum`.

**Tech Stack:** Python, pandas/numpy, LightGBM (LambdaRank), scipy (DSR), optuna (HPO, optional), matplotlib (report plots, optional/fail-soft), the existing `Trainer`/registry/serve-smoke machinery.

**Spec:** `docs/ML_TRAINING_READINESS_AND_PIPELINE_2026_06_16.md` §C (9-stage spine table) + §D (hook contract). Reference implementation to mirror: `ml/training/trainers/lgbm_signal_gate.py` (already does EDA-gate + quality + folds + DSR/PBO inline — we are EXTRACTING that into the spine).

**Run tests:** `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest <path> -v`
**Gate after each task:** the touched tests pass · the full suite stays green (`python3 -m pytest -q`, currently 892) · `lint-imports` 1 kept/0 broken.

---

## File Structure

- Create `ml/training/specs.py` — declarative per-engine contract: `CVSpec`, `EvalSpec`, `EDASpec`, `EngineSpec`. Pure dataclasses, no logic. (Task 1)
- Create `ml/training/pipeline.py` — `Stage` enum, `PipelineContext`, `PipelineError`, `run_pipeline(trainer, out_dir)` template-method. The 9-stage orchestrator. (Tasks 3–6 fill its stages)
- Modify `ml/training/base.py` — add `PipelineTrainer(Trainer)` with the hook surface + a `train()` that delegates to `run_pipeline`. (Task 2)
- Create `ml/training/report.py` — `write_report(metrics, out_dir, model_name)` → `report.json` + `report.md` + PNG plots (fail-soft if matplotlib absent). (Task 7)
- Create `ml/training/baseline_drift.py` — `write_baseline(feats_df, feature_cols, out_dir)` → `drift_baseline.json` (per-feature train-window mean/std/quantiles) so `ml/eval` drift can fire later. (Task 8)
- Modify `ml/training/trainers/momentum_lambdarank.py` — replace the bespoke `train_momentum`/`MomentumTrainer.train` with a `PipelineTrainer` subclass + hooks + a LightGBM `SearchSpace`. Keep the artifact/metric contract identical (so serving + the existing momentum tests are unaffected). (Task 9)
- Tests: `tests/ml/training/test_specs.py`, `test_pipeline_trainer_base.py`, `test_pipeline_core.py`, `test_pipeline_eda_gate.py`, `test_pipeline_quality.py`, `test_pipeline_hpo.py`, `test_report.py`, `test_baseline_drift.py`, `test_momentum_pipeline.py`.

**Out of scope (follow-on):** migrating the *legacy* trainers (`lgbm_signal_gate`, `qlib_alpha158`, `regime_hmm`, `tft_swing`) onto the spine — they already work; refactoring them is cleanup, tracked separately. This plan builds the spine and proves it on momentum (the engine about to be GPU-trained).

---

### Task 1: `specs.py` — declarative engine contract

**Files:**
- Create: `ml/training/specs.py`
- Test: `tests/ml/training/test_specs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/training/test_specs.py
from ml.training.specs import CVSpec, EvalSpec, EDASpec, EngineSpec


def test_engine_spec_defaults_for_a_ranking_engine():
    spec = EngineSpec(name="momentum_lambdarank")
    assert spec.name == "momentum_lambdarank"
    assert spec.eval.task == "ranking"
    assert spec.eval.primary_metric == "rank_ic_mean"
    assert spec.eval.min_ic == 0.02 and spec.eval.min_icir == 0.5
    assert spec.cv.n_folds == 5 and spec.cv.embargo_days == 20
    assert spec.eda.max_nan_pct == 0.50 and spec.eda.min_abs_ic == 0.005
    assert spec.label_col == "relevance" and spec.fwd_return_col == "fwd_return"
    assert spec.horizon == 20 and spec.hpo_trials == 0


def test_specs_are_overridable():
    spec = EngineSpec(
        name="x", horizon=10, hpo_trials=15,
        cv=CVSpec(n_folds=3, test_days=21),
        eval=EvalSpec(task="classification", primary_metric="f1", min_ic=0.0),
        eda=EDASpec(run_ic_leakage=False, min_class_pct=0.1),
    )
    assert spec.cv.n_folds == 3 and spec.eval.task == "classification"
    assert spec.eda.run_ic_leakage is False and spec.eda.min_class_pct == 0.1
    assert spec.hpo_trials == 15
```

- [ ] **Step 2: Run, confirm FAIL** (`ModuleNotFoundError: ml.training.specs`).

Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/ml/training/test_specs.py -v`

- [ ] **Step 3: Implement** `ml/training/specs.py`:

```python
"""Declarative per-engine contract for the canonical training spine.

An engine = one EngineSpec (these dataclasses) + the PipelineTrainer hooks
(build_features/build_labels/make_model/...). The spine reads the spec to
parameterize the shared stages (EDA thresholds, CV windows, eval gates) so
every engine produces a uniform metrics dict in model_versions.metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class CVSpec:
    """Purged walk-forward CV windows (passed to purged_walk_forward_by_date)."""
    n_folds: int = 5
    test_days: int = 63
    embargo_days: int = 20
    train_days: int = 378


@dataclass
class EvalSpec:
    """What 'good' means + the promote/quality gate thresholds."""
    task: str = "ranking"            # "ranking" | "classification" | "regression"
    primary_metric: str = "rank_ic_mean"
    min_ic: float = 0.02             # ranking: OOS mean rank-IC floor
    min_icir: float = 0.5            # ranking: IC information ratio floor
    min_deflated_sharpe: float = 0.0  # 0 disables the DSR gate (reported regardless)
    max_pbo: float = 1.0             # 1 disables the PBO gate (reported regardless)


@dataclass
class EDASpec:
    """Stage-1 pre-train audit thresholds (ml/preprocessing/eda.py)."""
    max_nan_pct: float = 0.50
    min_abs_ic: float = 0.005
    max_leakage_corr: float = 0.95
    run_ic_leakage: bool = True       # ranking/regression: IC + leakage gates
    check_class_balance: bool = False  # classification only
    min_class_pct: float = 0.05
    expected_classes: Optional[Sequence[Any]] = None
    max_constant_features: int = 5     # Stage-2 audit_feature_matrix fatal cap


@dataclass
class EngineSpec:
    """The full declarative contract for one engine."""
    name: str
    horizon: int = 20
    label_col: str = "relevance"      # the y column build_labels emits
    fwd_return_col: str = "fwd_return"  # the realized-return column for evaluation
    hpo_trials: int = 0               # >0 enables Optuna; momentum sets e.g. 30
    cv: CVSpec = field(default_factory=CVSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)
    eda: EDASpec = field(default_factory=EDASpec)


__all__ = ["CVSpec", "EvalSpec", "EDASpec", "EngineSpec"]
```

- [ ] **Step 4: Run, confirm PASS.**

- [ ] **Step 5: Commit**

```bash
git add ml/training/specs.py tests/ml/training/test_specs.py
git commit -m "feat(pipeline): EngineSpec/CVSpec/EvalSpec/EDASpec declarative contract"
```

---

### Task 2: `PipelineTrainer` base + hook surface

**Files:**
- Modify: `ml/training/base.py`
- Test: `tests/ml/training/test_pipeline_trainer_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/training/test_pipeline_trainer_base.py
import pytest
from ml.training.base import PipelineTrainer, Trainer, TrainResult
from ml.training.specs import EngineSpec


def test_pipeline_trainer_is_a_trainer_and_requires_hooks():
    class Incomplete(PipelineTrainer):
        name = "incomplete"
        def engine_spec(self):
            return EngineSpec(name="incomplete")
    t = Incomplete()
    assert isinstance(t, Trainer)
    with pytest.raises(NotImplementedError):
        t.build_features(None)
    with pytest.raises(NotImplementedError):
        t.build_labels(None)
    with pytest.raises(NotImplementedError):
        t.make_model({})


def test_train_delegates_to_run_pipeline(monkeypatch, tmp_path):
    called = {}
    import ml.training.base as base_mod

    class T(PipelineTrainer):
        name = "t"
        def engine_spec(self):
            return EngineSpec(name="t")

    def fake_run_pipeline(trainer, out_dir):
        called["trainer"] = trainer.name
        called["out_dir"] = out_dir
        return TrainResult(artifacts=[], metrics={"ok": True})

    monkeypatch.setattr(base_mod, "_run_pipeline", fake_run_pipeline, raising=False)
    res = T().train(tmp_path)
    assert called["trainer"] == "t" and res.metrics["ok"] is True
```

- [ ] **Step 2: Run, confirm FAIL** (`ImportError: cannot import name 'PipelineTrainer'`).

- [ ] **Step 3: Implement** — at the top of `ml/training/base.py` add a TYPE_CHECKING import + a module-level test seam, then append the `PipelineTrainer` class.

Add near the existing imports:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ml.training.specs import EngineSpec

_run_pipeline = None  # module-level seam so tests can monkeypatch the spine
```

Append after the `Trainer` class:

```python
class PipelineTrainer(Trainer):
    """A Trainer whose train() delegates to the canonical 9-stage spine.

    Subclasses provide a declarative EngineSpec + the data/model hooks; the
    spine (ml.training.pipeline.run_pipeline) runs EDA, quality, purged-CV,
    HPO, evaluation, and report in fixed order. Default hooks raise — an engine
    MUST implement build_features/build_labels/make_model. fit_args/search_space
    are optional (ranking engines override fit_args to pass LightGBM `group`).
    """

    def engine_spec(self) -> "EngineSpec":
        raise NotImplementedError("engine_spec() must return an EngineSpec")

    def load_panel(self):
        """Return a tidy long OHLCV panel ['date','symbol',ohlcv]."""
        raise NotImplementedError("load_panel() must return an OHLCV panel")

    def build_features(self, panel):
        """panel -> (feats_df['date','symbol',*feature_cols], feature_cols)."""
        raise NotImplementedError

    def build_labels(self, panel):
        """panel -> labels_df['date','symbol', spec.label_col, spec.fwd_return_col]."""
        raise NotImplementedError

    def make_model(self, params: Dict[str, Any]):
        """Return a FRESH estimator built from hyperparams (LGBMRanker etc.)."""
        raise NotImplementedError

    def fit_args(self, df_tr) -> Dict[str, Any]:
        """Extra kwargs for model.fit (e.g. LightGBM ranking `group`)."""
        return {}

    def search_space(self):
        """Optional ml.training.optuna_search.SearchSpace for HPO. None = none."""
        return None

    def train(self, out_dir: Path) -> TrainResult:
        fn = _run_pipeline
        if fn is None:
            from ml.training.pipeline import run_pipeline as fn  # noqa: PLC0415
        return fn(self, out_dir)
```

- [ ] **Step 4: Run, confirm PASS.**

- [ ] **Step 5: Commit**

```bash
git add ml/training/base.py tests/ml/training/test_pipeline_trainer_base.py
git commit -m "feat(pipeline): PipelineTrainer base with declarative hooks; train() delegates to spine"
```

---

### Task 3: `pipeline.py` core — data → features → labels → purged-CV → fit → evaluation (ranking)

This is the spine's backbone. EDA (Task 4), quality (Task 5), HPO (Task 6), report (Task 7) slot in afterward. Build the ranking path end-to-end with a fake trainer on synthetic data.

**Files:**
- Create: `ml/training/pipeline.py`
- Test: `tests/ml/training/test_pipeline_core.py`

- [ ] **Step 1: Write the failing test** (a tiny in-memory ranking engine; no disk, no network)

```python
# tests/ml/training/test_pipeline_core.py
import json
import numpy as np, pandas as pd
from ml.training.base import PipelineTrainer, TrainResult
from ml.training.specs import EngineSpec, CVSpec, EvalSpec, EDASpec


def _toy_panel(n_days=500, syms=("A", "B", "C", "D", "E")):
    days = pd.bdate_range("2021-01-01", periods=n_days)
    rng = np.random.default_rng(0)
    rows = []
    for i, s in enumerate(syms):
        close = 100 + np.cumsum(rng.normal(0.05 * (i + 1), 1.0, n_days))
        for d, c in zip(days, close):
            rows.append({"date": d, "symbol": s, "close": c})
    return pd.DataFrame(rows)


class _ToyRanker(PipelineTrainer):
    name = "toy_ranker"
    skip_promote_gate = True

    def engine_spec(self):
        return EngineSpec(
            name="toy_ranker", horizon=5,
            cv=CVSpec(n_folds=2, test_days=40, embargo_days=5, train_days=200),
            eval=EvalSpec(task="ranking", min_ic=-1.0, min_icir=-1.0),
            eda=EDASpec(min_abs_ic=0.0, run_ic_leakage=False),
        )

    def load_panel(self):
        return _toy_panel()

    def build_features(self, panel):
        df = panel.sort_values(["symbol", "date"]).copy()
        df["mom_5"] = df.groupby("symbol")["close"].transform(lambda s: s / s.shift(5) - 1.0)
        df["xs_rank"] = df.groupby("date")["mom_5"].rank(pct=True)
        cols = ["mom_5", "xs_rank"]
        return df[["date", "symbol", *cols]], cols

    def build_labels(self, panel):
        df = panel.sort_values(["symbol", "date"]).copy()
        df["fwd_return"] = df.groupby("symbol")["close"].transform(lambda s: s.shift(-5) / s - 1.0)
        df["relevance"] = df.groupby("date")["fwd_return"].rank(pct=True).mul(9).round()
        return df[["date", "symbol", "relevance", "fwd_return"]]

    def make_model(self, params):
        import lightgbm as lgb
        base = dict(objective="lambdarank", metric="ndcg", n_estimators=60,
                    num_leaves=15, learning_rate=0.05, verbose=-1, random_state=0)
        base.update(params or {})
        return lgb.LGBMRanker(**base)

    def fit_args(self, df_tr):
        return {"group": df_tr["date"].groupby(df_tr["date"], sort=False).size().to_numpy()}


def test_run_pipeline_ranking_end_to_end(tmp_path):
    from ml.training.pipeline import run_pipeline
    res = run_pipeline(_ToyRanker(), tmp_path)
    assert isinstance(res, TrainResult)
    m = res.metrics
    assert "rank_ic_mean" in m and "rank_ic_per_fold" in m and "n_folds" in m
    assert m["n_features"] == 2 and m["n_folds"] == 2
    names = {p.name for p in res.artifacts}
    assert "feature_order.json" in names and "metrics.json" in names
    fo = json.loads((tmp_path / "feature_order.json").read_text())
    assert fo == ["mom_5", "xs_rank"]
```

- [ ] **Step 2: Run, confirm FAIL** (`ModuleNotFoundError: ml.training.pipeline`).

- [ ] **Step 3: Implement** `ml/training/pipeline.py` (EDA/quality/HPO/report are no-op seams here; Tasks 4–7 fill them):

```python
"""Canonical 9-stage training spine (run_pipeline).

Fixed-order, fail-loud stages with a single namespaced metrics dict so
model_versions.metrics is uniform across engines. Shared stages live here;
per-engine behavior comes from PipelineTrainer hooks. Reuses M0 primitives
verbatim (eda, quality_check, purged_cv, optuna_search, eval/*).
"""
from __future__ import annotations

import enum
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.training.base import TrainResult
from ml.training.purged_cv import PurgedCVConfig, purged_walk_forward_by_date

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when a stage gate fails (EDA blockers, fatal quality, etc.)."""


class Stage(enum.Enum):
    DATA = "data"
    EDA = "eda"
    QUALITY = "quality"
    LABEL = "label"
    FEATURE = "feature"
    CV = "cv"
    FIT = "fit"
    HPO = "hpo"
    EVALUATION = "evaluation"
    REPORT = "report"


@dataclass
class PipelineContext:
    trainer: Any
    spec: Any
    out_dir: Path
    panel: Optional[pd.DataFrame] = None
    feats: Optional[pd.DataFrame] = None
    feature_cols: List[str] = field(default_factory=list)
    df: Optional[pd.DataFrame] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    n_hpo_trials: int = 1
    fold_preds: List[pd.DataFrame] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


def _groups_for(dates: pd.Series) -> np.ndarray:
    return dates.groupby(dates, sort=False).size().to_numpy()


def _rank_ic(frame: pd.DataFrame) -> tuple[float, int]:
    from scipy.stats import spearmanr  # noqa: PLC0415
    ics = []
    for _, g in frame.groupby("date"):
        if len(g) >= 5:
            ic = spearmanr(g["pred"], g["fwd_return"]).statistic
            if not np.isnan(ic):
                ics.append(ic)
    return (float(np.mean(ics)) if ics else float("nan")), len(ics)


def _decile_spread_frame(frame: pd.DataFrame) -> list:
    per_date = []
    for _, g in frame.groupby("date"):
        if len(g) >= 20:
            g = g.sort_values("pred")
            k = max(1, len(g) // 10)
            per_date.append(g["fwd_return"].iloc[-k:].mean() - g["fwd_return"].iloc[:k].mean())
    return per_date


# --- stage seams filled by later tasks (no-ops now) -----------------------
def _stage_eda(ctx: "PipelineContext") -> None:          # Task 4
    return None


def _stage_quality(ctx: "PipelineContext") -> None:      # Task 5
    return None


def _stage_hpo(ctx: "PipelineContext", n_trials: Optional[int] = None) -> None:  # Task 6
    ctx.best_params = {}
    ctx.n_hpo_trials = 1


def _stage_report(ctx: "PipelineContext") -> List[Path]:  # Task 7
    return []
# --------------------------------------------------------------------------


def _build_dataset(ctx: PipelineContext) -> None:
    """Stages DATA + FEATURE + LABEL: panel -> merged df, warmup-dropped."""
    t = ctx.trainer
    ctx.panel = t.load_panel()
    if ctx.panel is None or ctx.panel.empty:
        raise PipelineError(f"[{ctx.spec.name}] load_panel returned no data")
    ctx.feats, ctx.feature_cols = t.build_features(ctx.panel)
    labels = t.build_labels(ctx.panel)
    df = ctx.feats.merge(labels, on=["date", "symbol"], how="inner")
    df = df.dropna(subset=ctx.feature_cols + [ctx.spec.label_col, ctx.spec.fwd_return_col])
    ctx.df = df.sort_values("date").reset_index(drop=True)
    if ctx.df.empty:
        raise PipelineError(f"[{ctx.spec.name}] dataset empty after warmup dropna")


def _cv_and_fit(ctx: PipelineContext) -> None:
    """Stages CV + FIT + per-fold scoring (collect OOS preds)."""
    t, spec, df = ctx.trainer, ctx.spec, ctx.df
    cv = PurgedCVConfig(
        n_folds=spec.cv.n_folds, test_days=spec.cv.test_days,
        embargo_days=spec.cv.embargo_days, train_days=spec.cv.train_days,
    )
    folds = list(purged_walk_forward_by_date(df["date"], cv))
    if not folds:
        raise PipelineError(f"[{spec.name}] purged CV produced 0 folds (history too short)")
    for tr_idx, te_idx in folds:
        tr = df.iloc[tr_idx].sort_values("date")
        te = df.iloc[te_idx].sort_values("date")
        model = t.make_model(ctx.best_params)
        model.fit(tr[ctx.feature_cols], tr[spec.label_col], **t.fit_args(tr))
        preds = model.predict(te[ctx.feature_cols])
        ctx.fold_preds.append(pd.DataFrame({
            "date": te["date"].to_numpy(), "symbol": te["symbol"].to_numpy(),
            "pred": np.asarray(preds, dtype=float),
            "fwd_return": te[spec.fwd_return_col].to_numpy(),
        }))
    ctx.metrics["n_folds"] = len(folds)


def _stage_evaluation(ctx: PipelineContext) -> None:
    """Stage EVALUATION: rank-IC/ICIR/decile spread + DSR/PBO (uniform metrics)."""
    fold_ic, fold_spread_means, n_dates, fold_returns = [], [], [], []
    for fp in ctx.fold_preds:
        ic, n = _rank_ic(fp)
        fold_ic.append(ic); n_dates.append(n)
        per_date = _decile_spread_frame(fp)
        fold_returns.append(per_date)
        fold_spread_means.append(float(np.mean(per_date)) if per_date else float("nan"))

    def _mean(xs):
        v = [x for x in xs if x == x]
        return float(np.mean(v)) if v else float("nan")

    ic_mean = _mean(fold_ic)
    ic_std = float(np.std([x for x in fold_ic if x == x])) if any(x == x for x in fold_ic) else float("nan")
    icir = ic_mean / (ic_std + 1e-9) if ic_std == ic_std else float("nan")

    try:
        from ml.eval.overfitting import dsr_pbo_from_fold_returns  # noqa: PLC0415
        dsr_pbo = dsr_pbo_from_fold_returns(
            fold_returns=fold_returns,
            n_trials=max(ctx.n_hpo_trials * max(len(ctx.fold_preds), 1), 1),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("DSR/PBO failed: %s", exc)
        dsr_pbo = {"deflated_sharpe": 0.0, "probability_backtest_overfitting": 0.5}

    ctx.metrics.update({
        "model": ctx.spec.name,
        "primary_metric": ctx.spec.eval.primary_metric,
        "rank_ic_mean": round(ic_mean, 4),
        "rank_ic_std": round(ic_std, 4),
        "rank_icir": round(icir, 4),
        "decile_spread_mean": round(_mean(fold_spread_means), 4),
        "rank_ic_per_fold": [round(x, 4) for x in fold_ic],
        "decile_spread_per_fold": [round(x, 4) for x in fold_spread_means],
        "deflated_sharpe": dsr_pbo.get("deflated_sharpe"),
        "probability_backtest_overfitting": dsr_pbo.get("probability_backtest_overfitting"),
        "primary_value": round(ic_mean, 4),
        "n_test_dates": int(sum(n_dates)),
    })
    ev = ctx.spec.eval
    passed = (
        ic_mean == ic_mean and ic_mean >= ev.min_ic
        and icir == icir and icir >= ev.min_icir
        and (ev.min_deflated_sharpe <= 0 or (dsr_pbo.get("deflated_sharpe") or 0) >= ev.min_deflated_sharpe)
        and (ev.max_pbo >= 1 or (dsr_pbo.get("probability_backtest_overfitting") or 1) <= ev.max_pbo)
    )
    ctx.metrics[f"{ctx.spec.name}_quality_pass"] = bool(passed)
    if not passed:
        ctx.metrics[f"{ctx.spec.name}_quality_reason"] = (
            f"ic={ic_mean} icir={icir} dsr={dsr_pbo.get('deflated_sharpe')} "
            f"pbo={dsr_pbo.get('probability_backtest_overfitting')} below gate"
        )


def _final_fit_and_save(ctx: PipelineContext) -> List[Path]:
    """Fit on ALL usable data; write booster + feature_order + drift baseline."""
    t, spec, df = ctx.trainer, ctx.spec, ctx.df
    model = t.make_model(ctx.best_params)
    model.fit(df[ctx.feature_cols], df[spec.label_col], **t.fit_args(df))
    booster = getattr(model, "booster_", model)
    model_path = ctx.out_dir / f"{spec.name}.txt"
    booster.save_model(str(model_path))
    try:
        imp = booster.feature_importance(importance_type="gain")
        ctx.feature_importance = {c: float(v) for c, v in zip(ctx.feature_cols, imp)}
    except Exception:  # noqa: BLE001
        ctx.feature_importance = {}
    fo_path = ctx.out_dir / "feature_order.json"
    fo_path.write_text(json.dumps(list(ctx.feature_cols), indent=2))
    ctx.metrics.update({
        "n_features": len(ctx.feature_cols),
        "n_rows": int(len(df)),
        "n_symbols": int(df["symbol"].nunique()),
        "n_dates": int(df["date"].nunique()),
        "horizon": spec.horizon,
        "best_params": ctx.best_params,
        "feature_importance": ctx.feature_importance,
    })
    paths = [model_path, fo_path]
    try:
        from ml.training.baseline_drift import write_baseline  # noqa: PLC0415
        paths.append(write_baseline(df, list(ctx.feature_cols), ctx.out_dir))
    except Exception as exc:  # noqa: BLE001 — non-fatal
        logger.warning("[%s] drift baseline failed: %s", spec.name, exc)
    return paths


def run_pipeline(trainer: Any, out_dir: Path) -> TrainResult:
    """Execute the 9 stages in fixed order and write artifacts to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = trainer.engine_spec()
    ctx = PipelineContext(trainer=trainer, spec=spec, out_dir=out_dir)
    t0 = time.time()

    _build_dataset(ctx)            # DATA + FEATURE + LABEL
    _stage_eda(ctx)                # EDA gate (Task 4)
    _stage_quality(ctx)            # QUALITY gate (Task 5)
    _stage_hpo(ctx, n_trials=spec.hpo_trials or None)  # HPO (Task 6)
    _cv_and_fit(ctx)               # CV + FIT + OOS preds
    _stage_evaluation(ctx)         # EVALUATION
    artifacts = [*_final_fit_and_save(ctx)]

    ctx.metrics["train_seconds"] = round(time.time() - t0, 1)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(ctx.metrics, indent=2, default=str))
    artifacts.append(metrics_path)
    artifacts.extend(_stage_report(ctx))   # REPORT (Task 7)

    logger.info("[%s] pipeline done: %s feats, %s folds, ic=%s",
                spec.name, ctx.metrics.get("n_features"),
                ctx.metrics.get("n_folds"), ctx.metrics.get("rank_ic_mean"))
    return TrainResult(artifacts=artifacts, metrics=ctx.metrics)


__all__ = ["Stage", "PipelineContext", "PipelineError", "run_pipeline"]
```

- [ ] **Step 4: Run, confirm PASS.**

Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/ml/training/test_pipeline_core.py -v`

- [ ] **Step 5: Commit**

```bash
git add ml/training/pipeline.py tests/ml/training/test_pipeline_core.py
git commit -m "feat(pipeline): run_pipeline core — data/feature/label/purged-CV/fit/evaluation (ranking) with uniform metrics"
```

> Note: Task 8 creates `ml/training/baseline_drift.py`; until then the lazy import in `_final_fit_and_save` is caught (non-fatal) so this task's test still passes without it.

---

### Task 4: EDA stage (fail-loud Stage 1)

**Files:**
- Modify: `ml/training/pipeline.py` (`_stage_eda`)
- Test: `tests/ml/training/test_pipeline_eda_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/training/test_pipeline_eda_gate.py
import numpy as np, pandas as pd, pytest
from ml.training.pipeline import _stage_eda, PipelineContext, PipelineError
from ml.training.specs import EngineSpec, EDASpec


def _ctx(df, feature_cols, eda):
    spec = EngineSpec(name="t", eda=eda)
    c = PipelineContext(trainer=None, spec=spec, out_dir=None)
    c.df = df; c.feature_cols = feature_cols
    return c


def test_eda_blocks_on_high_nan_feature():
    df = pd.DataFrame({
        "date": pd.bdate_range("2022-01-01", periods=100),
        "symbol": ["A"] * 100,
        "good": np.linspace(0, 1, 100),
        "mostly_nan": [np.nan] * 80 + list(np.linspace(0, 1, 20)),
        "relevance": np.arange(100) % 10, "fwd_return": np.linspace(-0.1, 0.1, 100),
    })
    ctx = _ctx(df, ["good", "mostly_nan"], EDASpec(max_nan_pct=0.50, run_ic_leakage=False))
    with pytest.raises(PipelineError) as e:
        _stage_eda(ctx)
    assert "high_nan" in str(e.value)


def test_eda_passes_clean_data_and_records_summary():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "date": pd.bdate_range("2022-01-01", periods=200),
        "symbol": ["A"] * 200,
        "f1": rng.normal(0, 1, 200), "f2": rng.normal(0, 1, 200),
        "relevance": rng.integers(0, 10, 200), "fwd_return": rng.normal(0, 0.02, 200),
    })
    ctx = _ctx(df, ["f1", "f2"], EDASpec(min_abs_ic=0.0, run_ic_leakage=False))
    _stage_eda(ctx)
    assert "eda" in ctx.metrics and ctx.metrics["eda"]["n_features"] == 2
```

- [ ] **Step 2: Run, confirm FAIL** (`_stage_eda` is a no-op → no raise, no `ctx.metrics["eda"]`).

- [ ] **Step 3: Implement** — replace the `_stage_eda` seam in `ml/training/pipeline.py`:

```python
def _stage_eda(ctx: PipelineContext) -> None:
    """Stage EDA — fail-loud pre-train audit (ml/preprocessing/eda.py)."""
    from ml.preprocessing.eda import (  # noqa: PLC0415
        EDAReport, eda_classification_balance, eda_dataframe_summary,
        eda_feature_label_ic, eda_leakage_check, eda_near_constant_features,
    )
    spec, df, cols = ctx.spec, ctx.df, ctx.feature_cols
    rep = EDAReport(trainer=spec.name, n_rows=len(df), n_features=len(cols),
                    n_symbols=int(df["symbol"].nunique()))
    fs = eda_dataframe_summary(df, cols, max_nan_pct=spec.eda.max_nan_pct)
    rep.feature_summary = fs.get("per_feature", {})
    rep.blockers.extend(fs.get("blockers", []))
    rep.near_constant = eda_near_constant_features(df, cols)
    if spec.eda.check_class_balance:
        bal = eda_classification_balance(
            df[spec.label_col], min_class_pct=spec.eda.min_class_pct,
            expected_classes=spec.eda.expected_classes)
        rep.label_summary = bal
        rep.blockers.extend(bal.get("blockers", []))
    if spec.eda.run_ic_leakage:
        eda_df = df[cols].copy()
        eda_df["_label"] = df[spec.fwd_return_col].to_numpy()
        ic = eda_feature_label_ic(eda_df, cols, "_label", min_abs_mean_ic=spec.eda.min_abs_ic)
        rep.ic_summary = ic
        rep.blockers.extend(ic.get("blockers", []))
        leak = eda_leakage_check(eda_df, cols, "_label", max_corr=spec.eda.max_leakage_corr)
        rep.leakage_summary = leak
        rep.blockers.extend(leak.get("blockers", []))
    ctx.metrics["eda"] = rep.to_dict()
    if not rep.ok:
        raise PipelineError(f"[{spec.name}] EDA gate FAILED: {rep.blockers}")
```

> The IC/leakage check correlates each feature against `fwd_return` (the realized forward return), NOT the integer `relevance` bucket — that's the genuine look-ahead test.

- [ ] **Step 4: Run, confirm PASS** + re-run Task 3's core test.

Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/ml/training/test_pipeline_eda_gate.py tests/ml/training/test_pipeline_core.py -v`

- [ ] **Step 5: Commit**

```bash
git add ml/training/pipeline.py tests/ml/training/test_pipeline_eda_gate.py
git commit -m "feat(pipeline): Stage 1 EDA gate (NaN/near-constant/IC/leakage), fail-loud"
```

---

### Task 5: Quality stage (Stage 2 — dead/constant feature audit)

**Files:**
- Modify: `ml/training/pipeline.py` (`_stage_quality`)
- Test: `tests/ml/training/test_pipeline_quality.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/training/test_pipeline_quality.py
import numpy as np, pandas as pd, pytest
from ml.training.pipeline import _stage_quality, PipelineContext, PipelineError
from ml.training.specs import EngineSpec, EDASpec


def _ctx(df, cols, max_constant=5):
    spec = EngineSpec(name="t", eda=EDASpec(max_constant_features=max_constant))
    c = PipelineContext(trainer=None, spec=spec, out_dir=None)
    c.df = df; c.feature_cols = cols
    return c


def test_quality_blocks_on_too_many_dead_features():
    n = 200
    df = pd.DataFrame({"date": pd.bdate_range("2022-01-01", periods=n), "symbol": ["A"] * n})
    cols = []
    for i in range(6):
        df[f"dead{i}"] = 1.0; cols.append(f"dead{i}")
    df["live"] = np.linspace(0, 1, n); cols.append("live")
    ctx = _ctx(df, cols, max_constant=5)
    with pytest.raises(PipelineError) as e:
        _stage_quality(ctx)
    assert "constant" in str(e.value).lower()
    assert ctx.metrics["feature_audit"]["n_constant"] == 6


def test_quality_passes_live_features():
    rng = np.random.default_rng(0); n = 200
    df = pd.DataFrame({"date": pd.bdate_range("2022-01-01", periods=n), "symbol": ["A"] * n,
                       "a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    ctx = _ctx(df, ["a", "b"])
    _stage_quality(ctx)
    assert ctx.metrics["feature_audit"]["n_constant"] == 0
```

- [ ] **Step 2: Run, confirm FAIL** (no-op `_stage_quality`).

- [ ] **Step 3: Implement** — replace the `_stage_quality` seam:

```python
def _stage_quality(ctx: PipelineContext) -> None:
    """Stage QUALITY — dead/constant feature audit (catches un-ingested cols)."""
    from ml.data.quality_check import audit_feature_matrix  # noqa: PLC0415
    audit = audit_feature_matrix(
        ctx.df[ctx.feature_cols], feature_names=list(ctx.feature_cols),
        fatal_max_constant=ctx.spec.eda.max_constant_features,
    )
    ctx.metrics["feature_audit"] = audit
    if audit.get("fatal"):
        raise PipelineError(
            f"[{ctx.spec.name}] feature quality FATAL: {audit['n_constant']} dead "
            f"features {audit['constant_features']}"
        )
```

- [ ] **Step 4: Run, confirm PASS** + re-run Task 3 core test.

- [ ] **Step 5: Commit**

```bash
git add ml/training/pipeline.py tests/ml/training/test_pipeline_quality.py
git commit -m "feat(pipeline): Stage 2 quality audit (dead/constant feature gate)"
```

---

### Task 6: HPO stage (Stage 7 — Optuna over OOS rank-IC, opt-in)

**Files:**
- Modify: `ml/training/pipeline.py` (`_stage_hpo` + a fold-objective helper)
- Test: `tests/ml/training/test_pipeline_hpo.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/training/test_pipeline_hpo.py
import numpy as np, pandas as pd
from ml.training.pipeline import _stage_hpo, PipelineContext
from ml.training.specs import EngineSpec
from ml.training.optuna_search import SearchSpace


class _Eng:
    name = "t"
    def make_model(self, params):
        import lightgbm as lgb
        base = dict(objective="lambdarank", n_estimators=40, verbose=-1, random_state=0)
        base.update(params or {}); return lgb.LGBMRanker(**base)
    def fit_args(self, df_tr):
        return {"group": df_tr["date"].groupby(df_tr["date"], sort=False).size().to_numpy()}
    def search_space(self):
        return SearchSpace(suggest=lambda tr: {"num_leaves": tr.suggest_int("num_leaves", 7, 31)})


def _toy_df(n=400):
    days = pd.bdate_range("2021-01-01", periods=n); rng = np.random.default_rng(0); rows = []
    for i, s in enumerate("ABCDE"):
        close = 100 + np.cumsum(rng.normal(0.04 * (i + 1), 1, n))
        for d, c in zip(days, close):
            rows.append({"date": d, "symbol": s, "close": c})
    df = pd.DataFrame(rows).sort_values(["symbol", "date"])
    df["f"] = df.groupby("symbol")["close"].transform(lambda s: s / s.shift(5) - 1)
    df["fwd_return"] = df.groupby("symbol")["close"].transform(lambda s: s.shift(-5) / s - 1)
    df["relevance"] = df.groupby("date")["fwd_return"].rank(pct=True).mul(9).round()
    return df.dropna().sort_values("date").reset_index(drop=True)


def test_hpo_sets_best_params_and_trial_count():
    eng = _Eng()
    spec = EngineSpec(name="t", horizon=5, hpo_trials=4)
    ctx = PipelineContext(trainer=eng, spec=spec, out_dir=None)
    ctx.df = _toy_df(); ctx.feature_cols = ["f"]
    _stage_hpo(ctx, n_trials=4)
    assert "num_leaves" in ctx.best_params
    assert ctx.n_hpo_trials >= 1
    assert ctx.metrics["hpo"]["optimized"] in (True, False)


def test_hpo_skipped_without_search_space():
    class NoSpace(_Eng):
        def search_space(self): return None
    spec = EngineSpec(name="t", hpo_trials=4)
    ctx = PipelineContext(trainer=NoSpace(), spec=spec, out_dir=None)
    ctx.df = _toy_df(); ctx.feature_cols = ["f"]
    _stage_hpo(ctx, n_trials=4)
    assert ctx.best_params == {} and ctx.metrics["hpo"]["optimized"] is False
```

- [ ] **Step 2: Run, confirm FAIL** (the no-op `_stage_hpo` sets empty params + no `ctx.metrics["hpo"]`).

- [ ] **Step 3: Implement** — replace the `_stage_hpo` seam in `ml/training/pipeline.py` and add the objective helper above it:

```python
def _oos_rank_ic_for_params(ctx: PipelineContext, params: Dict[str, Any]) -> float:
    """Train each purged fold with `params`, return mean OOS rank-IC. The HPO
    objective — identical metric to the evaluation stage so we tune what we ship."""
    t, spec, df = ctx.trainer, ctx.spec, ctx.df
    cv = PurgedCVConfig(n_folds=spec.cv.n_folds, test_days=spec.cv.test_days,
                        embargo_days=spec.cv.embargo_days, train_days=spec.cv.train_days)
    ics = []
    for tr_idx, te_idx in purged_walk_forward_by_date(df["date"], cv):
        tr = df.iloc[tr_idx].sort_values("date")
        te = df.iloc[te_idx].sort_values("date")
        model = t.make_model(params)
        model.fit(tr[ctx.feature_cols], tr[spec.label_col], **t.fit_args(tr))
        fp = pd.DataFrame({"date": te["date"].to_numpy(),
                           "pred": np.asarray(model.predict(te[ctx.feature_cols]), float),
                           "fwd_return": te[spec.fwd_return_col].to_numpy()})
        ic, _ = _rank_ic(fp)
        if ic == ic:
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("-inf")


def _stage_hpo(ctx: PipelineContext, n_trials: Optional[int] = None) -> None:
    """Stage HPO — optional Optuna TPE over OOS rank-IC. No space/budget -> skip."""
    budget = n_trials if n_trials is not None else getattr(ctx.spec, "hpo_trials", 0)
    space = ctx.trainer.search_space() if hasattr(ctx.trainer, "search_space") else None
    if space is None or not budget:
        ctx.best_params = {}; ctx.n_hpo_trials = 1
        ctx.metrics["hpo"] = {"optimized": False, "n_trials_run": 0,
                              "reason": "no search_space" if space is None else "hpo_trials=0"}
        return
    from ml.training.optuna_search import OptunaConfig, run_optuna_search  # noqa: PLC0415
    cfg = OptunaConfig(n_trials=int(budget), direction="maximize", n_jobs=1)
    result = run_optuna_search(
        objective=lambda params: _oos_rank_ic_for_params(ctx, params),
        space=space, cfg=cfg,
    )
    ctx.best_params = result.get("best_params", {}) or {}
    ctx.n_hpo_trials = int(result.get("n_trials_run", 1))
    ctx.metrics["hpo"] = {
        "optimized": result.get("optimized"),
        "n_trials_run": ctx.n_hpo_trials,
        "best_value": result.get("best_value"),
        "best_params": ctx.best_params,
    }
```

- [ ] **Step 4: Run, confirm PASS** + re-run Task 3 core test (toy ranker has `hpo_trials=0` default → HPO skipped, params empty).

Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/ml/training/test_pipeline_hpo.py tests/ml/training/test_pipeline_core.py -v`

- [ ] **Step 5: Commit**

```bash
git add ml/training/pipeline.py tests/ml/training/test_pipeline_hpo.py
git commit -m "feat(pipeline): Stage 7 Optuna HPO over OOS rank-IC (opt-in via EngineSpec.hpo_trials)"
```

---

### Task 7: `report.py` — Stage 9 results (report.json + report.md + PNGs)

**Files:**
- Create: `ml/training/report.py`
- Modify: `ml/training/pipeline.py` (`_stage_report`)
- Test: `tests/ml/training/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/training/test_report.py
import json
from ml.training.report import write_report


def test_write_report_emits_json_and_md(tmp_path):
    metrics = {
        "model": "toy", "rank_ic_mean": 0.08, "rank_icir": 2.0,
        "decile_spread_mean": 0.03, "rank_ic_per_fold": [0.07, 0.09],
        "decile_spread_per_fold": [0.02, 0.04], "n_folds": 2, "n_features": 3,
        "deflated_sharpe": 0.6, "probability_backtest_overfitting": 0.3,
        "feature_importance": {"a": 10.0, "b": 5.0, "c": 0.0},
        "toy_quality_pass": True,
    }
    paths = write_report(metrics, tmp_path, model_name="toy")
    names = {p.name for p in paths}
    assert "report.json" in names and "report.md" in names
    rj = json.loads((tmp_path / "report.json").read_text())
    assert rj["rank_ic_mean"] == 0.08
    md = (tmp_path / "report.md").read_text()
    assert "toy" in md and "rank_ic_mean" in md and "shippable" in md.lower()


def test_write_report_survives_without_matplotlib(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__
    def no_mpl(name, *a, **k):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_mpl)
    paths = write_report({"model": "toy", "rank_ic_mean": 0.0, "feature_importance": {}},
                         tmp_path, model_name="toy")
    assert (tmp_path / "report.json").exists()
```

- [ ] **Step 2: Run, confirm FAIL** (`ModuleNotFoundError: ml.training.report`).

- [ ] **Step 3: Implement** `ml/training/report.py`:

```python
"""Stage-9 training report: report.json + report.md + PNG plots.

report.json = the full metrics dict (verbatim, JSONB-safe). report.md = a
human one-pager (headline metrics + per-fold + a 'shippable?' verdict). PNGs
(rank-IC by fold, top feature importances) are best-effort — if matplotlib is
unavailable the report still writes, just without plots.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _verdict(m: Dict[str, Any], model_name: str) -> str:
    q = m.get(f"{model_name}_quality_pass")
    if q is True:
        return "SHIPPABLE — passed the quality gate."
    if q is False:
        return f"NOT shippable — {m.get(f'{model_name}_quality_reason', 'quality gate failed')}"
    return "UNKNOWN — no quality gate recorded."


def _plots(m: Dict[str, Any], out_dir: Path) -> List[Path]:
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.info("matplotlib unavailable — skipping report PNGs: %s", exc)
        return []
    paths: List[Path] = []
    ic = m.get("rank_ic_per_fold") or []
    if ic:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(range(len(ic)), ic); ax.axhline(0, color="k", lw=0.5)
        ax.set_title("rank-IC by fold"); ax.set_xlabel("fold"); ax.set_ylabel("rank-IC")
        p = out_dir / "rank_ic_by_fold.png"; fig.tight_layout(); fig.savefig(p, dpi=90); plt.close(fig)
        paths.append(p)
    fi = m.get("feature_importance") or {}
    if fi:
        top = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)[:20]
        fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(top))))
        ax.barh([k for k, _ in reversed(top)], [v for _, v in reversed(top)])
        ax.set_title("top feature importance (gain)")
        p = out_dir / "feature_importance.png"; fig.tight_layout(); fig.savefig(p, dpi=90); plt.close(fig)
        paths.append(p)
    return paths


def write_report(metrics: Dict[str, Any], out_dir: Path, *, model_name: str) -> List[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    rj = out_dir / "report.json"
    rj.write_text(json.dumps(metrics, indent=2, default=str)); paths.append(rj)

    lines = [
        f"# Training report — {model_name}", "",
        f"**Verdict:** {_verdict(metrics, model_name)}", "",
        "## Headline metrics", "",
        "| metric | value |", "| --- | --- |",
    ]
    for k in ("rank_ic_mean", "rank_icir", "decile_spread_mean", "deflated_sharpe",
              "probability_backtest_overfitting", "n_features", "n_folds", "n_rows",
              "n_symbols", "n_dates", "train_seconds"):
        if k in metrics:
            lines.append(f"| {k} | {metrics[k]} |")
    if metrics.get("rank_ic_per_fold"):
        lines += ["", "## Per-fold rank-IC", "", f"`{metrics['rank_ic_per_fold']}`"]
    fi = metrics.get("feature_importance") or {}
    if fi:
        top = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)[:15]
        lines += ["", "## Top features (gain)", "", "| feature | gain |", "| --- | --- |"]
        lines += [f"| {k} | {round(float(v), 1)} |" for k, v in top]
    rm = out_dir / "report.md"; rm.write_text("\n".join(lines) + "\n"); paths.append(rm)

    paths.extend(_plots(metrics, out_dir))
    return paths
```

- [ ] **Step 4: Wire into the spine** — replace the `_stage_report` seam in `ml/training/pipeline.py`:

```python
def _stage_report(ctx: PipelineContext) -> List[Path]:
    from ml.training.report import write_report  # noqa: PLC0415
    try:
        return write_report(ctx.metrics, ctx.out_dir, model_name=ctx.spec.name)
    except Exception as exc:  # noqa: BLE001 — report is non-fatal
        logger.warning("[%s] report stage failed (non-fatal): %s", ctx.spec.name, exc)
        return []
```

- [ ] **Step 5: Run, confirm PASS** + re-run the core test.

Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/ml/training/test_report.py tests/ml/training/test_pipeline_core.py -v`

- [ ] **Step 6: Commit**

```bash
git add ml/training/report.py ml/training/pipeline.py tests/ml/training/test_report.py
git commit -m "feat(pipeline): Stage 9 report.py (report.json/md + PNGs, matplotlib fail-soft)"
```

---

### Task 8: `baseline_drift.py` — train-window drift baseline

**Files:**
- Create: `ml/training/baseline_drift.py`
- Test: `tests/ml/training/test_baseline_drift.py`

(The spine's `_final_fit_and_save` already calls `write_baseline` behind a non-fatal try — once this module exists the baseline ships with every artifact.)

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/training/test_baseline_drift.py
import json
import numpy as np, pandas as pd
from ml.training.baseline_drift import write_baseline


def test_baseline_records_per_feature_stats(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(5, 2, 500)})
    p = write_baseline(df, ["a", "b"], tmp_path)
    assert p.name == "drift_baseline.json" and p.exists()
    base = json.loads(p.read_text())
    assert set(base["features"]) == {"a", "b"}
    assert abs(base["stats"]["b"]["mean"] - 5) < 0.5
    assert "p10" in base["stats"]["a"] and "p90" in base["stats"]["a"]
    assert base["n_rows"] == 500
```

- [ ] **Step 2: Run, confirm FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement** `ml/training/baseline_drift.py`:

```python
"""Write a train-window feature-distribution baseline so ml/eval drift checks
can fire later. Without a stored baseline the drift monitor has nothing to
compare live serving features against."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd


def write_baseline(df: pd.DataFrame, feature_cols: List[str], out_dir: Path) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for c in feature_cols:
        col = pd.to_numeric(df[c], errors="coerce").dropna()
        if col.empty:
            stats[c] = {"mean": None, "std": None, "p10": None, "p50": None, "p90": None}
            continue
        stats[c] = {
            "mean": float(col.mean()), "std": float(col.std()),
            "p10": float(col.quantile(0.10)), "p50": float(col.quantile(0.50)),
            "p90": float(col.quantile(0.90)),
        }
    payload = {"features": list(feature_cols), "n_rows": int(len(df)), "stats": stats}
    p = out_dir / "drift_baseline.json"
    p.write_text(json.dumps(payload, indent=2))
    return p
```

- [ ] **Step 4: Run, confirm PASS** + re-run core test (now also writes drift_baseline.json — `res.artifacts` includes it).

- [ ] **Step 5: Commit**

```bash
git add ml/training/baseline_drift.py tests/ml/training/test_baseline_drift.py
git commit -m "feat(pipeline): drift_baseline.json shipped with every artifact"
```

---

### Task 9: Migrate `momentum_lambdarank` onto the spine

The payoff: momentum gains EDA + quality + HPO + DSR/PBO + report, with the SAME artifact + metric contract so serving + existing tests are unaffected.

**Files:**
- Modify: `ml/training/trainers/momentum_lambdarank.py`
- Modify: `scripts/runpod/smoke_momentum_local.sh`
- Test: `tests/ml/training/test_momentum_pipeline.py`

- [ ] **Step 0: Confirm the label column name** — open `ml/labeling/ranking_labels.py` and verify `forward_return_quantile_labels(...)` returns columns named `relevance` and `fwd_return`. The existing `momentum_lambdarank` code merges on those and computes `_rank_ic` on `fwd_return`, so they should match. If the realized-return column has a different name, set `EngineSpec.fwd_return_col` to that name in Step 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/training/test_momentum_pipeline.py
import json
from datetime import date
from ml.training.trainers.momentum_lambdarank import MomentumTrainer, MomentumConfig, cached_universe
from ml.features.momentum_features import MOMENTUM_FEATURE_ORDER


def test_momentum_trainer_runs_via_spine(tmp_path):
    cfg = MomentumConfig(with_forecasts=False, start=date(2021, 1, 1), end=date(2026, 2, 1))
    t = MomentumTrainer(cfg=cfg, symbols=cached_universe(limit=8))
    res = t.train(tmp_path)
    m = res.metrics
    assert m["model"] == "momentum_lambdarank"
    assert m["n_features"] == len(MOMENTUM_FEATURE_ORDER)   # RS included (benchmark wired)
    assert "rank_ic_mean" in m and "eda" in m and "feature_audit" in m
    fo = json.loads((tmp_path / "feature_order.json").read_text())
    assert fo == list(MOMENTUM_FEATURE_ORDER)
    assert (tmp_path / "report.md").exists() and (tmp_path / "drift_baseline.json").exists()
    ok, why = t.serve_smoke(tmp_path)
    assert ok, why
```

- [ ] **Step 2: Run, confirm FAIL** (current `MomentumTrainer` is bespoke; no `eda`/`feature_audit` keys).

- [ ] **Step 3: Implement** — in `ml/training/trainers/momentum_lambdarank.py`:

(a) Add imports near the top:

```python
from ml.training.base import PipelineTrainer, TrainResult
from ml.training.specs import EngineSpec, CVSpec, EvalSpec, EDASpec
from ml.training.optuna_search import SearchSpace
```

(b) Add `hpo_trials: int = 0` to the `MomentumConfig` dataclass (so HPO is opt-in; the RunPod run sets e.g. `hpo_trials=30`).

(c) DELETE the bespoke `train_momentum()` function and the old `MomentumTrainer` class. Replace `MomentumTrainer` with:

```python
class MomentumTrainer(PipelineTrainer):
    name = "momentum_lambdarank"
    requires_gpu = False
    skip_promote_gate = True

    def __init__(self, cfg: Optional[MomentumConfig] = None, symbols: Optional[List[str]] = None):
        self.cfg = cfg or MomentumConfig()
        self.symbols = symbols or cached_universe()

    def engine_spec(self) -> EngineSpec:
        return EngineSpec(
            name=self.name, horizon=self.cfg.horizon,
            label_col="relevance", fwd_return_col="fwd_return",
            hpo_trials=getattr(self.cfg, "hpo_trials", 0),
            cv=CVSpec(n_folds=self.cfg.cv.n_folds, test_days=self.cfg.cv.test_days,
                      embargo_days=self.cfg.cv.embargo_days, train_days=self.cfg.cv.train_days),
            eval=EvalSpec(task="ranking", primary_metric="rank_ic_mean",
                          min_ic=0.02, min_icir=0.5),
            eda=EDASpec(max_nan_pct=0.50, min_abs_ic=0.0, run_ic_leakage=True,
                        max_leakage_corr=0.999, max_constant_features=8),
        )

    def load_panel(self) -> pd.DataFrame:
        return load_ohlcv(self.symbols, self.cfg.start, self.cfg.end)

    def build_features(self, panel):
        from ml.data.benchmark import load_nifty_benchmark  # noqa: PLC0415
        bench = load_nifty_benchmark(self.cfg.start, self.cfg.end)
        feats = build_momentum_features(panel, benchmark=bench)
        feature_cols = list(MOMENTUM_FEATURE_ORDER)
        if self.cfg.with_forecasts:
            from ml.features.forecast_features import (  # noqa: PLC0415
                FORECAST_FEATURES, kronos_forecast_features,
                merge_forecast_features, timesfm_forecast_features)
            tsfm = timesfm_forecast_features(panel, horizon=self.cfg.horizon, stride=self.cfg.forecast_stride)
            kronos = kronos_forecast_features(panel, horizon=self.cfg.horizon, stride=self.cfg.forecast_stride)
            feats = merge_forecast_features(feats, [tsfm, kronos])
            feature_cols += list(FORECAST_FEATURES)
        return feats[["date", "symbol", *feature_cols]], feature_cols

    def build_labels(self, panel):
        labels = forward_return_quantile_labels(
            panel[["date", "symbol", "close"]], horizon=self.cfg.horizon,
            n_quantiles=self.cfg.n_quantiles,
        )
        return labels[["date", "symbol", "relevance", "fwd_return"]]

    def make_model(self, params):
        import lightgbm as lgb  # noqa: PLC0415
        p = dict(self.cfg.lgbm_params); p.update(params or {})
        return lgb.LGBMRanker(**p)

    def fit_args(self, df_tr):
        return {"group": _groups_for(df_tr["date"])}

    def search_space(self):
        if not getattr(self.cfg, "hpo_trials", 0):
            return None
        return SearchSpace(suggest=lambda tr: {
            "num_leaves": tr.suggest_int("num_leaves", 15, 63),
            "learning_rate": tr.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": tr.suggest_int("min_child_samples", 20, 100),
            "subsample": tr.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": tr.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": tr.suggest_float("reg_lambda", 0.0, 5.0),
        })

    def serve_smoke(self, out_dir: Path) -> tuple[bool, str]:
        from ml.training.serve_smoke import smoke_artifact  # noqa: PLC0415
        return smoke_artifact(out_dir, "momentum_lambdarank.txt")
```

(d) Keep `_groups_for`, `_rank_ic`, `_decile_spread`, `cached_universe`, `MomentumConfig` (now with `hpo_trials`). The spine owns the rank-IC/decile/eval now, but leaving the module's helper functions is harmless (and `_groups_for` is used by `fit_args`). If `__main__` referenced `train_momentum`, repoint it:

```python
if __name__ == "__main__":
    import argparse, json as _json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train the momentum LambdaRank ranker")
    ap.add_argument("--with-forecasts", action="store_true")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hpo-trials", type=int, default=0)
    args = ap.parse_args()
    cfg = MomentumConfig(with_forecasts=args.with_forecasts, forecast_stride=args.stride,
                         hpo_trials=args.hpo_trials)
    t = MomentumTrainer(cfg=cfg, symbols=cached_universe(limit=args.limit))
    out = _ROOT / "artifacts" / "models" / "momentum_lambdarank"
    res = t.train(out)
    print(_json.dumps(res.metrics, indent=2, default=str))
```

(e) Update `scripts/runpod/smoke_momentum_local.sh` — replace the `train_momentum(...)` call:

```python
from ml.training.trainers.momentum_lambdarank import MomentumTrainer, MomentumConfig, cached_universe
# ...
t = MomentumTrainer(cfg=cfg, symbols=cached_universe(limit=6))
res = t.train(out)
m = res.metrics
```

And `scripts/runpod/train_momentum_gpu.sh` already invokes `python -m ml.training.trainers.momentum_lambdarank --with-forecasts --stride 5` — that still works via the new `__main__` (add `--hpo-trials 30` there for the real run).

- [ ] **Step 4: Run the migrated trainer test + the existing momentum tests**

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/ml/training/test_momentum_pipeline.py tests/ml/serving/test_momentum_engine.py tests/ml/features/test_momentum_features.py -v
```
Expected: all pass; `n_features == len(MOMENTUM_FEATURE_ORDER)`, `eda`/`feature_audit` present, serve-smoke green.

- [ ] **Step 5: Confirm discovery still finds it**

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m ml.training.runner --list | grep momentum
```
Expected: `momentum_lambdarank   CPU   deps=-`

- [ ] **Step 6: Commit**

```bash
git add ml/training/trainers/momentum_lambdarank.py scripts/runpod/smoke_momentum_local.sh tests/ml/training/test_momentum_pipeline.py
git commit -m "feat(momentum): migrate trainer onto the canonical spine (EDA+quality+HPO+DSR/PBO+report)"
```

---

### Task 10: Full-stack DL smoke + gate

**Files:** none (verification).

- [ ] **Step 1: Run the full-stack local smoke** (now exercises the spine end-to-end incl TimesFM+Kronos):

```bash
bash scripts/runpod/smoke_momentum_local.sh
```
Expected: `✓ FULL STACK GREEN`, `n_features >= 70`, serve-contract OK, and the run now also wrote `report.md` + `drift_baseline.json` and recorded `eda`/`feature_audit` in metrics. (Negative rank-IC on the 6-symbol smoke is still expected noise — the gate is "runs clean", not "model good".)

- [ ] **Step 2: Full suite + lint-imports green**

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest -q
lint-imports
```
Expected: ≥ 892 + new tests passing; `lint-imports` 1 kept/0 broken.

- [ ] **Step 3: Commit** any fixes from Steps 1–2.

---

## Notes

- **CV module choice:** the spine standardizes on `purged_cv.purged_walk_forward_by_date` (date-aware, embargoed — López de Prado) for all engines. The legacy `wfcv.walk_forward_split` (index-based, used by `lgbm_signal_gate`) stays for the not-yet-migrated trainers; new engines use the spine's purged CV.
- **HPO budget vs DSR null:** `n_trials` fed to `dsr_pbo_from_fold_returns` is `hpo_trials × n_folds` so the deflated-Sharpe null benchmark accounts for the full search — promoting on a Sharpe that only looks good because we tried many variants is exactly what DSR defends against.
- **Uniform metrics contract:** every spine run emits `model, primary_metric, primary_value, rank_ic_mean/std, rank_icir, decile_spread_mean, *_per_fold, deflated_sharpe, probability_backtest_overfitting, n_features/rows/symbols/dates/folds, eda, feature_audit, hpo, feature_importance, best_params, <name>_quality_pass`. This is what makes an admin model-compare view possible (observability follow-on).
- **Serve-smoke unchanged:** `serve_smoke.py` + `MomentumTrainer.serve_smoke` already close the train/serve-skew gate; the spine preserves the exact artifact contract (`<name>.txt` + `feature_order.json`) so it keeps working.
- **Legacy-trainer migration (follow-on, not this plan):** refactor `lgbm_signal_gate`, `qlib_alpha158`, `regime_hmm`, `tft_swing` onto the spine to kill the remaining copy-paste drift. They work today; this is cleanup once momentum proves the spine in the RunPod run.
- **Classification engines:** `EvalSpec.task="classification"` + `EDASpec.check_class_balance=True` are wired in the spec, but the spine's evaluation path here is ranking-first; a classification branch (precision/recall/F1) is a small additive follow-on when the first classification engine needs it (none in the momentum/swing/positional/intraday set — all are rankers).
```
