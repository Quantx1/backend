"""
QlibEngine — production inference over the real Qlib Alpha158 pipeline.

On process start (or first ``load()`` call):
  - ``qlib.init(provider_uri=..., region=REG_CN)``
  - ``Alpha158`` handler pinned to the most recent ingestion window
  - LightGBM booster loaded via the model registry (B2 → disk fallback)

For the nightly rank job we take a snapshot of the final trading day,
ask the handler to compute features for the instrument universe, and
let the LightGBM booster score every row. Cross-sectional rank =
``alpha_scores`` payload.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ...core.config import settings

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROVIDER = Path("~/.qlib/qlib_data/nse_data").expanduser()
_DISK_FALLBACK = _ROOT / "artifacts" / "models" / "qlib_alpha158"
_MODEL_NAME = "qlib_alpha158"


class QlibEngine:
    """Thread-safe after ``load()``. Singleton in production via
    ``get_qlib_engine()``."""

    _lock = threading.Lock()

    def __init__(self, provider_uri: Optional[Path] = None):
        self.provider_uri = Path(
            provider_uri or _DEFAULT_PROVIDER
        ).expanduser().resolve()
        self._booster = None
        self._meta: Dict = {}
        self._qlib_ready = False

    @property
    def loaded(self) -> bool:
        return self._booster is not None and self._qlib_ready

    # ----------------------------------------------------------------- load

    def load(self) -> bool:
        """Initialize Qlib + load the trained booster. Returns True on
        full success. Failure modes (missing provider dir, missing model,
        missing Qlib install) are all handled by logging + returning
        False — the caller treats that as model-not-ready."""
        with self._lock:
            if self.loaded:
                return True
            return self._load_locked()

    def _load_locked(self) -> bool:
        # (1) Qlib itself
        try:
            import qlib
            from qlib.constant import REG_CN
        except Exception as e:
            logger.warning("pyqlib not installed (%s) — QlibEngine disabled", e)
            return False

        if not self.provider_uri.exists():
            logger.info(
                "Qlib provider dir missing (%s). Run scripts/data/ingest_nse_to_qlib.py.",
                self.provider_uri,
            )
            return False

        try:
            qlib.init(provider_uri=str(self.provider_uri), region=REG_CN)
            self._qlib_ready = True
        except Exception as e:
            logger.error("qlib.init failed: %s", e)
            return False

        # (2) LightGBM booster via registry → disk fallback
        try:
            import lightgbm as lgb
        except Exception as e:
            logger.warning("lightgbm missing: %s", e)
            return False

        model_path, meta_path = self._resolve_paths()
        if model_path is None:
            logger.info(
                "qlib_alpha158 model not yet trained. Run scripts/train/train_qlib_alpha158.py."
            )
            return False

        try:
            # model_str avoids a native model_file parser crash on Windows.
            with open(model_path, "r", encoding="utf-8") as _fh:
                self._booster = lgb.Booster(model_str=_fh.read())
        except Exception as e:
            logger.error("Booster load failed: %s", e)
            return False

        if meta_path is not None and meta_path.exists():
            try:
                self._meta = json.loads(meta_path.read_text())
            except Exception as e:
                logger.debug("Meta file unreadable: %s", e)

        logger.info(
            "QlibEngine loaded — instruments=%s trained_at=%s qlib=%s",
            self._meta.get("instruments", "unknown"),
            self._meta.get("trained_at", "unknown"),
            self._meta.get("qlib_version", "?"),
        )
        return True

    def _resolve_paths(self):
        try:
            from ..registry import resolve_model_file  # type: ignore
            model_path = resolve_model_file(
                _MODEL_NAME, "qlib_alpha158.txt", _DISK_FALLBACK / "qlib_alpha158.txt",
            )
            meta_path = resolve_model_file(
                _MODEL_NAME, "qlib_alpha158_meta.json",
                _DISK_FALLBACK / "qlib_alpha158_meta.json",
            )
            return model_path, meta_path
        except Exception as e:
            logger.debug("Registry resolve failed: %s", e)
        disk_model = _DISK_FALLBACK / "qlib_alpha158.txt"
        disk_meta = _DISK_FALLBACK / "qlib_alpha158_meta.json"
        if disk_model.exists():
            return disk_model, (disk_meta if disk_meta.exists() else None)
        return None, None

    # -------------------------------------------------------------- ranking

    def rank_universe(
        self,
        *,
        instruments: str = "nse_all",
        trade_date: Optional[date] = None,
    ) -> List[dict]:
        """Return one ``{symbol, qlib_rank, qlib_score_raw, top_factors}``
        record per instrument for the trade_date (default: latest NSE
        session before ``date.today()``).

        This is the one hot-path the nightly job runs; kept as a single
        call so we instantiate the Alpha158 handler only once per run.
        """
        if not self.loaded:
            return []

        from qlib.contrib.data.handler import Alpha158

        horizon = int(self._meta.get("horizon_days", 5))
        end = trade_date or (date.today() - timedelta(days=1))
        start = end - timedelta(days=90)  # window needed for Alpha158 rolling

        try:
            handler = Alpha158(
                instruments=instruments,
                start_time=start.isoformat(),
                end_time=end.isoformat(),
                fit_start_time=start.isoformat(),
                fit_end_time=end.isoformat(),
                label=([f"Ref($close, -{horizon}) / $close - 1"], ["LABEL0"]),
            )
            feat = handler.fetch(col_set="feature", data_key="infer")
        except Exception as e:
            logger.warning("Alpha158 fetch failed: %s", e)
            return []

        if feat is None or feat.empty:
            return []

        # Only the most recent date per instrument — that's what we rank on.
        latest_idx = feat.reset_index().groupby("instrument")["datetime"].idxmax()
        latest = feat.reset_index().loc[latest_idx].set_index(["datetime", "instrument"])

        X = latest.values.astype("float32")
        scores = self._booster.predict(X)

        rows = []
        for (dt, inst), score in zip(latest.index, scores):
            rows.append({
                "symbol": str(inst).upper(),
                "trade_date": pd.Timestamp(dt).date().isoformat(),
                "qlib_score_raw": round(float(score), 6),
            })

        # Cross-sectional rank — highest score first.
        rows.sort(key=lambda r: r["qlib_score_raw"], reverse=True)
        for i, r in enumerate(rows, start=1):
            r["qlib_rank"] = i

        return rows


# --------------------------------------------------------------- singleton

_engine: Optional[QlibEngine] = None
_engine_lock = threading.Lock()


def get_qlib_engine() -> QlibEngine:
    """Process-wide singleton. First caller triggers Qlib init + model
    load; subsequent callers reuse the handle. Safe to call when the
    provider dir doesn't exist yet — ``loaded`` stays False and
    ``rank_universe()`` returns []."""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            provider = getattr(settings, "QLIB_PROVIDER_URI", None)
            _engine = QlibEngine(provider_uri=provider)
            _engine.load()
    return _engine
