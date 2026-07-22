"""Regime history refresh — the ensemble → ``regime_history`` bridge.

Why this exists: the public regime timeline (/api/public/regime/history → the
/markets RegimeGauge) reads the ``regime_history`` table, which was appended by
a fragile 8:15 job (a serialized single-HMM model artifact + live provider
fetch + a scheduler that must be alive). When that job stalls the gauge
silently serves one stale row forever — which is exactly what happened
("Sideways · 100% confidence" frozen since Jul 1).

This module recomputes the timeline from what we fully control:

    NIFTY daily closes in the `candles` store
      → ml.regime.features.build_regime_features
      → ml.regime.ensemble.RegimeEnsemble (jump + HMM + rules, majority vote,
        3-day hysteresis — the REAL regime module)
      → idempotent per-day upsert into regime_history

Probabilities are the ensemble's honest agreement measure: the official state
gets `confidence` (fraction of models agreeing), the other two split the rest —
no more 0.9999 single-model overconfidence. Runs in a few seconds over ~500
bars; safe to call from the scheduler, boot pre-warm, or a manual backfill.
"""
from __future__ import annotations

import logging
from datetime import date as _date
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Post-close stamp (10:30 UTC = 16:00 IST) so intraday consumers reading
# "today's" row after the bell get a timestamp that is honestly post-settlement.
_STAMP_UTC = "T10:30:00+00:00"


def _read_series() -> tuple["object", Dict[str, float]]:
    """NIFTY OHLCV frame + {date→INDIAVIX close} from the candle store."""
    import pandas as pd
    from ...data.ohlc_store import pg_connect

    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT stock_symbol, timestamp::date AS dt, open, high, low, close, volume
                FROM candles
                WHERE interval='1d' AND stock_symbol IN ('NIFTY','INDIAVIX','VIX')
                ORDER BY timestamp
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    nifty = pd.DataFrame(
        [
            {"date": r[1], "open": float(r[2]), "high": float(r[3]),
             "low": float(r[4]), "close": float(r[5]),
             "volume": float(r[6] or 0)}
            for r in rows if r[0] == "NIFTY" and r[5] is not None
        ]
    )
    vix_by_date: Dict[str, float] = {}
    for r in rows:
        if r[0] in ("INDIAVIX", "VIX") and r[5] is not None:
            vix_by_date[r[1].isoformat()] = float(r[5])
    return nifty, vix_by_date


def refresh_regime_history(days: int = 180) -> Dict[str, Any]:
    """Recompute the ensemble regime over the candle store and upsert the last
    ``days`` sessions into ``regime_history`` (insert-missing-dates only, so
    repeated runs are no-ops). Returns a summary incl. the current regime.
    """
    out: Dict[str, Any] = {"inserted": 0, "current": None, "as_of": None}

    from ml.regime.features import build_regime_features
    from ml.regime.ensemble import RegimeEnsemble

    nifty, vix_by_date = _read_series()
    if nifty is None or len(nifty) < 120:
        logger.warning("regime refresh: insufficient NIFTY history (%s bars)", len(nifty) if nifty is not None else 0)
        return out

    feats = build_regime_features(nifty)
    ens = RegimeEnsemble().fit(feats)
    timeline = ens.run_online(feats).tail(days)
    if timeline.empty:
        return out

    close_by_date = {r["date"].isoformat(): r["close"] for _, r in nifty.iterrows()}

    from ...data.ohlc_store import pg_connect
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            first_day = timeline["date"].iloc[0]
            cur.execute(
                "SELECT detected_at::date FROM regime_history WHERE detected_at >= %s",
                (str(first_day),),
            )
            existing = {r[0].isoformat() for r in cur.fetchall()}

            to_insert = []
            for _, row in timeline.iterrows():
                d: _date = row["date"] if isinstance(row["date"], _date) else row["date"].date()
                iso = d.isoformat()
                if iso in existing:
                    continue
                state = str(row["state_name"])
                conf = float(row["confidence"])
                rest = round((1.0 - conf) / 2.0, 4)
                probs = {"bull": rest, "sideways": rest, "bear": rest}
                probs[state] = round(conf, 4)
                to_insert.append((
                    state, probs["bull"], probs["sideways"], probs["bear"],
                    vix_by_date.get(iso), close_by_date.get(iso),
                    f"{iso}{_STAMP_UTC}",
                ))

            if to_insert:
                cur.executemany(
                    """
                    INSERT INTO regime_history
                      (regime, prob_bull, prob_sideways, prob_bear, vix, nifty_close, detected_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    to_insert,
                )
                conn.commit()
            out["inserted"] = len(to_insert)
    finally:
        conn.close()

    last = timeline.iloc[-1]
    out["current"] = {
        "regime": str(last["state_name"]),
        "confidence": round(float(last["confidence"]), 4),
        "date": str(last["date"]),
    }
    out["as_of"] = str(last["date"])
    logger.info(
        "regime refresh: +%d rows, current=%s (%.0f%% agreement) as of %s",
        out["inserted"], out["current"]["regime"],
        out["current"]["confidence"] * 100, out["as_of"],
    )
    return out


def current_regime() -> Optional[Dict[str, Any]]:
    """Convenience: refresh (idempotent) and return just the current regime."""
    try:
        return refresh_regime_history(days=30).get("current")
    except Exception as e:  # noqa: BLE001
        logger.warning("current_regime failed: %s", e)
        return None
