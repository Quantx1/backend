"""NSE data facade — FII/DII flow + F&O OI spurts + delivery % + bulk deals.

Used by scanners 34 (High Delivery %), 35 (Bulk Deals), 36-42 (FII/DII/OI
buildup, long/short buildup, long unwinding, short covering) and 87/88
(Long Unwinding, OI Spike).

Created 2026-05-31 to unblock scanners that previously raised ImportError
on `from .nse_data import get_nse_data` (the module didn't exist; failures
were silently swallowed and the scanners returned empty pd.DataFrame()).

Data sources (in priority order):
    1. NSE live `fiidiiTradeReact` API via `ml.data.fii_dii_history` —
       only working free source per memory `project_high_quality_data_2026_05_12`.
       Returns today's row when NSE responds; cached parquet otherwise.
    2. jugaad-data F&O bhavcopy — used for end-of-day OI deltas. Often
       rate-limited / bot-blocked; degrades to empty gracefully.
    3. Kite admin (premium tier) for live NFO quotes with OI — used when
       admin Kite token is healthy.

No synthetic data per memory lock `project_no_fallbacks_no_refunds_2026_04_19`.
When all sources fail, returns empty + sets `last_error` so callers can
surface "data unavailable" rather than show fake numbers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# In-process TTL so we don't hammer NSE / jugaad on every scanner invocation
_TTL_S = 300.0   # 5 minutes — institutional flow data updates slowly
_cache: Dict[str, tuple] = {}


def _coerce_int(v) -> Optional[int]:
    """NSE CSV numbers carry thousands separators ('1,00,000') and '-'."""
    try:
        return int(float(str(v).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


def _coerce_float(v) -> Optional[float]:
    try:
        f = float(str(v).replace(",", "").strip())
        return None if f != f else f  # reject NaN (blank NSE cells)
    except (TypeError, ValueError):
        return None


def _front_month_price_change(fut: "pd.DataFrame") -> Dict[str, float]:
    """Per-symbol day price change from the FRONT-month futures contract.

    The F&O bhavcopy carries each contract's OPEN/CLOSE, so the day move is
    (close - ref)/ref where ref = prev close if present else open. This is the
    price-direction leg the OI build-up buckets need (long/short build-up,
    unwinding, short covering) — it was previously hardcoded to 0.0, which left
    every price-directional bucket permanently empty despite real OI deltas."""
    try:
        f = fut.copy()
        if "EXPIRY_DT" in f.columns:
            f["_exp"] = pd.to_datetime(f["EXPIRY_DT"], errors="coerce")
            f = f.sort_values("_exp")
        if "SYMBOL" not in f.columns:
            return {}
        front = f.groupby("SYMBOL").first().reset_index()
        out: Dict[str, float] = {}
        for _, r in front.iterrows():
            close = _coerce_float(r.get("CLOSE"))
            ref = _coerce_float(r.get("PREV_CLOSE")) or _coerce_float(r.get("OPEN"))
            if close and ref and ref > 0:
                out[str(r.get("SYMBOL", ""))] = round((close - ref) / ref * 100, 2)
        return out
    except Exception:
        return {}


@dataclass
class NSEFlowSnapshot:
    """Today's institutional flow snapshot."""
    fii_net: float = 0.0          # ₹ Cr — positive = FII buying
    dii_net: float = 0.0
    date: Optional[str] = None
    source: str = "unavailable"   # nse_live | cache | unavailable
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fii_net": round(self.fii_net, 2),
            "dii_net": round(self.dii_net, 2),
            "date": self.date,
            "source": self.source,
            "last_error": self.last_error,
        }


@dataclass
class OIRow:
    """One row of F&O participant OI data."""
    symbol: str
    change_pct: float = 0.0      # spot price change today
    oi: int = 0                   # current OI (futures)
    oi_change_pct: float = 0.0    # ΔOI / prior_OI × 100
    volume: int = 0
    classification: Optional[str] = None  # long_buildup | short_buildup | long_unwinding | short_covering | oi_spike


class NSEDataProvider:
    """Facade over the working FII/DII + best-effort OI sources."""

    def get_fii_dii(self) -> NSEFlowSnapshot:
        """Today's FII/DII net (₹ Cr). Calls NSE live; caches 5 min."""
        cache_key = "fii_dii"
        now = time.monotonic()
        hit = _cache.get(cache_key)
        if hit and now - hit[0] < _TTL_S:
            return hit[1]

        snap = NSEFlowSnapshot()
        try:
            # Lazy import — fii_dii_history is in the ml/ tree, not always loaded
            import sys
            from pathlib import Path
            ml_path = str(Path(__file__).resolve().parents[3] / "ml")
            if ml_path not in sys.path:
                sys.path.insert(0, ml_path)
            from data.fii_dii_history import backfill_today_via_nse_live, _load_cache

            today_df = backfill_today_via_nse_live()
            if not today_df.empty:
                row = today_df.iloc[0]
                snap.fii_net = float(row.get("fii_net", 0) or 0)
                snap.dii_net = float(row.get("dii_net", 0) or 0)
                snap.date = today_df.index[0].date().isoformat()
                snap.source = "nse_live"
            else:
                # Fall back to last cached row
                cached = _load_cache()
                if not cached.empty:
                    row = cached.iloc[-1]
                    snap.fii_net = float(row.get("fii_net", 0) or 0)
                    snap.dii_net = float(row.get("dii_net", 0) or 0)
                    snap.date = cached.index[-1].date().isoformat()
                    snap.source = "cache"
                    snap.last_error = "nse_live empty; using last cached row"
                else:
                    snap.last_error = "nse_live empty and no cached history"
        except Exception as e:
            snap.last_error = f"{type(e).__name__}: {str(e)[:200]}"
            logger.debug("nse_data.get_fii_dii fetch failed: %s", e)

        _cache[cache_key] = (now, snap)
        return snap

    def get_fii_dii_activity(self) -> Dict[str, Any]:
        """Backwards-compatible shape for the legacy scanners 36-38."""
        snap = self.get_fii_dii()
        return {
            "fii_net": snap.fii_net,
            "dii_net": snap.dii_net,
            "source": snap.source,
            "date": snap.date,
        }

    # ── OI spurts ─────────────────────────────────────────────────

    def get_participant_oi(self) -> Dict[str, Any]:
        """End-of-day F&O OI spurts ranked by absolute OI change.

        Calls jugaad bhavcopy_fo_raw for yesterday vs day-before to get
        per-symbol OI deltas. NSE often rate-limits these so degrades
        cleanly to empty rather than crashing the scanner.
        """
        cache_key = "oi_spurts"
        now = time.monotonic()
        hit = _cache.get(cache_key)
        if hit and now - hit[0] < _TTL_S:
            return hit[1]

        out: Dict[str, Any] = {"data": [], "source": "unavailable", "last_error": None}
        try:
            from jugaad_data.nse import bhavcopy_fo_raw

            today = date.today()
            # F&O bhavcopy publishes after market close; try yesterday first
            for delta in range(1, 5):
                d = today - timedelta(days=delta)
                if d.weekday() >= 5:
                    continue
                try:
                    raw = bhavcopy_fo_raw(d)
                    if raw:
                        # bhavcopy returns CSV text — parse to DataFrame
                        from io import StringIO
                        df = pd.read_csv(StringIO(raw))
                        # Columns: TIMESTAMP, INSTRUMENT, SYMBOL, EXPIRY_DT, ...
                        # OPEN_INT, CHG_IN_OI, VOL_LK, etc. (column names vary
                        # across NSE format revisions — defensive .get())
                        fut = df[df["INSTRUMENT"].astype(str).str.startswith("FUT")] \
                            if "INSTRUMENT" in df.columns else df
                        if fut.empty:
                            continue
                        # Aggregate per symbol (sum across expiries)
                        agg = fut.groupby("SYMBOL").agg({
                            "OPEN_INT": "sum",
                            "CHG_IN_OI": "sum" if "CHG_IN_OI" in fut.columns else "first",
                        }).reset_index() if "SYMBOL" in fut.columns else pd.DataFrame()
                        if agg.empty:
                            continue
                        # price-direction leg per symbol (front-month futures
                        # close vs open/prev-close) — real, not the 0.0 stub
                        # that left every build-up/unwinding bucket empty.
                        price_chg = _front_month_price_change(fut)
                        rows: List[Dict[str, Any]] = []
                        for _, r in agg.iterrows():
                            oi = int(r.get("OPEN_INT", 0) or 0)
                            chg = int(r.get("CHG_IN_OI", 0) or 0)
                            if oi <= 0:
                                continue
                            oi_chg_pct = chg / (oi - chg) * 100 if (oi - chg) > 0 else 0.0
                            sym = str(r.get("SYMBOL", ""))
                            rows.append({
                                "symbol": sym,
                                "oi": oi,
                                "oi_change": chg,
                                "oi_change_pct": round(oi_chg_pct, 2),
                                "change_pct": price_chg.get(sym, 0.0),
                            })
                        out["data"] = sorted(rows, key=lambda r: abs(r["oi_change_pct"]), reverse=True)[:200]
                        out["source"] = f"bhavcopy_{d.isoformat()}"
                        break
                except Exception as e:
                    logger.debug("bhavcopy %s fetch failed: %s", d, e)
                    continue

            if not out["data"]:
                out["last_error"] = "bhavcopy_fo_raw returned empty across last 4 sessions"

        except ImportError:
            out["last_error"] = "jugaad-data not installed"
        except Exception as e:
            out["last_error"] = f"{type(e).__name__}: {str(e)[:200]}"

        _cache[cache_key] = (now, out)
        return out

    def get_oi_spurts(self) -> Dict[str, Any]:
        """Alias for get_participant_oi (legacy scanner name)."""
        return self.get_participant_oi()

    # ── Delivery % + bulk deals (scanners 34/35) ──────────────────────

    def get_delivery_data(self) -> pd.DataFrame:
        """Security-wise delivery % for the latest available session.

        Source: NSE ``sec_bhavdata_full`` archive via jugaad-data
        ``full_bhavcopy_raw``. The NSE CSV prefixes every column AND its
        SERIES / DELIV_PER values with a leading space, so both are
        stripped. EQ series only; rows where DELIV_PER is '-' (no
        delivery, e.g. some non-EQ series) are dropped — not coerced to 0.

        Returns ``DataFrame[symbol, delivery_pct]`` — the exact contract
        scanner 34 (``_filter_high_delivery``) consumes. Empty when NSE is
        unavailable; never synthetic (no-fallbacks lock).
        """
        cache_key = "delivery"
        now = time.monotonic()
        hit = _cache.get(cache_key)
        if hit and now - hit[0] < _TTL_S:
            return hit[1]

        out = pd.DataFrame(columns=["symbol", "delivery_pct"])
        last_error: Optional[str] = None
        try:
            from io import StringIO

            from jugaad_data.nse import full_bhavcopy_raw

            today = date.today()
            for delta in range(1, 6):  # walk back up to 5 days for the last session
                d = today - timedelta(days=delta)
                if d.weekday() >= 5:
                    continue
                try:
                    raw = full_bhavcopy_raw(d)
                except Exception as e:  # NSE timeout / 404 on a non-trading day
                    last_error = f"{type(e).__name__}: {str(e)[:120]}"
                    continue
                if not raw or "SYMBOL" not in raw:
                    continue
                df = pd.read_csv(StringIO(raw))
                df.columns = [c.strip() for c in df.columns]
                if "DELIV_PER" not in df.columns or "SYMBOL" not in df.columns:
                    continue
                # Values also carry the leading space in the NSE CSV.
                df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
                if "SERIES" in df.columns:
                    df["SERIES"] = df["SERIES"].astype(str).str.strip()
                    df = df[df["SERIES"] == "EQ"]
                df["delivery_pct"] = pd.to_numeric(df["DELIV_PER"], errors="coerce")
                df = df.dropna(subset=["delivery_pct"])
                out = (
                    df[["SYMBOL", "delivery_pct"]]
                    .rename(columns={"SYMBOL": "symbol"})
                    .reset_index(drop=True)
                )
                break
        except ImportError:
            last_error = "jugaad-data not installed"
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:120]}"
        if last_error:
            logger.debug("get_delivery_data: %s", last_error)

        _cache[cache_key] = (now, out)
        return out

    def get_bulk_deals(self) -> List[Dict[str, Any]]:
        """Today's bulk deals from NSE (``/content/equities/bulk.csv``).

        Source: jugaad-data ``NSEArchives.bulk_deals_raw`` — an *instance*
        method, not a top-level export. Returns a list of dicts each
        carrying at least ``symbol`` (scanner 35's contract) plus
        best-effort client / side / qty / price. Empty when NSE blocks or
        rate-limits; never synthetic.
        """
        cache_key = "bulk_deals"
        now = time.monotonic()
        hit = _cache.get(cache_key)
        if hit and now - hit[0] < _TTL_S:
            return hit[1]

        deals: List[Dict[str, Any]] = []
        last_error: Optional[str] = None
        try:
            from io import StringIO

            from jugaad_data.nse import NSEArchives

            raw = NSEArchives().bulk_deals_raw()
            if raw and "Symbol" in raw:
                df = pd.read_csv(StringIO(raw))
                df.columns = [c.strip() for c in df.columns]
                for _, r in df.iterrows():
                    sym = str(r.get("Symbol", "")).strip()
                    if not sym or sym.lower() == "nan":
                        continue
                    deals.append({
                        "symbol": sym,
                        "client": (str(r.get("Client Name", "")).strip() or None),
                        "side": (str(r.get("Buy/Sell", "")).strip() or None),
                        "qty": _coerce_int(r.get("Quantity Traded")),
                        "price": _coerce_float(r.get("Trade Price / Wght. Avg. Price")),
                        "date": (str(r.get("Date", "")).strip() or None),
                    })
            else:
                last_error = "bulk.csv empty or unexpected format"
        except ImportError:
            last_error = "jugaad-data not installed"
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:120]}"
        if last_error:
            logger.debug("get_bulk_deals: %s", last_error)

        _cache[cache_key] = (now, deals)
        return deals


_singleton: Optional[NSEDataProvider] = None


def get_nse_data() -> NSEDataProvider:
    """Singleton accessor — keeps the in-process cache shared."""
    global _singleton
    if _singleton is None:
        _singleton = NSEDataProvider()
    return _singleton
