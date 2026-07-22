"""Market Pulse — the market-internals layer of the daily desk.

One SEBI-safe payload, computed entirely from EOD data we already hold:

  breadth   %-above 20/50/200-DMA across the candle store, today's A/D,
            52-week new-high/new-low counts, and a composite 0-100
            Breadth Score (documented, deterministic — no black box).
  vol       NIFTY realized vol (HV 10/20/30) vs the INDIAVIX close —
            the "options rich/cheap vs realized" read.
  flows     FII / DII streak intelligence: consecutive net-buy/sell run
            length + cumulative ₹ Cr during the run (from the EOD series).
  diff      "What changed vs yesterday" — the same internals computed as
            of the previous session, diffed, and emitted only when the
            delta is meaningful (the 10-second catch-up strip).

Everything is EOD-derived analytics (SEBI Path-A safe when labelled).
Mirrors breadth.py's shape: pure aggregation + one candles window query,
10-minute in-process cache, honest-empty on failure, coverage disclosed.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CACHE: Dict[str, tuple] = {}
_TTL_S = 600

# Index/derivative series that live in `candles` but are not equities —
# excluded from issue counts so breadth reflects stocks only.
_NON_EQUITY = (
    "NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY",
    "INDIAVIX", "VIX",
)

# 52-week proximity thresholds — mirror the scanner definitions
# (backend/data/screener/filters.py _filter_52w_high / _filter_52w_low).
_HIGH_TOL = 0.98
_LOW_TOL = 1.05


def _snapshot(cutoff: Optional[date]) -> Optional[Dict[str, Any]]:
    """Cross-sectional internals as of `cutoff` (None = latest bar per symbol).

    One window query over `candles`; per symbol take the latest bar at/before
    the cutoff, the 20/50/200-bar SMAs and the 252-bar high/low ending there.
    Symbols whose last bar is >7 calendar days stale are dropped (delisted /
    dead tickers must not pollute the denominators).
    """
    from ...data.ohlc_store import pg_connect

    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH ranked AS (
                  SELECT stock_symbol,
                         timestamp::date AS dt,
                         close, high, low,
                         row_number() OVER (PARTITION BY stock_symbol
                                            ORDER BY timestamp DESC) AS rn
                  FROM candles
                  WHERE interval = '1d'
                    AND timestamp >= now() - interval '420 days'
                    AND (%s::date IS NULL OR timestamp::date <= %s::date)
                    AND stock_symbol NOT IN %s
                )
                SELECT stock_symbol,
                       max(CASE WHEN rn = 1 THEN close END) AS last_close,
                       max(CASE WHEN rn = 1 THEN dt END)    AS last_dt,
                       max(CASE WHEN rn = 2 THEN close END) AS prev_close,
                       avg(close) FILTER (WHERE rn <= 20)   AS sma20,
                       avg(close) FILTER (WHERE rn <= 50)   AS sma50,
                       avg(close) FILTER (WHERE rn <= 200)  AS sma200,
                       count(*)                             AS bars,
                       max(high) FILTER (WHERE rn <= 252)   AS high52,
                       min(low)  FILTER (WHERE rn <= 252)   AS low52
                FROM ranked
                GROUP BY stock_symbol
                """,
                (cutoff, cutoff, _NON_EQUITY),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    as_of = max(r[2] for r in rows if r[2] is not None)
    stale_floor = as_of - timedelta(days=7)

    total = above20 = n20 = above50 = n50 = above200 = n200 = 0
    adv = dec = 0
    new_highs = new_lows = 0
    nhl_n = 0
    fresh_today = 0

    for (_sym, last_close, last_dt, prev_close, sma20, sma50, sma200,
         bars, high52, low52) in rows:
        if last_close is None or last_dt is None or last_dt < stale_floor:
            continue
        total += 1
        if last_dt == as_of:
            fresh_today += 1
            if prev_close is not None:
                if last_close > prev_close:
                    adv += 1
                elif last_close < prev_close:
                    dec += 1
        if sma20 is not None and bars >= 20:
            n20 += 1
            above20 += last_close > float(sma20)
        if sma50 is not None and bars >= 50:
            n50 += 1
            above50 += last_close > float(sma50)
        if sma200 is not None and bars >= 200:
            n200 += 1
            above200 += last_close > float(sma200)
        if bars >= 200 and high52 and low52:
            nhl_n += 1
            if last_close >= float(high52) * _HIGH_TOL:
                new_highs += 1
            elif last_close <= float(low52) * _LOW_TOL:
                new_lows += 1

    if total == 0:
        return None

    pct = lambda a, n: round(a / n * 100.0, 1) if n else None  # noqa: E731
    pct20, pct50, pct200 = pct(above20, n20), pct(above50, n50), pct(above200, n200)
    ad_pct = pct(adv, adv + dec) if (adv + dec) else None
    hl_pct = pct(new_highs, new_highs + new_lows) if (new_highs + new_lows) else 50.0

    # Composite Breadth Score — a documented weighted mean, not a model:
    #   35% trend participation (>50DMA) · 25% long-trend health (>200DMA)
    #   25% today's A/D · 15% new-high dominance. Missing parts renormalize.
    parts = [(pct50, 0.35), (pct200, 0.25), (ad_pct, 0.25), (hl_pct, 0.15)]
    avail = [(v, w) for v, w in parts if v is not None]
    score = round(sum(v * w for v, w in avail) / sum(w for _, w in avail)) if avail else None
    band = (None if score is None else
            "Strong" if score >= 65 else
            "Healthy" if score >= 55 else
            "Neutral" if score >= 45 else
            "Weak" if score >= 35 else "Washed-out")

    return {
        "as_of": as_of.isoformat(),
        "coverage": {"symbols": total, "fresh_today": fresh_today},
        "pct_above_20dma": pct20,
        "pct_above_50dma": pct50,
        "pct_above_200dma": pct200,
        "adv": adv, "dec": dec, "ad_pct": ad_pct,
        "new_highs": new_highs, "new_lows": new_lows,
        "score": score, "band": band,
    }


def _delivery_intel(limit: int = 6) -> Optional[Dict[str, Any]]:
    """Delivery-% accumulation read from our own EOD candle store (SEBI-safe
    derived analytics): names whose latest delivery %% is ≥1.4× their trailing
    30-session average WITH price up on the day — the classic "strong hands
    accumulating" flag — plus the market-wide average delivery %%."""
    from ...data.ohlc_store import pg_connect

    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH ranked AS (
                  SELECT stock_symbol, timestamp::date AS dt, close, delivery_pct,
                         lag(close) OVER (PARTITION BY stock_symbol ORDER BY timestamp) AS prev_close,
                         row_number() OVER (PARTITION BY stock_symbol ORDER BY timestamp DESC) AS rn
                  FROM candles
                  WHERE interval='1d' AND timestamp >= now() - interval '60 days'
                    AND delivery_pct IS NOT NULL
                    AND stock_symbol NOT IN %s
                ),
                agg AS (
                  SELECT stock_symbol,
                         max(CASE WHEN rn=1 THEN delivery_pct END) AS dlv,
                         max(CASE WHEN rn=1 THEN close END)        AS last_close,
                         max(CASE WHEN rn=1 THEN prev_close END)   AS prev_close,
                         max(CASE WHEN rn=1 THEN dt END)           AS dt,
                         avg(delivery_pct) FILTER (WHERE rn BETWEEN 2 AND 31) AS avg30,
                         count(*) AS bars
                  FROM ranked GROUP BY stock_symbol
                )
                SELECT stock_symbol, dlv, avg30, last_close, prev_close, dt
                FROM agg
                WHERE dlv IS NOT NULL AND avg30 IS NOT NULL AND bars >= 15
                  AND dt = (SELECT max(dt) FROM agg)
                """,
                (_NON_EQUITY,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return None
    dlvs = [float(r[1]) for r in rows if r[1] is not None]
    spikes = []
    for sym, dlv, avg30, last_close, prev_close, _dt in rows:
        dlv, avg30 = float(dlv), float(avg30)
        if avg30 <= 0 or last_close is None or prev_close is None:
            continue
        chg = (float(last_close) / float(prev_close) - 1.0) * 100.0
        if dlv >= avg30 * 1.4 and dlv >= 40.0 and chg > 0:
            spikes.append({
                "symbol": sym, "delivery_pct": round(dlv, 1),
                "avg_30d": round(avg30, 1), "change_pct": round(chg, 2),
            })
    spikes.sort(key=lambda s: s["delivery_pct"] / max(s["avg_30d"], 1e-9), reverse=True)
    return {
        "market_avg_delivery_pct": round(sum(dlvs) / len(dlvs), 1) if dlvs else None,
        "accumulation_count": len(spikes),
        "spikes": spikes[:limit],
        "note": "EOD · derived from published delivery data",
    }


def _nifty_vol() -> Dict[str, Any]:
    """NIFTY HV(10/20/30) from stored EOD closes + latest INDIAVIX close."""
    from ...data.ohlc_store import pg_connect

    out: Dict[str, Any] = {"hv": None, "vix": None, "vix_prev": None, "read": None}
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT stock_symbol, timestamp, close FROM candles
                WHERE interval='1d' AND stock_symbol IN ('NIFTY','INDIAVIX','VIX')
                  AND timestamp >= now() - interval '400 days'
                ORDER BY timestamp
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    nifty = [float(c) for s, _t, c in rows if s == "NIFTY" and c]
    vix_rows = [float(c) for s, _t, c in rows if s in ("INDIAVIX", "VIX") and c]
    # NIFTY short returns — feeds the index-vs-breadth divergence detector.
    if len(nifty) >= 6:
        out["nifty_ret_1d"] = round((nifty[-1] / nifty[-2] - 1.0) * 100.0, 2)
        out["nifty_ret_5d"] = round((nifty[-1] / nifty[-6] - 1.0) * 100.0, 2)
    if len(nifty) >= 30:
        try:
            from ..fno_scanner.volatility import compute_hv
            hv = compute_hv(nifty)
            if hv:
                out["hv"] = {k: round(v, 1) for k, v in hv["hv"].items()}
                out["latest_hv"] = round(hv["latest_hv"], 1)
        except Exception as e:  # pragma: no cover
            logger.debug("market_pulse hv failed: %s", e)
    if vix_rows:
        out["vix"] = round(vix_rows[-1], 2)
        if len(vix_rows) >= 2:
            out["vix_prev"] = round(vix_rows[-2], 2)
    hv20 = (out.get("hv") or {}).get("20")
    if hv20 and out["vix"]:
        ratio = out["vix"] / hv20
        out["read"] = ("options rich vs realized" if ratio > 1.15 else
                       "options cheap vs realized" if ratio < 0.85 else
                       "options fair vs realized")
        out["vix_vs_hv20"] = round(ratio, 2)
    return out


def _flow_streaks() -> Dict[str, Any]:
    """FII / DII net-flow streaks from the EOD series (₹ Cr, provisional).

    A streak = consecutive sessions with the same flow sign, walking back
    from the most recent published day. Nobody publishes this computed —
    it's the "FII sold 7 straight sessions, ₹18,400 Cr" read."""
    out: Dict[str, Any] = {"fii": None, "dii": None, "last_date": None}
    try:
        from ml.data.fii_dii_history import fii_dii_series
        end = date.today()
        df = fii_dii_series(end - timedelta(days=60), end)
        if df is None or df.empty:
            return out
        df = df.dropna(how="all").sort_index()
        out["last_date"] = df.index.max().date().isoformat()
        for col, key in (("fii_net", "fii"), ("dii_net", "dii")):
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if s.empty:
                continue
            sign = 1 if s.iloc[-1] >= 0 else -1
            days = 0
            cum = 0.0
            for v in reversed(s.tolist()):
                if (1 if v >= 0 else -1) != sign:
                    break
                days += 1
                cum += float(v)
            out[key] = {
                "side": "buying" if sign > 0 else "selling",
                "days": days,
                "cum_cr": round(cum, 0),
            }
    except Exception as e:
        logger.debug("market_pulse flows failed: %s", e)
    return out


_POS_CACHE: Dict[str, tuple] = {}
_POS_TTL_S = 3600


def _fii_positioning() -> Optional[Dict[str, Any]]:
    """FII index-futures positioning from NSE's EOD-published participant-wise
    OI file (via nselib): long/short contracts, long-share %, and the delta vs
    the prior session. The read every institutional desk checks pre-open —
    published EOD statistics, labelled as such. Cached 1h; honest-None."""
    hit = _POS_CACHE.get("p")
    if hit and (time.monotonic() - hit[0]) < _POS_TTL_S:
        return hit[1]
    try:
        from nselib import derivatives

        def fetch(d: date) -> Optional[Dict[str, float]]:
            try:
                df = derivatives.participant_wise_open_interest(trade_date=d.strftime("%d-%m-%Y"))
                row = df[df["Client Type"].astype(str).str.strip() == "FII"]
                if row.empty:
                    return None
                long_ = float(row["Future Index Long"].iloc[0])
                short_ = float(row["Future Index Short"].iloc[0])
                if long_ + short_ <= 0:
                    return None
                return {"long": long_, "short": short_}
            except Exception:
                return None

        found: list[tuple[str, Dict[str, float]]] = []
        d = date.today()
        for _ in range(10):
            r = fetch(d)
            if r:
                found.append((d.isoformat(), r))
                if len(found) == 2:
                    break
            d -= timedelta(days=1)
        if not found:
            return None

        day, cur = found[0]
        total = cur["long"] + cur["short"]
        out: Dict[str, Any] = {
            "date": day,
            "long": int(cur["long"]),
            "short": int(cur["short"]),
            "net": int(cur["long"] - cur["short"]),
            "long_share_pct": round(cur["long"] / total * 100.0, 1),
            "label": "NSE · EOD published (participant-wise OI)",
        }
        if len(found) == 2:
            _, prev = found[1]
            out["net_prev"] = int(prev["long"] - prev["short"])
            out["net_delta"] = out["net"] - out["net_prev"]
        _POS_CACHE["p"] = (time.monotonic(), out)
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug("fii positioning failed: %s", e)
        return None


_VAL_CACHE: Dict[str, tuple] = {}
_VAL_TTL_S = 3600


def _valuation() -> Optional[Dict[str, Any]]:
    """Market valuation snapshot from NSE's EOD-published per-symbol P/E file
    (nselib pe_ratio) joined with our index_constituents membership: median
    P/E for NIFTY 50 / NIFTY 500 / whole file, plus expensiveness breadth
    (% above 50x) and the value pocket (% below 15x). Derived from published
    statistics — SEBI-safe, labelled. Cached 1h; honest-None."""
    hit = _VAL_CACHE.get("v")
    if hit and (time.monotonic() - hit[0]) < _VAL_TTL_S:
        return hit[1]
    try:
        from nselib import capital_market as cm

        df = None
        d = date.today()
        for _ in range(6):
            try:
                df = cm.pe_ratio(trade_date=d.strftime("%d-%m-%Y"))
                if df is not None and not df.empty:
                    break
            except Exception:
                pass
            d -= timedelta(days=1)
        if df is None or df.empty:
            return None

        pe: Dict[str, float] = {}
        for _, r in df.iterrows():
            try:
                v = float(r.get("ADJUSTEDP/E") or r.get("SYMBOLP/E") or 0)
            except Exception:
                continue
            if 0 < v < 2000:
                pe[str(r.get("SYMBOL") or "").strip()] = v
        if len(pe) < 100:
            return None

        from ...data.ohlc_store import pg_connect
        conn = pg_connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT index_name, symbol FROM index_constituents "
                    "WHERE index_name IN ('NIFTY 50','NIFTY 500')"
                )
                members: Dict[str, list] = {"NIFTY 50": [], "NIFTY 500": []}
                for idx, sym in cur.fetchall():
                    if sym in pe:
                        members[idx].append(pe[sym])
        finally:
            conn.close()

        def median(vals: list) -> Optional[float]:
            if not vals:
                return None
            s = sorted(vals)
            n = len(s)
            return round(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2, 1)

        allv = list(pe.values())
        out = {
            "as_of": d.isoformat(),
            "nifty50_median_pe": median(members["NIFTY 50"]),
            "nifty500_median_pe": median(members["NIFTY 500"]),
            "market_median_pe": median(allv),
            "pct_above_50x": round(sum(v > 50 for v in allv) / len(allv) * 100, 1),
            "pct_below_15x": round(sum(v < 15 for v in allv) / len(allv) * 100, 1),
            "coverage": len(allv),
            "label": "NSE · EOD published P/E · derived medians",
        }
        _VAL_CACHE["v"] = (time.monotonic(), out)
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug("valuation failed: %s", e)
        return None


def _diff_chips(today: Dict[str, Any], prev: Optional[Dict[str, Any]],
                vol: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The 'what changed vs yesterday' strip — meaningful deltas only."""
    chips: List[Dict[str, Any]] = []
    if prev:
        st, sp = today.get("score"), prev.get("score")
        if st is not None and sp is not None and abs(st - sp) >= 3:
            chips.append({
                "metric": "breadth_score", "delta": st - sp,
                "label": f"Breadth {'improving' if st > sp else 'deteriorating'}",
                "detail": f"score {sp} → {st}",
            })
        p50t, p50p = today.get("pct_above_50dma"), prev.get("pct_above_50dma")
        if p50t is not None and p50p is not None and abs(p50t - p50p) >= 2:
            chips.append({
                "metric": "pct_above_50dma", "delta": round(p50t - p50p, 1),
                "label": f"{'More' if p50t > p50p else 'Fewer'} stocks above 50-DMA",
                "detail": f"{p50p}% → {p50t}%",
            })
        nht = (today.get("new_highs") or 0) - (today.get("new_lows") or 0)
        nhp = (prev.get("new_highs") or 0) - (prev.get("new_lows") or 0)
        if (nht >= 0) != (nhp >= 0):
            chips.append({
                "metric": "net_new_highs", "delta": nht - nhp,
                "label": ("New highs retake the lead" if nht >= 0
                          else "New lows take the lead"),
                "detail": f"net {nhp:+d} → {nht:+d}",
            })
    vix, vix_prev = vol.get("vix"), vol.get("vix_prev")
    if vix is not None and vix_prev is not None and abs(vix - vix_prev) >= 0.5:
        chips.append({
            "metric": "vix", "delta": round(vix - vix_prev, 2),
            "label": f"VIX {'up' if vix > vix_prev else 'down'} "
                     f"{abs(vix - vix_prev):.1f}",
            "detail": f"{vix_prev} → {vix}",
        })

    # Index-vs-breadth DIVERGENCE — the professional warning read: price
    # marching one way while participation walks the other. Deterministic
    # thresholds, fires only when both legs disagree meaningfully.
    ret5 = vol.get("nifty_ret_5d")
    score_t, score_p = today.get("score"), (prev or {}).get("score")
    score_falling = score_t is not None and score_p is not None and (score_t - score_p) <= -3
    score_rising = score_t is not None and score_p is not None and (score_t - score_p) >= 3
    ad = today.get("ad_pct")
    if ret5 is not None:
        if ret5 >= 0.5 and ((ad is not None and ad < 45) or score_falling):
            chips.append({
                "metric": "divergence", "delta": -1,
                "label": "Bearish divergence: index up, breadth lagging",
                "detail": f"NIFTY {ret5:+.1f}% 5d vs breadth "
                          f"{f'{ad}% adv' if ad is not None and ad < 45 else f'score {score_p} → {score_t}'}",
            })
        elif ret5 <= -0.5 and ((ad is not None and ad > 55) or score_rising):
            chips.append({
                "metric": "divergence", "delta": 1,
                "label": "Bullish divergence: index down, breadth holding",
                "detail": f"NIFTY {ret5:+.1f}% 5d vs breadth "
                          f"{f'{ad}% adv' if ad is not None and ad > 55 else f'score {score_p} → {score_t}'}",
            })
    return chips


def market_pulse() -> Dict[str, Any]:
    """Full pulse payload (see module docstring). Cached 10 minutes."""
    hit = _CACHE.get("p")
    if hit and (time.monotonic() - hit[0]) < _TTL_S:
        return hit[1]

    out: Dict[str, Any] = {
        "breadth": None, "vol": None, "flows": None, "positioning": None,
        "delivery": None, "valuation": None, "diff": [], "label": "EOD · derived analytics",
    }
    try:
        today = _snapshot(None)
        out["breadth"] = today
        prev = None
        if today:
            prev_cutoff = date.fromisoformat(today["as_of"]) - timedelta(days=1)
            prev = _snapshot(prev_cutoff)
        vol = _nifty_vol()
        out["vol"] = vol
        out["flows"] = _flow_streaks()
        out["positioning"] = _fii_positioning()
        try:
            out["delivery"] = _delivery_intel()
        except Exception as e:  # noqa: BLE001
            logger.debug("delivery intel failed: %s", e)
        out["valuation"] = _valuation()
        if today:
            out["diff"] = _diff_chips(today, prev, vol)
        if today:
            _CACHE["p"] = (time.monotonic(), out)
    except Exception as e:
        logger.warning("market_pulse failed: %s", e)
    return out
