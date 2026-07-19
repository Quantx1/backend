"""F&O index option-chain snapshot.

Fetches the live option chain for a target index (NIFTY / BANKNIFTY /
FINNIFTY / MIDCPNIFTY) via the admin Kite provider, computes:
  - Max Pain (Zerodha Varsity formula — strike that minimises writer loss)
  - PCR (OI-weighted, volume-weighted)
  - ATM IV + IV skew (PE_OTM - CE_OTM)
  - Top 3 OI buildup strikes (call-side = resistance, put-side = support)
  - Pull-to-MaxPain signal (gap % from spot to Max Pain)

All formulas verified against:
  - Zerodha Varsity Ch. on Max Pain & PCR:
      https://zerodha.com/varsity/chapter/max-pain-pcr-ratio/
  - StockMojo Nifty Max Pain methodology
  - PL Capital "Option Chain Analysis" blog
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# In-process TTL cache for snapshots. Option premiums tick fast, but
# Max Pain / PCR move on aggregate OI which updates slowly — 30s is
# generous enough to hide the latency without showing stale state.
_SNAPSHOT_TTL_S = 30.0
_snapshot_cache: Dict[str, tuple] = {}


@dataclass
class StrikeOI:
    strike: float
    call_oi: int = 0
    put_oi: int = 0
    call_oi_change: int = 0
    put_oi_change: int = 0
    call_iv: float = 0.0
    put_iv: float = 0.0


@dataclass
class IndexSnapshot:
    """One snapshot of an index's F&O dashboard."""

    symbol: str                              # NIFTY, BANKNIFTY, FINNIFTY
    spot: float                              # underlying spot
    expiry: Optional[str]                    # ISO date of the chain queried
    days_to_expiry: Optional[int]

    # Aggregates
    pcr_oi: float                            # PE_OI / CE_OI
    pcr_volume: Optional[float] = None
    total_call_oi: int = 0
    total_put_oi: int = 0

    # Max Pain
    max_pain: Optional[float] = None
    max_pain_distance_pct: Optional[float] = None  # (spot - MP) / spot * 100

    # IV
    iv_atm: Optional[float] = None           # mean of 3 ATM strikes
    iv_skew: Optional[float] = None          # PE OTM IV - CE OTM IV

    # Historical / realized volatility of the underlying (annualized %, by window)
    hv: Optional[Dict[str, Any]] = None      # {"hv": {"10":..,"20":..,"30":..}, "latest_hv": .., "note": ..}

    # Top-3 OI strikes (resistance / support)
    top_call_oi_strikes: List[float] = field(default_factory=list)
    top_put_oi_strikes: List[float] = field(default_factory=list)

    # Highest single-strike OI delta today (institutional fingerprint)
    biggest_oi_buildup: Optional[Dict[str, Any]] = None

    # Tags / signals
    pcr_tag: str = "normal"                  # extreme_bullish | bullish | normal | bearish | extreme_bearish
    pull_to_max_pain_signal: bool = False    # |spot - MP| > 1.5% and >2 DTE

    # Diagnostics
    source: str = "kite"
    strike_count: int = 0
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "spot": round(self.spot, 2),
            "expiry": self.expiry,
            "days_to_expiry": self.days_to_expiry,
            "pcr_oi": round(self.pcr_oi, 3),
            "pcr_volume": round(self.pcr_volume, 3) if self.pcr_volume is not None else None,
            "pcr_tag": self.pcr_tag,
            "total_call_oi": self.total_call_oi,
            "total_put_oi": self.total_put_oi,
            "max_pain": round(self.max_pain, 2) if self.max_pain is not None else None,
            "max_pain_distance_pct": (
                round(self.max_pain_distance_pct, 3) if self.max_pain_distance_pct is not None else None
            ),
            "pull_to_max_pain_signal": self.pull_to_max_pain_signal,
            "iv_atm": round(self.iv_atm, 4) if self.iv_atm is not None else None,
            "iv_skew": round(self.iv_skew, 4) if self.iv_skew is not None else None,
            "hv": self.hv,
            "top_call_oi_strikes": [round(s, 1) for s in self.top_call_oi_strikes],
            "top_put_oi_strikes": [round(s, 1) for s in self.top_put_oi_strikes],
            "biggest_oi_buildup": self.biggest_oi_buildup,
            "source": self.source,
            "strike_count": self.strike_count,
            "timestamp": self.timestamp,
        }


def teach_snapshot(d: Dict[str, Any]) -> List[str]:
    """Deterministic plain-English read of an option-chain snapshot — the
    'Options Teacher'. ZERO LLM tokens: every line is derived from the real
    snapshot numbers, so it can't hallucinate and costs nothing to serve."""
    out: List[str] = []
    spot = d.get("spot")
    pcr = d.get("pcr_oi")
    if pcr is not None:
        if pcr >= 1.3:
            lean = "heavy put-writing — usually a bullish / oversold tilt"
        elif pcr >= 1.0:
            lean = "more open puts than calls — a mild bullish lean"
        elif pcr >= 0.7:
            lean = "more open calls than puts — a mild bearish / cautious lean"
        else:
            lean = "heavy call-writing — usually a bearish / overbought tilt"
        out.append(f"PCR (OI) is {pcr:.2f}: {lean}.")
    mp = d.get("max_pain")
    if mp is not None:
        line = (f"Max pain is {mp:g} — the strike where option writers lose the "
                "least, so price often drifts toward it into expiry.")
        dist = d.get("max_pain_distance_pct")
        if dist is not None and spot:
            rel = "above" if spot > mp else "below"
            line += f" Spot is {abs(dist):.1f}% {rel} it."
        out.append(line)
    tc = d.get("top_call_oi_strikes") or []
    if tc:
        out.append("Heaviest call OI at " + ", ".join(f"{s:g}" for s in tc[:2])
                   + " — call writers defend these, so they tend to act as resistance.")
    tp = d.get("top_put_oi_strikes") or []
    if tp:
        out.append("Heaviest put OI at " + ", ".join(f"{s:g}" for s in tp[:2])
                   + " — put writers defend these, so they tend to act as support.")
    b = d.get("biggest_oi_buildup") or {}
    if b.get("strike") and b.get("direction"):
        side = str(b.get("side") or "").upper()
        meaning = {
            "writing": "fresh writing (sellers adding) — that level is being capped",
            "unwinding": "unwinding (sellers covering) — that level is loosening",
        }.get(b.get("direction"), str(b.get("direction")))
        out.append(f"Biggest OI move: {side} {meaning} at {b['strike']:g}.")
    iv = d.get("iv_atm")
    if iv is not None:
        out.append(f"ATM IV is {iv * 100:.1f}% — higher IV = pricier options "
                   "(favours sellers), lower IV favours buyers.")
        hv_blob = d.get("hv") or {}
        latest_hv = hv_blob.get("latest_hv") if isinstance(hv_blob, dict) else None
        if latest_hv is not None:
            iv_pct = iv * 100
            diff = iv_pct - latest_hv
            if diff > 1.0:
                lean = "options are pricing a premium over realized movement — favours writers (sell premium)"
            elif diff < -1.0:
                lean = "options are cheap vs realized movement — favours buyers (long premium)"
            else:
                lean = "in line with realized movement — no strong vol edge"
            out.append(f"IV {iv_pct:.1f}% vs HV(20) {latest_hv:.1f}% — {lean}.")
    ivr = d.get("iv_rank")
    if ivr is not None:
        if ivr >= 70:
            lean = "high — options are rich, premium-selling (spreads/condors) is favoured"
        elif ivr <= 30:
            lean = "low — options are cheap, premium-buying (debit spreads/long options) is favoured"
        else:
            lean = "mid-range — no strong vol edge either way"
        out.append(f"IV Rank is {ivr:.0f}: {lean}.")
    dte = d.get("days_to_expiry")
    if isinstance(dte, int) and dte <= 2:
        out.append(f"Only {dte} day(s) to expiry — time decay accelerates and "
                   "pin risk near max pain rises.")
    return out


# ── Max Pain math ───────────────────────────────────────────────────


def _compute_max_pain(by_strike: Dict[float, StrikeOI], strikes: List[float]) -> Optional[float]:
    """Zerodha Varsity definition: for each candidate expiry-price K*,
    sum option-writer losses across all strikes, then pick the K* that
    minimises that total. Writers' loss at expiry-price S:
        Σ max(S - K_call, 0) × CE_OI(K) + Σ max(K_put - S, 0) × PE_OI(K)
    """
    if not strikes:
        return None
    best_strike = None
    best_loss = None
    for candidate_s in strikes:
        total_loss = 0.0
        for k in strikes:
            strike_oi = by_strike[k]
            if candidate_s > k:
                total_loss += (candidate_s - k) * strike_oi.call_oi
            if candidate_s < k:
                total_loss += (k - candidate_s) * strike_oi.put_oi
        if best_loss is None or total_loss < best_loss:
            best_loss = total_loss
            best_strike = candidate_s
    return best_strike


# ── PCR classification ──────────────────────────────────────────────


def _classify_pcr(pcr: float, symbol: str) -> str:
    """Zerodha Varsity cutoffs (Nifty): >1.3 = contrarian bullish (panic
    put-buying), <0.5 = contrarian bearish (call-buying euphoria).
    BankNifty PCR runs structurally lower (heavier institutional put-writing)
    so we shift the bands -0.1 per the IIFL/India-Infoline calibration.
    """
    shift = -0.10 if symbol.upper() in ("BANKNIFTY", "BANKEX") else 0.0
    if pcr >= 1.4 + shift:
        return "extreme_bullish"          # contrarian — extreme fear = floor
    if pcr >= 1.0 + shift:
        return "bullish"
    if pcr <= 0.4 + shift:
        return "extreme_bearish"          # contrarian — euphoria = ceiling
    if pcr <= 0.7 + shift:
        return "bearish"
    return "normal"


# ── Main entrypoint ─────────────────────────────────────────────────


def fetch_index_snapshot(symbol: str, expiry: Optional[str] = None) -> Optional[IndexSnapshot]:
    """Fetch a fresh F&O snapshot for one of the index symbols.

    Returns None if the option chain provider is unavailable. Public-safe:
    uses the admin Kite provider (not per-user broker creds).
    """
    sym = symbol.upper().strip()
    cache_key = f"{sym}:{expiry or 'near'}"
    now = time.monotonic()
    hit = _snapshot_cache.get(cache_key)
    if hit and now - hit[0] < _SNAPSHOT_TTL_S:
        return hit[1]

    # Lazy import — keep the module importable even if market.py is broken
    try:
        from ...data.market import get_market_data_provider
    except Exception as e:
        logger.warning("fno_scanner: market provider import failed: %s", e)
        return None

    try:
        mp = get_market_data_provider()
        chain = mp.get_option_chain(sym, expiry or "")
    except Exception as e:
        logger.warning("fno_scanner: get_option_chain(%s) failed: %s", sym, e)
        return None

    if not chain:
        return None

    # Build a per-strike aggregation
    by_strike: Dict[float, StrikeOI] = {}
    total_call_vol = 0
    total_put_vol = 0
    expiry_seen: Optional[str] = None

    for row in chain:
        try:
            strike = float(row.get("strike", 0) or 0)
            if strike <= 0:
                continue
            otype = str(row.get("option_type", "")).upper()
            oi = int(row.get("oi", 0) or 0)
            oi_change = int(row.get("oi_change", 0) or 0)
            volume = int(row.get("volume", 0) or 0)
            iv = float(row.get("iv", 0) or 0)
            exp = str(row.get("expiry", "") or "")
            if expiry_seen is None and exp:
                expiry_seen = exp
            entry = by_strike.setdefault(strike, StrikeOI(strike=strike))
            if otype == "CE":
                entry.call_oi += oi
                entry.call_oi_change += oi_change
                entry.call_iv = iv if entry.call_iv == 0 else (entry.call_iv + iv) / 2
                total_call_vol += volume
            elif otype == "PE":
                entry.put_oi += oi
                entry.put_oi_change += oi_change
                entry.put_iv = iv if entry.put_iv == 0 else (entry.put_iv + iv) / 2
                total_put_vol += volume
        except Exception:
            continue

    if not by_strike:
        return None

    strikes = sorted(by_strike.keys())
    total_ce_oi = sum(v.call_oi for v in by_strike.values())
    total_pe_oi = sum(v.put_oi for v in by_strike.values())

    # Spot — best-effort via separate quote, else infer from ATM strike density
    spot = 0.0
    try:
        quote = mp.get_quote(sym)
        spot = float(quote.ltp) if quote and quote.ltp else 0.0
    except Exception:
        pass
    if spot <= 0:
        # Fallback: median strike (close enough to spot for option chains)
        spot = float(strikes[len(strikes) // 2])

    pcr_oi = (total_pe_oi / total_ce_oi) if total_ce_oi > 0 else 1.0
    pcr_vol = (total_put_vol / total_call_vol) if total_call_vol > 0 else None
    pcr_tag = _classify_pcr(pcr_oi, sym)

    max_pain = _compute_max_pain(by_strike, strikes)
    mp_dist = (
        ((spot - max_pain) / spot * 100.0) if (max_pain is not None and spot > 0) else None
    )

    # Top 3 OI strikes — call OI = resistance, put OI = support
    sorted_calls = sorted(by_strike.values(), key=lambda v: v.call_oi, reverse=True)
    sorted_puts = sorted(by_strike.values(), key=lambda v: v.put_oi, reverse=True)
    top_calls = [v.strike for v in sorted_calls[:3]]
    top_puts = [v.strike for v in sorted_puts[:3]]

    # Biggest single-strike OI buildup today (Δ OI)
    biggest = None
    biggest_delta = 0
    for v in by_strike.values():
        for side, delta in (("CE", v.call_oi_change), ("PE", v.put_oi_change)):
            if abs(delta) > biggest_delta:
                biggest_delta = abs(delta)
                biggest = {
                    "strike": v.strike,
                    "side": side,
                    "oi_change": delta,
                    "direction": (
                        "writing" if delta > 0 else "unwinding"
                    ),
                }

    # ATM IV — mean of 3 ATM strikes' CE+PE IV
    iv_atm = None
    if spot > 0:
        atm_sorted = sorted(by_strike.values(), key=lambda v: abs(v.strike - spot))[:3]
        ivs = [v.call_iv for v in atm_sorted if v.call_iv > 0] + \
              [v.put_iv for v in atm_sorted if v.put_iv > 0]
        if ivs:
            iv_atm = sum(ivs) / len(ivs)

    # IV skew — PE OTM avg - CE OTM avg
    iv_skew = None
    if spot > 0:
        pe_otm_iv = [v.put_iv for v in by_strike.values()
                     if v.strike < spot * 0.99 and v.put_iv > 0]
        ce_otm_iv = [v.call_iv for v in by_strike.values()
                     if v.strike > spot * 1.01 and v.call_iv > 0]
        if pe_otm_iv and ce_otm_iv:
            iv_skew = sum(pe_otm_iv) / len(pe_otm_iv) - sum(ce_otm_iv) / len(ce_otm_iv)

    # Days to expiry
    dte = None
    if expiry_seen:
        try:
            from datetime import date, datetime
            d = datetime.strptime(expiry_seen[:10], "%Y-%m-%d").date()
            dte = max(0, (d - date.today()).days)
        except Exception:
            pass

    # Pull-to-MP signal: >1.5% gap with at least 2 days left
    pull_signal = bool(
        mp_dist is not None and abs(mp_dist) > 1.5 and dte is not None and dte >= 2
    )

    from datetime import datetime

    # Historical/realized volatility of the underlying (annualized %, 0 LLM tokens).
    # Honest-empty: stays None when there aren't enough daily closes.
    hv_data = None
    try:
        from .volatility import realized_vol
        hv_data = realized_vol(sym)
    except Exception as e:
        logger.debug("fno_scanner: realized_vol(%s) failed: %s", sym, e)

    snap = IndexSnapshot(
        symbol=sym,
        spot=spot,
        expiry=expiry_seen,
        days_to_expiry=dte,
        pcr_oi=pcr_oi,
        pcr_volume=pcr_vol,
        pcr_tag=pcr_tag,
        total_call_oi=total_ce_oi,
        total_put_oi=total_pe_oi,
        max_pain=max_pain,
        max_pain_distance_pct=mp_dist,
        pull_to_max_pain_signal=pull_signal,
        iv_atm=iv_atm,
        iv_skew=iv_skew,
        hv=hv_data,
        top_call_oi_strikes=top_calls,
        top_put_oi_strikes=top_puts,
        biggest_oi_buildup=biggest,
        source="kite",
        strike_count=len(strikes),
        timestamp=datetime.now().isoformat(),
    )
    _snapshot_cache[cache_key] = (now, snap)
    if len(_snapshot_cache) > 32:
        oldest = min(_snapshot_cache.items(), key=lambda kv: kv[1][0])[0]
        _snapshot_cache.pop(oldest, None)
    return snap
