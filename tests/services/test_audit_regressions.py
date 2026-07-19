"""Regression tests for the 2026-06-04 full-app audit fixes.

Each test pins one bug the audit found — the kind that slipped through because
the unit tests bypassed the route/schema seam. Behavioral where feasible;
source-guard (like the existing forbidden-import guards) where the seam needs a
live route/DB. See docs/AUDIT_REPORT_2026_06_04.md.
"""
from __future__ import annotations

import pathlib

import pytest


def _src(mod) -> str:
    return pathlib.Path(mod.__file__).read_text()


# ── Batch 1 ────────────────────────────────────────────────────────────────

def test_ai_performance_does_not_fabricate_winrate():
    """/api/ai/performance must never return the old hardcoded 67.2 / 56.1."""
    import backend.api.market_routes as mr
    s = _src(mr)
    assert "67.2" not in s and "56.1" not in s
    assert "insufficient_data" in s


def test_copilot_limiter_method_exists():
    """The cap was a no-op because it called consume() (doesn't exist)."""
    from backend.services.assistant.credit_limiter import AssistantCreditLimiter
    assert hasattr(AssistantCreditLimiter, "consume_if_available")
    assert not hasattr(AssistantCreditLimiter, "consume")
    import backend.api.ai_routes as ai
    assert "consume_if_available" in _src(ai)


def test_autopilot_enrolment_column():
    """Rebalancer/supervisor must read auto_trader_enabled (the toggle's column),
    not autopilot_enabled, and must not select the non-existent subscription_tier."""
    import backend.trading.autopilot_service as a
    import backend.services.autopilot.supervisor as sup
    asrc, ssrc = _src(a), _src(sup)
    assert '.eq("auto_trader_enabled", True)' in asrc
    assert '.eq("autopilot_enabled", True)' not in asrc
    assert "subscription_tier" not in ssrc
    assert "..services.market_regime" not in asrc  # the dead regime import


def test_fno_strategy_leg_uses_action_not_side():
    """The F&O Deploy 500: route read StrategyLeg.side/.qty_lots; the dataclass
    exposes .action and no qty_lots."""
    import dataclasses
    from backend.ai.fo.strategies import StrategyLeg
    from backend.ai.strategy.dsl import OptionSide
    fields = {f.name for f in dataclasses.fields(StrategyLeg)}
    assert "action" in fields and "side" not in fields and "qty_lots" not in fields
    for action in ("BUY", "SELL"):
        assert OptionSide(action.lower()) in (OptionSide.BUY, OptionSide.SELL)


# ── Batch 2 ────────────────────────────────────────────────────────────────

def test_marketplace_tier_order_includes_elite():
    import backend.api.marketplace_routes as mp
    s = _src(mp)
    assert "tier_order" in s and '"elite"' in s


def test_confluence_has_no_pattern_scanners_import():
    import backend.services.screener_v2.confluence as c
    s = _src(c)
    assert "import SCANNER_MENU, PATTERN_SCANNERS" not in s  # deleted 2026-05-31; was a hard 500
    assert "in PATTERN_SCANNERS" not in s


# ── Batch 3 ────────────────────────────────────────────────────────────────

def test_digest_uses_correct_position_columns_and_conf():
    import backend.ai.digest.generator as dg
    s = _src(dg)
    assert "int(conf * 100)" not in s          # was "(7500% conf)"
    assert '.eq("status", "open")' not in s     # positions has is_active
    assert "unrealized_pnl_percent" in s        # not _percentage


def test_weekly_review_uses_detected_at():
    import backend.ai.weekly_review.generator as wg
    s = _src(wg)
    assert "as_of" not in s                      # regime_history col is detected_at


def test_digest_whatsapp_import_path():
    import backend.ai.digest.delivery as d
    s = _src(d)
    assert "from ...services import whatsapp_service" not in s
    assert "platform import whatsapp" in s


# ── Batch 4 ────────────────────────────────────────────────────────────────

def test_compute_greeks_keyword_only_and_nonzero():
    """MTM Greeks were always 0: compute_greeks() is keyword-only but was called
    positionally → TypeError → swallowed."""
    from backend.services.execution.options_greeks import compute_greeks
    with pytest.raises(TypeError):
        compute_greeks(22000, 22000, 0.05, 0.065, 0.14, True)  # type: ignore[misc]
    g = compute_greeks(S=22000.0, K=22000.0, T=0.05, sigma=0.14, is_call=True)
    assert abs(g.delta) > 0.1  # ATM call delta ~0.5, not 0


def test_paper_mtm_greeks_call_is_keyword():
    import backend.services.execution.paper_options_executor as p
    s = _src(p)
    assert "compute_greeks(\n                S=" in s or "compute_greeks(S=" in s


# ── Batch 6: PR-S18 institutional scanners revived ──────────────────────────
#
# 12 institutional scanners (72-86) silently returned empty because
# extract_summary_row dropped the precomputed PR-S18 columns, so every
# `if "ema_10" not in df.columns: return empty` guard tripped. These tests
# pin the producer/consumer contract through the REAL indicator pipeline —
# the exact route/schema seam the unit tests bypassed.

def _synth_ohlcv(n: int = 220):
    import numpy as np
    import pandas as pd
    # Deterministic gently-uptrending series with real intrabar range + volume.
    t = np.arange(n, dtype=float)
    close = 100.0 + t * 0.15 + 4.0 * np.sin(t / 9.0)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 1.0 + 0.5 * np.abs(np.sin(t / 3.0))
    low = np.minimum(open_, close) - 1.0 - 0.5 * np.abs(np.cos(t / 4.0))
    vol = 1_000_000.0 + (t % 7) * 50_000.0 + 200_000.0 * (np.sin(t / 5.0) > 0)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _scanner_guard_columns() -> set:
    """Columns each scanner early-returns on when absent (the silent-empty trap)."""
    import re
    import backend.data.screener.filters as f
    return set(re.findall(r'"([a-z_0-9]+)" not in df\.columns', _src(f)))


def test_extract_summary_row_supplies_every_scanner_guard_column():
    """Producer/consumer contract: every column a scanner guards on MUST be
    produced by extract_summary_row, else that scanner silently returns
    nothing. This is the bug that killed the 12 PR-S18 institutional scanners."""
    from ml.features.indicators import compute_all_indicators
    from backend.data.screener.sources import extract_summary_row
    df = compute_all_indicators(_synth_ohlcv())
    row = extract_summary_row("TEST", df, df.iloc[-1])
    assert row is not None
    missing = _scanner_guard_columns() - set(row.keys())
    assert not missing, f"summary row missing scanner guard columns: {sorted(missing)}"


def test_pr_s18_institutional_scanners_run_without_keyerror():
    """Each registered institutional scanner (72-86) must execute against a
    real summary_df without raising. Before the fix, a passed guard hit a
    body-referenced column not forwarded (pocket_pivot_volume / cpr_bc /
    sma_150_rising) → KeyError 500 (or empty via the dropped guard column)."""
    import pandas as pd
    from ml.features.indicators import compute_all_indicators
    from backend.data.screener.sources import extract_summary_row
    from backend.data.screener.filters import SCANNER_FILTERS
    df = compute_all_indicators(_synth_ohlcv())
    summary_df = pd.DataFrame([extract_summary_row("TEST", df, df.iloc[-1])])
    for sid in range(72, 87):  # 72-86; 87/88 need live NSE OI data
        fn = SCANNER_FILTERS.get(sid)
        if fn is None:
            continue
        out = fn(summary_df.copy())
        assert isinstance(out, pd.DataFrame), f"scanner {sid} did not return a DataFrame"


# ── Batch 7: no-fallback violations (no synthetic market data) ──────────────

def test_market_provider_returns_empty_not_synthetic(monkeypatch):
    """MarketDataProvider.get_option_chain must return [] when the live Kite
    chain is unavailable — NOT a fabricated Black-Scholes synthetic chain
    (no-fallbacks lock). The _synthetic_option_chain generator is removed."""
    from backend.data.market import get_market_data_provider
    mp = get_market_data_provider()
    assert not hasattr(mp, "_synthetic_option_chain"), "synthetic chain generator must be gone"

    def _boom():
        raise RuntimeError("no kite")
    monkeypatch.setattr(mp, "_get_kite_provider", _boom)
    assert mp.get_option_chain("NIFTY") == []   # honest-empty, not invented


def test_scheduler_market_data_has_no_simulated_fallback():
    """_fetch_market_data must not return hardcoded simulated Nifty/VIX on
    failure; market_open_check must skip when data is unavailable."""
    import backend.platform.scheduler as sch
    s = _src(sch)
    assert "Fallback to simulated" not in s
    assert "21800" not in s and "21850" not in s          # old fake Nifty values
    assert "market data unavailable — skipping" in s       # the honest skip guard
