"""NSE delivery % + bulk-deal data sources (scanners 34 / 35).

Before this, ``_filter_high_delivery`` (34) and ``_filter_bulk_deals`` (35)
called ``nse.get_delivery_data()`` / ``nse.get_bulk_deals()`` — methods that
did not exist on ``NSEDataProvider`` → AttributeError → swallowed by the
scanner's broad try/except → silent empty. These tests pin the two new methods
through the REAL parse path (faithful NSE CSV format, including the leading
spaces NSE puts on sec_bhavdata_full columns AND values) and the exact
producer→consumer contract the two scanners depend on.

jugaad-data's network calls are mocked so the tests are deterministic offline.
"""
from __future__ import annotations

import pandas as pd
import pytest

import backend.data.screener.nse_data as nd


# Faithful sec_bhavdata_full slice — NSE prefixes every column AND every value
# after SYMBOL with a leading space; non-EQ series (GS) and '-' DELIV_PER rows
# must be dropped, not coerced to 0.
DELIVERY_CSV = (
    "SYMBOL, SERIES, DELIV_QTY, DELIV_PER\n"
    "RELIANCE, EQ, 1000000, 65.50\n"
    "TCS, EQ, 500000, 45.20\n"
    "NIFTYBEES, GS, 100, 99.85\n"
    "JUNKSTOCK, EQ, 0, -\n"
)

# Faithful /content/equities/bulk.csv header (no leading spaces here).
BULK_CSV = (
    "Date,Symbol,Security Name,Client Name,Buy/Sell,Quantity Traded,"
    "Trade Price / Wght. Avg. Price,Remarks\n"
    "05-Jun-2026,APOLLO,Apollo Ltd,SOME FUND,BUY,100000,1500.50,-\n"
    "05-Jun-2026,ATALREAL,Atal Realty,OTHER FUND,SELL,50000,250.25,-\n"
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """The provider caches by key at module scope — clear between tests."""
    nd._cache.clear()
    yield
    nd._cache.clear()


# ── coercion helpers ────────────────────────────────────────────────────────

def test_coerce_handles_separators_and_dash():
    assert nd._coerce_int("1,00,000") == 100000
    assert nd._coerce_int(" 50000 ") == 50000
    assert nd._coerce_int("-") is None
    assert nd._coerce_float("1500.50") == 1500.50
    assert nd._coerce_float("-") is None


# ── delivery % (scanner 34) ─────────────────────────────────────────────────

def test_get_delivery_data_parses_real_format(monkeypatch):
    monkeypatch.setattr("jugaad_data.nse.full_bhavcopy_raw", lambda d: DELIVERY_CSV)
    df = nd.NSEDataProvider().get_delivery_data()

    assert list(df.columns) == ["symbol", "delivery_pct"]
    syms = set(df["symbol"])
    assert syms == {"RELIANCE", "TCS"}          # GS series + '-' row dropped
    assert "NIFTYBEES" not in syms              # non-EQ filtered
    assert "JUNKSTOCK" not in syms              # DELIV_PER '-' dropped, not 0
    rel = df.loc[df["symbol"] == "RELIANCE", "delivery_pct"].iloc[0]
    assert rel == pytest.approx(65.50)          # leading-space value coerced


def test_get_delivery_data_empty_on_source_failure(monkeypatch):
    def _boom(d):
        raise RuntimeError("NSE blocked")
    monkeypatch.setattr("jugaad_data.nse.full_bhavcopy_raw", _boom)
    df = nd.NSEDataProvider().get_delivery_data()
    assert df.empty
    assert list(df.columns) == ["symbol", "delivery_pct"]  # honest-empty, no synthetic


# ── bulk deals (scanner 35) ─────────────────────────────────────────────────

def test_get_bulk_deals_parses_real_format(monkeypatch):
    monkeypatch.setattr(
        "jugaad_data.nse.NSEArchives.bulk_deals_raw", lambda self: BULK_CSV
    )
    deals = nd.NSEDataProvider().get_bulk_deals()
    assert isinstance(deals, list) and len(deals) == 2
    by_sym = {d["symbol"]: d for d in deals}
    assert set(by_sym) == {"APOLLO", "ATALREAL"}
    assert by_sym["APOLLO"]["side"] == "BUY"
    assert by_sym["APOLLO"]["qty"] == 100000
    assert by_sym["APOLLO"]["price"] == pytest.approx(1500.50)


def test_get_bulk_deals_empty_on_source_failure(monkeypatch):
    def _boom(self):
        raise RuntimeError("NSE blocked")
    monkeypatch.setattr("jugaad_data.nse.NSEArchives.bulk_deals_raw", _boom)
    assert nd.NSEDataProvider().get_bulk_deals() == []


# ── producer→consumer contract: the scanners actually run ───────────────────

def _out_symbols(out: pd.DataFrame) -> set:
    """Scanner results carry symbol in a 'symbol' column (production fallback
    path) or the index (merged path) — accept either."""
    return set(out["symbol"]) if "symbol" in out.columns else set(out.index)


def test_scanner_34_consumes_delivery_contract(monkeypatch):
    """Scanner 34 must run end-to-end against the real provider output and
    surface the >50% delivery name. This is the seam that AttributeError'd.

    Production shape: summary_df has a 'symbol' COLUMN + RangeIndex (engine.py
    builds pd.DataFrame(summary_rows)), so the scanner falls through to its
    high_del fallback and returns the >50% delivery names."""
    monkeypatch.setattr("jugaad_data.nse.full_bhavcopy_raw", lambda d: DELIVERY_CSV)
    from backend.data.screener.filters import SCANNER_FILTERS

    summary_df = pd.DataFrame({
        "symbol": ["RELIANCE", "TCS"],
        "volume_ratio": [1.0, 1.0],
        "close": [2900.0, 3800.0],
    })
    out = SCANNER_FILTERS[34](summary_df.copy())
    assert isinstance(out, pd.DataFrame)
    syms = _out_symbols(out)
    assert "RELIANCE" in syms          # 65.5% > 50 threshold
    assert "TCS" not in syms           # 45.2% filtered out below threshold


def test_scanner_35_consumes_bulk_contract(monkeypatch):
    """Scanner 35 surfaces today's bulk-deal names. Production-shape summary_df
    (symbol column + RangeIndex)."""
    monkeypatch.setattr(
        "jugaad_data.nse.NSEArchives.bulk_deals_raw", lambda self: BULK_CSV
    )
    from backend.data.screener.filters import SCANNER_FILTERS

    summary_df = pd.DataFrame({
        "symbol": ["APOLLO", "UNRELATED"],
        "volume_ratio": [2.0, 1.0],
        "close": [1500.0, 250.0],
    })
    out = SCANNER_FILTERS[35](summary_df.copy())
    assert isinstance(out, pd.DataFrame)
    syms = _out_symbols(out)
    assert "APOLLO" in syms             # had a bulk deal today
    assert "UNRELATED" not in syms      # no bulk deal → not surfaced


def test_oi_scanners_match_production_shape(monkeypatch):
    """OI scanners 39-42 used bare df.index.isin → silent-empty in the live
    (symbol-column) shape, even with real OI data. Pin the fix on scanner 40
    (Long Buildup): price-up + OI-up names must surface."""
    monkeypatch.setattr(
        nd.NSEDataProvider,
        "get_participant_oi",
        lambda self: {"data": [
            {"symbol": "TATASTEEL", "change_pct": 1.2, "oi_change_pct": 8.0},
            {"symbol": "WIPRO", "change_pct": -0.2, "oi_change_pct": 9.0},
        ], "source": "test"},
    )
    from backend.data.screener.filters import SCANNER_FILTERS

    summary_df = pd.DataFrame({
        "symbol": ["TATASTEEL", "WIPRO"],
        "change_pct": [1.2, -0.2],
        "volume_ratio": [1.5, 1.0],
    })
    out = SCANNER_FILTERS[40](summary_df.copy())
    syms = _out_symbols(out)
    assert "TATASTEEL" in syms          # price up + OI up = long buildup
    assert "WIPRO" not in syms          # price down → not a long buildup
