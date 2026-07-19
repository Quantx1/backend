"""NSE reference mappers — pure-function tests (no network).

Covers the index-constituent parser + sector/mcap derivation that drive the
full-universe ingestion (scripts/data/ingest_nse_universe.py).
"""
import pandas as pd

from backend.data.reference.nse_reference import (
    INDEX_CSV_MAP,
    FNO_INDEX_NAME,
    build_mcap_map,
    build_sector_map,
    map_equity_master_rows,
    map_index_constituent_rows,
)


def _idx_df():
    # NSE index CSV shape: Company Name, Industry, Symbol, Series, ISIN Code
    return pd.DataFrame([
        {"Company Name": "Reliance Industries Ltd", "Industry": "Oil Gas & Consumable Fuels",
         "Symbol": "RELIANCE", "Series": "EQ", "ISIN Code": "INE002A01018"},
        {"Company Name": "HDFC Bank Ltd", "Industry": "Financial Services",
         "Symbol": "hdfcbank", "Series": "EQ", "ISIN Code": "INE040A01034"},
    ])


def test_map_index_constituent_rows_basic():
    rows = map_index_constituent_rows(_idx_df(), "NIFTY 50")
    assert len(rows) == 2
    assert rows[0]["index_name"] == "NIFTY 50"
    assert rows[0]["symbol"] == "RELIANCE"
    assert rows[0]["industry"] == "Oil Gas & Consumable Fuels"
    assert rows[0]["source"] == "nseindices"
    # symbol upper-cased
    assert rows[1]["symbol"] == "HDFCBANK"


def test_map_index_constituent_rows_skips_blank_and_empty_df():
    df = pd.DataFrame([
        {"Company Name": "x", "Industry": "IT", "Symbol": "", "ISIN Code": "z"},
        {"Company Name": "y", "Industry": "IT", "Symbol": "None", "ISIN Code": "z"},
    ])
    assert map_index_constituent_rows(df, "NIFTY IT") == []
    assert map_index_constituent_rows(pd.DataFrame(), "NIFTY IT") == []


def test_build_sector_map_prefers_authoritative_index():
    # Same symbol classified by a low-priority index AND NIFTY TOTAL MARKET —
    # the authoritative source must win.
    cbi = {
        "NIFTY AUTO": [{"symbol": "TVSMOTOR", "industry": "WRONG"}],
        "NIFTY TOTAL MARKET": [{"symbol": "TVSMOTOR", "industry": "Automobile and Auto Components"}],
    }
    sector = build_sector_map(cbi)
    assert sector["TVSMOTOR"] == "Automobile and Auto Components"


def test_build_mcap_map_most_inclusive_tier_wins():
    cbi = {
        "NIFTY 100": [{"symbol": "RELIANCE", "industry": "x"}],
        "NIFTY MIDCAP 150": [{"symbol": "RELIANCE", "industry": "x"},
                             {"symbol": "POLYCAB", "industry": "y"}],
        "NIFTY SMALLCAP 250": [{"symbol": "CUPID", "industry": "z"}],
    }
    mcap = build_mcap_map(cbi)
    assert mcap["RELIANCE"] == "Large Cap"   # in 100 -> large (wins over midcap)
    assert mcap["POLYCAB"] == "Mid Cap"
    assert mcap["CUPID"] == "Small Cap"


def test_index_map_catalog_shape():
    # Curated map: every value is (csv_filename, category) with a known category.
    assert "NIFTY 50" in INDEX_CSV_MAP and "NIFTY BANK" in INDEX_CSV_MAP
    cats = {cat for _fn, cat in INDEX_CSV_MAP.values()}
    assert cats == {"broad", "sectoral"}
    assert FNO_INDEX_NAME == "F&O STOCKS"


def test_map_equity_master_rows_tolerates_spaced_columns():
    # EQUITY_L columns carry leading spaces (' SERIES', ' FACE VALUE', ...).
    df = pd.DataFrame([
        {"SYMBOL": "TCS", "NAME OF COMPANY": "Tata Consultancy Services Limited",
         " SERIES": "EQ", " DATE OF LISTING": "25-AUG-2004", " PAID UP VALUE": 1,
         " MARKET LOT": 1, " ISIN NUMBER": "INE467B01029", " FACE VALUE": 1},
    ])
    rows = map_equity_master_rows(df)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "TCS"
    assert rows[0]["instrument_type"] == "EQ"
    assert rows[0]["isin"] == "INE467B01029"
    assert rows[0]["series"] == "EQ"
