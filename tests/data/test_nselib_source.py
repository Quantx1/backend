import pandas as pd
from backend.data.providers.nselib_source import normalize_bhavcopy_rows


def test_normalize_bhavcopy_rows_to_candle_rows():
    df = pd.DataFrame([
        {"Symbol": "RELIANCE", "Date": "2026-06-06", "OpenPrice": 100, "HighPrice": 102,
         "LowPrice": 99, "ClosePrice": 101, "TotalTradedQuantity": 12345,
         "DeliverableQty": 6000, "%DlyQttoTradedQty": 48.6},
    ])
    rows = normalize_bhavcopy_rows(df)
    r = rows[0]
    assert r["stock_symbol"] == "RELIANCE"
    assert r["interval"] == "1d"
    assert r["open"] == 100 and r["close"] == 101 and r["volume"] == 12345
    assert r["delivery_qty"] == 6000
    assert r["source"] == "nselib"


def test_normalize_handles_empty():
    assert normalize_bhavcopy_rows(pd.DataFrame()) == []


def test_normalize_reads_bom_prefixed_symbol_column():
    # NSE prepends a BOM to the first CSV column, so "Symbol" arrives as
    # "﻿Symbol". Name-matching used to miss it -> symbol read as the
    # string "None" -> 247 bars landed under stock_symbol='None', not 'ABB'.
    df = pd.DataFrame([
        {"﻿Symbol": "ABB", "Date": "08-Jun-2026", "OpenPrice": 7105.5,
         "HighPrice": 7129.5, "LowPrice": 6893.0, "ClosePrice": 6958.0,
         "TotalTradedQuantity": 240753, "DeliverableQty": 105539,
         "%DlyQttoTradedQty": 43.84},
    ])
    rows = normalize_bhavcopy_rows(df)
    assert len(rows) == 1
    assert rows[0]["stock_symbol"] == "ABB"
    assert rows[0]["close"] == 6958.0


def test_normalize_symbol_override_is_authoritative():
    # When the caller passes the symbol it requested, that wins over whatever
    # (possibly BOM-garbled) value sits in the frame's Symbol column.
    df = pd.DataFrame([
        {"﻿Symbol": "None", "Date": "08-Jun-2026", "OpenPrice": 100,
         "HighPrice": 102, "LowPrice": 99, "ClosePrice": 101,
         "TotalTradedQuantity": 12345},
    ])
    rows = normalize_bhavcopy_rows(df, symbol="abb")
    assert len(rows) == 1
    assert rows[0]["stock_symbol"] == "ABB"  # forced + upper-cased


def test_normalize_tolerates_nan_int_fields():
    # NSE leaves blank int fields (DeliverableQty / volume) on some days -> NaN.
    # `int(float('nan'))` used to crash the whole fetch (and the EOD cron); NaN
    # must normalize to None, not raise. Regression for the 5yr-backfill blocker.
    import numpy as np
    df = pd.DataFrame([
        {"Symbol": "RELIANCE", "Date": "30-Apr-2026", "OpenPrice": 100, "HighPrice": 102,
         "LowPrice": 99, "ClosePrice": 101, "TotalTradedQuantity": np.nan,
         "DeliverableQty": np.nan, "%DlyQttoTradedQty": np.nan},
    ])
    rows = normalize_bhavcopy_rows(df)  # must not raise
    assert len(rows) == 1
    assert rows[0]["close"] == 101
    assert rows[0]["volume"] is None and rows[0]["delivery_qty"] is None
