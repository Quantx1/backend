import pandas as pd
from backend.data.reference.nse_reference import (
    map_equity_master_rows, map_corporate_action_rows,
)


def test_map_equity_master_rows():
    df = pd.DataFrame([
        {"SYMBOL": "RELIANCE", "NAME OF COMPANY": "Reliance", " ISIN NUMBER": "INE002A01018",
         " SERIES": "EQ", " FACE VALUE": "10", " DATE OF LISTING": "29-NOV-1995"},
    ])
    rows = map_equity_master_rows(df)
    assert rows[0]["symbol"] == "RELIANCE"
    assert rows[0]["isin"] == "INE002A01018"
    assert rows[0]["series"] == "EQ"
    assert rows[0]["instrument_type"] == "EQ"
    assert rows[0]["exchange"] == "NSE"
    assert rows[0]["source"] == "nselib"


def test_map_corporate_action_rows_split():
    raw = [{"symbol": "TCS", "exDate": "2024-01-25", "purpose": "Face Value Split From Rs 10 to Rs 5"}]
    rows = map_corporate_action_rows(raw)
    assert rows[0]["symbol"] == "TCS"
    assert rows[0]["ex_date"] == "2024-01-25"
    assert rows[0]["action_type"] in ("split", "fv_change")
    assert "Face Value Split" in rows[0]["details"]["purpose"]
