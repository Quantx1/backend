import pandas as pd
from backend.platform.scheduler import equity_master_to_rows, bhavcopy_to_candle_rows


def test_equity_master_to_rows():
    df = pd.DataFrame([{"SYMBOL": "TCS", " ISIN NUMBER": "INE467B01029", " SERIES": "EQ",
                        "NAME OF COMPANY": "TCS"}])
    rows = equity_master_to_rows(df)
    assert rows[0]["symbol"] == "TCS" and rows[0]["isin"] == "INE467B01029"


def test_bhavcopy_to_candle_rows():
    df = pd.DataFrame([{"Symbol": "TCS", "Date": "2026-06-06", "OpenPrice": 1, "HighPrice": 2,
                        "LowPrice": 1, "ClosePrice": 2, "TotalTradedQuantity": 5}])
    rows = bhavcopy_to_candle_rows(df)
    assert rows[0]["stock_symbol"] == "TCS" and rows[0]["close"] == 2
