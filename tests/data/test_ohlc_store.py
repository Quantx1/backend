import pandas as pd
from backend.data.ohlc_store import rows_to_df, df_to_candle_rows


def test_df_to_candle_rows_and_back():
    df = pd.DataFrame({
        "open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0], "volume": [1000],
    }, index=pd.to_datetime(["2026-06-06"]))
    rows = df_to_candle_rows("RELIANCE", df, interval="1d", source="nselib")
    assert rows[0]["stock_symbol"] == "RELIANCE"
    assert rows[0]["close"] == 101.0 and rows[0]["interval"] == "1d"

    back = rows_to_df([
        {"timestamp": "2026-06-06T00:00:00+00:00", "open": 100.0, "high": 102.0,
         "low": 99.0, "close": 101.0, "volume": 1000},
    ])
    assert list(back.columns) == ["open", "high", "low", "close", "volume"]
    assert back.iloc[0]["close"] == 101.0


def test_rows_to_df_empty():
    assert rows_to_df([]).empty
