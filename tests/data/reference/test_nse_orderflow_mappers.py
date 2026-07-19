# tests/data/reference/test_nse_orderflow_mappers.py
import pandas as pd
from backend.data.reference.nse_orderflow import (
    map_participant_oi_rows, map_fii_dii_rows, map_bulk_block_rows,
    map_short_selling_rows, map_fno_ban_symbols,
)


def test_bulk_block_normalizes_nse_date_to_iso():
    # NSE returns DD-Mon-YYYY; it must be normalized to ISO for the Postgres DATE col.
    df = pd.DataFrame([{"Symbol": "TCS", "Client Name": "ABC", "Buy/Sell": "BUY",
                        "Quantity Traded": 10, "Date": "06-Jun-2026"}])
    assert map_bulk_block_rows(df, "BULK")[0]["date"] == "2026-06-06"


def test_short_selling_falls_back_to_trade_date_on_bad_date():
    df = pd.DataFrame([{"Symbol": "INFY", "Quantity": 5, "Date": "garbage"}])
    assert map_short_selling_rows(df, "2026-06-06")[0]["date"] == "2026-06-06"


def test_map_participant_oi_rows():
    df = pd.DataFrame([
        {"Client Type": "FII", "Future Index Long": 10, "Future Stock Long": 5,
         "Future Index Short": 2, "Future Stock Short": 3,
         "Option Index Call Long": 1, "Option Stock Call Long": 1,
         "Option Index Call Short": 0, "Option Stock Call Short": 0,
         "Option Index Put Long": 4, "Option Stock Put Long": 0,
         "Option Index Put Short": 1, "Option Stock Put Short": 1},
    ])
    rows = map_participant_oi_rows(df, "2026-06-06")
    r = rows[0]
    assert r["date"] == "2026-06-06" and r["participant"] == "fii"
    assert r["fut_long"] == 15 and r["fut_short"] == 5
    assert r["opt_call_long"] == 2 and r["opt_put_long"] == 4
    assert r["source"] == "nselib"


def test_map_fii_dii_rows():
    df = pd.DataFrame([
        {"category": "FII/FPI", "buyValue": 1000, "sellValue": 800, "netValue": 200},
        {"category": "DII", "buyValue": 500, "sellValue": 600, "netValue": -100},
    ])
    rows = map_fii_dii_rows(df, "2026-06-06")
    assert rows[0]["date"] == "2026-06-06" and rows[0]["segment"] == "CASH"
    assert rows[0]["fii_net"] == 200 and rows[0]["dii_net"] == -100


def test_map_bulk_block_rows():
    df = pd.DataFrame([{"Symbol": "TCS", "Client Name": "ABC", "Buy/Sell": "BUY",
                        "Quantity Traded": 1000, "Trade Price / Wght. Avg. Price": 3900,
                        "Date": "2026-06-06"}])
    rows = map_bulk_block_rows(df, deal_type="BULK")
    assert rows[0]["symbol"] == "TCS" and rows[0]["deal_type"] == "BULK"
    assert rows[0]["qty"] == 1000 and rows[0]["buy_sell"] == "BUY"


def test_map_short_selling_and_ban():
    sdf = pd.DataFrame([{"Symbol": "INFY", "Quantity": 500, "Date": "2026-06-06"}])
    assert map_short_selling_rows(sdf, "2026-06-06")[0]["qty"] == 500
    bans = map_fno_ban_symbols(["RELIANCE", "TCS"], "2026-06-06")
    assert {b["symbol"] for b in bans} == {"RELIANCE", "TCS"}
    assert bans[0]["date"] == "2026-06-06"


def test_honest_empty():
    assert map_participant_oi_rows(pd.DataFrame(), "d") == []
    assert map_fno_ban_symbols([], "d") == []
