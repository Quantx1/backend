from backend.data.reference.nse_orderflow import map_participant_oi_rows, map_fno_ban_symbols
import pandas as pd


def test_participant_directions_complete():
    df = pd.DataFrame([{"Client Type": p} for p in ["Client", "Pro", "FII", "DII"]])
    rows = map_participant_oi_rows(df, "d")
    assert {r["participant"] for r in rows} == {"client", "pro", "fii", "dii"}


def test_fno_ban_mapper_honest_empty():
    assert map_fno_ban_symbols([], "d") == []
