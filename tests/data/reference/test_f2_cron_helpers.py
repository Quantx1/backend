import pandas as pd
from backend.platform.scheduler import participant_oi_to_rows


def test_participant_oi_to_rows():
    df = pd.DataFrame([{"Client Type": "DII", "Future Index Long": 7, "Future Stock Long": 0,
                        "Future Index Short": 1, "Future Stock Short": 0}])
    rows = participant_oi_to_rows(df, "2026-06-06")
    assert rows[0]["participant"] == "dii" and rows[0]["fut_long"] == 7
