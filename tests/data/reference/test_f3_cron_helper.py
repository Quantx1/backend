# tests/data/reference/test_f3_cron_helper.py
import pandas as pd
from backend.platform.scheduler import fno_bhav_to_option_rows


def test_fno_bhav_to_option_rows():
    df = pd.DataFrame([{"OptnTp": "CE", "TckrSymb": "TCS", "XpryDt": "26-06-2026",
                        "StrkPric": 3900, "OpnIntrst": 10, "TtlTradgVol": 1, "ClsPric": 5}])
    rows = fno_bhav_to_option_rows(df, "2026-06-06")
    assert rows[0]["symbol"] == "TCS" and rows[0]["option_type"] == "CE"
