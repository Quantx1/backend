# tests/data/reference/test_nse_derivatives_mappers.py
import pandas as pd
from backend.data.reference.nse_derivatives import (
    map_fno_options_rows, map_fno_futures_rows, build_derivatives_metrics,
)


def _bhav():
    # UDiFF-ish F&O bhavcopy: 2 options + 1 future for RELIANCE
    return pd.DataFrame([
        {"FinInstrmTp": "STO", "TckrSymb": "RELIANCE", "XpryDt": "26-06-2026",
         "StrkPric": 2900, "OptnTp": "CE", "OpnIntrst": 1000, "ChngInOpnIntrst": 100,
         "TtlTradgVol": 50, "ClsPric": 120},
        {"FinInstrmTp": "STO", "TckrSymb": "RELIANCE", "XpryDt": "26-06-2026",
         "StrkPric": 2900, "OptnTp": "PE", "OpnIntrst": 2000, "ChngInOpnIntrst": -50,
         "TtlTradgVol": 80, "ClsPric": 95},
        {"FinInstrmTp": "STF", "TckrSymb": "RELIANCE", "XpryDt": "26-06-2026",
         "OpnPric": 2890, "HghPric": 2910, "LwPric": 2880, "ClsPric": 2905,
         "OpnIntrst": 5000, "ChngInOpnIntrst": 200, "TtlTradgVol": 1200},
    ])


def test_map_options_rows():
    rows = map_fno_options_rows(_bhav(), "2026-06-06")
    assert len(rows) == 2
    ce = next(r for r in rows if r["option_type"] == "CE")
    assert ce["symbol"] == "RELIANCE" and ce["strike"] == 2900 and ce["oi"] == 1000
    assert ce["expiry"] == "2026-06-26" and ce["ltp"] == 120 and ce["date"] == "2026-06-06"


def test_map_futures_rows():
    rows = map_fno_futures_rows(_bhav(), "2026-06-06")
    assert len(rows) == 1
    assert rows[0]["close"] == 2905 and rows[0]["oi"] == 5000 and rows[0]["expiry"] == "2026-06-26"


def test_build_metrics_pcr_and_maxpain():
    opt_rows = map_fno_options_rows(_bhav(), "2026-06-06")
    metrics = build_derivatives_metrics(opt_rows)
    m = metrics[0]
    assert m["symbol"] == "RELIANCE" and m["expiry"] == "2026-06-26"
    assert round(m["pcr_oi"], 2) == 2.0          # PE_OI(2000)/CE_OI(1000)
    assert m["total_ce_oi"] == 1000 and m["total_pe_oi"] == 2000
    assert m["max_pain"] == 2900                  # only strike present


def test_honest_empty():
    assert map_fno_options_rows(pd.DataFrame(), "d") == []
    assert build_derivatives_metrics([]) == []
