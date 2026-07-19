"""AI Options Teacher — deterministic plain-English chain read (0 LLM tokens)."""
from backend.services.fno_scanner.snapshot import teach_snapshot


def test_teach_snapshot_reads_levels_and_lean():
    d = {
        "symbol": "NIFTY", "spot": 25000, "pcr_oi": 1.4, "max_pain": 24900,
        "max_pain_distance_pct": 0.4, "top_call_oi_strikes": [25200, 25500],
        "top_put_oi_strikes": [24800, 24500], "iv_atm": 0.12, "days_to_expiry": 1,
        "biggest_oi_buildup": {"strike": 25200, "side": "call",
                               "direction": "writing", "oi_change": 50000},
    }
    lines = teach_snapshot(d)
    txt = " ".join(lines)
    assert any(line.startswith("PCR (OI) is 1.40") for line in lines)
    assert "bullish" in txt                  # pcr 1.4 -> bullish/oversold tilt
    assert "Max pain is 24900" in txt
    assert "resistance" in txt and "support" in txt
    assert "ATM IV is 12.0%" in txt
    assert "expiry" in txt                    # 1 DTE warning
    assert "CALL" in txt and "capped" in txt  # buildup line


def test_teach_snapshot_empty_is_safe():
    assert teach_snapshot({}) == []
