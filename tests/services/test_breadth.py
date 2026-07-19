"""Market breadth — pure cumulative A/D (#breadth)."""
from backend.services.scanners.breadth import cumulative_ad


def test_cumulative_ad():
    daily = [
        {"date": "d1", "adv": 10, "dec": 5},
        {"date": "d2", "adv": 3, "dec": 8},
        {"date": "d3", "adv": 6, "dec": 6},
    ]
    out = cumulative_ad(daily)
    assert out[0]["net"] == 5 and out[0]["ad_line"] == 5
    assert out[1]["net"] == -5 and out[1]["ad_line"] == 0
    assert out[2]["net"] == 0 and out[2]["ad_line"] == 0


def test_cumulative_ad_empty():
    assert cumulative_ad([]) == []
