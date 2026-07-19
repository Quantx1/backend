"""Market Profile / TPO — pure compute (#21)."""
from backend.services.market.market_profile import compute_tpo


def test_tpo_poc_and_value_area():
    # Most bars overlap around 100; a few outliers up high.
    bars = [{"high": 105, "low": 95}] * 10 + [{"high": 102, "low": 98}] * 30 + [{"high": 122, "low": 118}] * 2
    out = compute_tpo(bars, bins=20)
    assert out["poc"] is not None
    assert 96 <= out["poc"] <= 104           # POC where bars cluster
    assert out["val"] <= out["poc"] <= out["vah"]
    assert out["total_tpo"] > 0


def test_tpo_empty_and_flat():
    assert compute_tpo([])["poc"] is None
    assert compute_tpo([{"high": 100, "low": 100}])["poc"] is None  # zero range
