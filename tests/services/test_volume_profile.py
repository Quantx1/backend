"""Volume Profile — pure compute_volume_profile core (no I/O)."""
from backend.services.market.volume_profile import compute_volume_profile


def test_honest_empty_below_min_bars():
    out = compute_volume_profile([1, 2], [0.5, 1], [10, 10], bins=10)
    assert out["poc"] is None and out["bins"] == []


def test_poc_at_heaviest_price_band():
    # 20 bars; the 9.9-10.1 band gets hammered with volume → POC near 10.
    highs = [10.1] * 20
    lows = [9.9] * 20
    vols = [1000] * 20
    # add a few light bars elsewhere
    highs += [12.0, 12.0]
    lows += [11.8, 11.8]
    vols += [10, 10]
    out = compute_volume_profile(highs, lows, vols, bins=24)
    assert out["poc"] is not None
    assert 9.8 <= out["poc"] <= 10.2  # POC sits in the heavy band


def test_value_area_holds_at_least_70pct():
    highs = [10 + (i % 5) * 0.1 for i in range(60)]
    lows = [9 + (i % 5) * 0.1 for i in range(60)]
    vols = [100 + (i % 7) * 50 for i in range(60)]
    out = compute_volume_profile(highs, lows, vols, bins=20)
    assert out["val"] is not None and out["vah"] is not None
    assert out["val"] <= out["poc"] <= out["vah"]
    assert out["value_area_pct"] >= 70.0


def test_hvn_lvn_labelled():
    # one dominant band → at least one HVN; sparse edges → LVN possible
    highs = [10.05] * 30 + [15.0, 16.0, 17.0]
    lows = [9.95] * 30 + [14.9, 15.9, 16.9]
    vols = [2000] * 30 + [5, 5, 5]
    out = compute_volume_profile(highs, lows, vols, bins=30)
    assert len(out["hvn"]) >= 1
    # the heavy node price should be among HVN
    assert any(9.5 <= p <= 10.5 for p in out["hvn"])


def test_flat_range_is_honest_empty():
    out = compute_volume_profile([10] * 10, [10] * 10, [100] * 10, bins=12)
    assert out["poc"] is None  # hi == lo → no range
