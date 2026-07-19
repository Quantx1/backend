"""Footprint / CVD — pure bar-delta + cumulative (#21)."""
from backend.services.market.footprint import bar_delta, buy_pct, compute_cvd


def test_bar_delta_buying():
    assert bar_delta(110, 90, 110, 1000) == 1000.0   # close at high
    assert buy_pct(110, 90, 110) == 100.0


def test_bar_delta_selling():
    assert bar_delta(110, 90, 90, 1000) == -1000.0   # close at low
    assert buy_pct(110, 90, 90) == 0.0


def test_bar_delta_mid_and_zero_range():
    assert bar_delta(110, 90, 100, 1000) == 0.0
    assert buy_pct(110, 90, 100) == 50.0
    assert bar_delta(100, 100, 100, 1000) == 0.0     # zero range


def test_compute_cvd_accumulates():
    bars = [
        {"date": "d1", "high": 110, "low": 90, "close": 110, "volume": 100},   # +100
        {"date": "d2", "high": 110, "low": 90, "close": 90, "volume": 50},     # -50
    ]
    out = compute_cvd(bars)
    assert out[0]["cvd"] == 100.0
    assert out[1]["delta"] == -50.0 and out[1]["cvd"] == 50.0
