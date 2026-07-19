"""Probability Engine — pure follow-through measurement (#17)."""
from backend.services.scanners.probability_engine import _followthrough


def test_followthrough_rate():
    # idx0 entry 100 -> reaches 105 (+5%) within 3 bars = success
    # idx4 entry 100 -> stays flat = fail
    closes = [100, 101, 105, 103, 100, 100, 100, 100]
    r = _followthrough([0, 4], closes, horizon=3, target=2.0)
    assert r["occurrences"] == 2
    assert r["success"] == 1
    assert r["prob_pct"] == 50.0


def test_followthrough_skips_near_end_and_empty():
    assert _followthrough([], [1, 2, 3], 2, 1)["prob_pct"] is None
    # an index without `horizon` bars ahead is skipped, not counted
    r = _followthrough([5], [1, 2, 3, 4, 5, 6], horizon=3, target=1)
    assert r["occurrences"] == 0 and r["prob_pct"] is None
