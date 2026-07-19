"""IV Rank / IV Percentile — pure compute (Volatility Intelligence, #13)."""
from backend.services.fno_scanner.iv_store import compute_iv_rank_percentile
from backend.services.fno_scanner.snapshot import teach_snapshot


def test_iv_rank_percentile_math():
    series = [10, 12, 14, 16, 18, 20] + [15] * 20  # 26 days, min 10 / max 20
    r = compute_iv_rank_percentile(series, 18.0)
    assert r["iv_rank"] == 80.0           # (18-10)/(20-10)*100
    assert r["days"] == 26
    assert 0 <= r["iv_percentile"] <= 100


def test_iv_rank_honest_null_until_min_days():
    r = compute_iv_rank_percentile([10, 12, 14], 13.0)
    assert r["iv_rank"] is None and r["iv_percentile"] is None and r["days"] == 3


def test_teach_includes_iv_rank_line():
    lines = teach_snapshot({"iv_rank": 85})
    assert any("IV Rank is 85" in line and "premium-selling" in line for line in lines)
