"""True relative strength vs benchmark — pure compute (#7)."""
from backend.services.scanners.relative_strength import compute_rel_return


def test_rel_return_outperformance():
    stock = [100.0] * 20 + [120.0]   # +20% over the 20-bar window
    bench = [100.0] * 20 + [110.0]   # +10%
    assert compute_rel_return(stock, bench, 20) == 10.0   # 20 - 10


def test_rel_return_underperformance():
    stock = [100.0] * 20 + [105.0]   # +5%
    bench = [100.0] * 20 + [112.0]   # +12%
    assert compute_rel_return(stock, bench, 20) == -7.0


def test_rel_return_short_series_is_none():
    assert compute_rel_return([1, 2, 3], [1, 2, 3], 20) is None
    assert compute_rel_return([], [], 5) is None
