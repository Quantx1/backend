"""AI Factor Screener — pure percentile + composite ranking + factor math.

No network / DB: every test feeds a fabricated factor matrix or close series.
"""
from backend.services.scanners.factor_screener import (
    AVAILABLE_FACTORS,
    _percentile_ranks,
    compute_factor_ranking,
    momentum_50d,
    low_volatility_20d,
    trend_above_50dma,
)


def test_percentile_ranks_best_and_worst():
    # 4 values: best -> 100, worst -> 0, evenly spaced in between.
    out = _percentile_ranks([1.0, 2.0, 3.0, 4.0])
    assert out[0] == 0.0
    assert out[3] == 100.0
    # middle two: 1 and 2 peers strictly below / (n-1=3)
    assert out[1] == round(1 / 3 * 100, 2)
    assert out[2] == round(2 / 3 * 100, 2)


def test_percentile_ranks_handles_none_and_singleton():
    out = _percentile_ranks([None, 5.0, None])
    # the lone present value is the whole cross-section -> top percentile
    assert out == [None, 100.0, None]
    assert _percentile_ranks([]) == []
    assert _percentile_ranks([None, None]) == [None, None]


def test_percentile_ranks_ties_share_lower_bound():
    # two tied at the top: each has 1 peer strictly below / (n-1=2) = 50
    out = _percentile_ranks([1.0, 3.0, 3.0])
    assert out[0] == 0.0
    assert out[1] == 50.0
    assert out[2] == 50.0


def test_composite_is_mean_of_selected_percentiles():
    # A is best on momentum but worst on low_vol; B is the all-rounder.
    matrix = {
        "A": {"momentum": 10.0, "low_volatility": 1.0},
        "B": {"momentum": 5.0, "low_volatility": 5.0},
        "C": {"momentum": 1.0, "low_volatility": 10.0},
    }
    res = compute_factor_ranking(matrix, ["momentum", "low_volatility"], top=25)
    by_sym = {r["symbol"]: r for r in res}
    # momentum pct: A=100 B=50 C=0 ; low_vol pct: A=0 B=50 C=100
    assert by_sym["A"]["factor_scores"]["momentum"] == 100.0
    assert by_sym["A"]["factor_scores"]["low_volatility"] == 0.0
    # composite is the mean -> everyone lands at 50, B included
    assert by_sym["A"]["composite"] == 50.0
    assert by_sym["B"]["composite"] == 50.0
    assert by_sym["C"]["composite"] == 50.0


def test_single_factor_ranking_sorted_desc_and_topn():
    matrix = {s: {"momentum": v} for s, v in
              {"A": 1.0, "B": 4.0, "C": 2.0, "D": 3.0}.items()}
    res = compute_factor_ranking(matrix, ["momentum"], top=2)
    assert [r["symbol"] for r in res] == ["B", "D"]  # top-2 by momentum
    assert res[0]["composite"] == 100.0
    assert res[1]["composite"] == round(2 / 3 * 100, 2)


def test_symbol_missing_all_requested_factors_is_dropped():
    matrix = {
        "A": {"momentum": 5.0},
        "B": {"momentum": None},   # no usable requested factor -> dropped
    }
    res = compute_factor_ranking(matrix, ["momentum"], top=25)
    assert [r["symbol"] for r in res] == ["A"]


def test_unknown_or_empty_factor_selection_is_honest_empty():
    matrix = {"A": {"momentum": 5.0}}
    assert compute_factor_ranking(matrix, ["fundamental_pe"], top=25) == []
    assert compute_factor_ranking(matrix, [], top=25) == []
    assert compute_factor_ranking({}, ["momentum"], top=25) == []


def test_momentum_50d_math():
    closes = [float(i) for i in range(1, 60)]  # 1..59 ascending
    # 50d window: closes[-51]=9, closes[-1]=59 -> (59/9 - 1)*100
    assert momentum_50d(closes) == round((59 / 9 - 1) * 100, 4)
    assert momentum_50d([1.0, 2.0]) is None  # too short


def test_low_volatility_inverse_of_vol():
    # A perfectly flat-return ramp has tiny vol -> large 1/vol factor; a
    # noisy series has higher vol -> smaller factor. Calmer must score higher.
    calm = [100.0 * (1.01 ** i) for i in range(25)]       # constant ~1% steps
    noisy = []
    px = 100.0
    for i in range(25):
        px *= 1.05 if i % 2 == 0 else 0.95
        noisy.append(px)
    fc = low_volatility_20d(calm)
    fn = low_volatility_20d(noisy)
    assert fc is not None and fn is not None
    assert fc > fn
    assert low_volatility_20d([1.0] * 10) is None  # too short


def test_trend_zero_when_50dma_not_rising():
    # Falling series -> 50DMA not rising -> factor pinned to 0.
    falling = [float(100 - i) for i in range(60)]  # 100 down to 41
    assert trend_above_50dma(falling) == 0.0
    # Rising series above a rising MA -> positive factor.
    rising = [float(40 + i) for i in range(60)]
    val = trend_above_50dma(rising)
    assert val is not None and val > 0.0
    assert trend_above_50dma([1.0] * 10) is None  # too short


def test_available_factors_only_declares_candle_derivable():
    # Honesty contract: no fundamentals-only factors are claimed.
    assert set(AVAILABLE_FACTORS) == {"momentum", "low_volatility", "trend"}
    assert "value" not in AVAILABLE_FACTORS
    assert "quality" not in AVAILABLE_FACTORS
