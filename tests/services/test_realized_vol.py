"""Historical / Realized Volatility (HV) — pure compute.

Surfaces HV alongside IV in the F&O panel. The annualisation math mirrors
``ai/strategy/indicators._realized_volatility`` (std of close-to-close log
returns × sqrt(252) × 100). These tests pin that math on synthetic close
series — no network, no provider, no DB.
"""

from __future__ import annotations

import math

from backend.services.fno_scanner.volatility import compute_hv


def test_constant_growth_series_has_near_zero_hv():
    """A perfectly geometric series (constant daily log return) has zero
    dispersion in its log returns, so realized vol must be ~0."""
    closes = [100.0 * (1.01 ** i) for i in range(40)]
    r = compute_hv(closes, windows=(10, 20, 30))
    assert r is not None
    # All windows present (40 closes > 30 + 1)
    assert set(r["hv"].keys()) == {"10", "20", "30"}
    for v in r["hv"].values():
        assert abs(v) < 1e-6
    assert r["latest_hv"] == r["hv"]["20"]


def test_alternating_series_matches_hand_computed_annualized_vol():
    """Closes alternating ×1.02 then ÷1.02 give log returns of ±ln(1.02).
    Over a 20-bar window (sample std, ddof=1) the annualised HV is a value we
    can compute by hand: 32.25%."""
    c = [100.0]
    for i in range(1, 41):
        c.append(c[-1] * 1.02 if i % 2 == 1 else c[-1] / 1.02)
    r = compute_hv(c, windows=(20,))
    assert r is not None

    ret = math.log(1.02)
    sample_std = math.sqrt(sum(x * x for x in ([ret, -ret] * 10)) / 19)  # mean 0, n=20, ddof=1
    expected = round(sample_std * math.sqrt(252) * 100, 2)
    assert r["hv"]["20"] == expected == 32.25
    assert r["latest_hv"] == r["hv"]["20"]


def test_honest_empty_when_series_too_short():
    """Not enough closes to fill the smallest window -> None (no fabrication)."""
    assert compute_hv([100.0, 101.0, 102.0], windows=(10, 20, 30)) is None


def test_partial_windows_only_computes_what_fits():
    """15 closes can fill the 10-window but not 20/30 -> only '10' returned,
    and latest_hv falls back to the largest computed window."""
    # gentle upward drift with small noise so vol is finite/positive
    closes = [100.0 + i + (0.5 if i % 2 else -0.5) for i in range(15)]
    r = compute_hv(closes, windows=(10, 20, 30))
    assert r is not None
    assert set(r["hv"].keys()) == {"10"}
    assert r["latest_hv"] == r["hv"]["10"]
    assert r["hv"]["10"] > 0


def test_non_positive_price_is_rejected():
    """log-returns need strictly positive prices; a zero/negative close bails
    honestly rather than emitting NaN-laced vol."""
    closes = [100.0] * 5 + [0.0] + [100.0] * 30
    assert compute_hv(closes, windows=(10, 20, 30)) is None


def test_note_and_shape():
    closes = [100.0 * (1.0 + 0.001 * ((-1) ** i)) for i in range(35)]
    r = compute_hv(closes, windows=(10, 20, 30))
    assert r is not None
    assert "hv" in r and "latest_hv" in r and "note" in r
    assert "realized" in r["note"].lower()
