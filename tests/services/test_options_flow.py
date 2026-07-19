"""Options Flow aggregator — pure aggregation + lean classification.

Feeds a fabricated option chain (no network, no DB, no LLM) and verifies:
  (a) writing totals sum ONLY positive Δ-OI per side,
  (b) PCR = total PE_OI / total CE_OI,
  (c) max-pain pull is signed (max_pain - spot) / spot,
  (d) top buildup is the largest |Δ-OI| moves, both sides,
  (e) the deterministic lean for bullish / bearish / neutral / conflict cases,
  (f) honest-empty (None) on an empty / all-zero chain.
"""
from backend.services.fno_scanner.options_flow import (
    aggregate_flow,
    _writing_vote,
    _pcr_vote,
    _combine_lean,
)


def _row(otype, strike, oi, oi_change):
    return {"option_type": otype, "strike": strike, "oi": oi, "oi_change": oi_change}


# A chain leaning BULLISH: put-writing dominates + PCR > 1 (more put OI).
_BULLISH_CHAIN = [
    _row("CE", 24000, 100_000, 5_000),
    _row("PE", 24000, 180_000, 90_000),    # heavy fresh put writing
    _row("CE", 24100, 80_000, 2_000),
    _row("PE", 23900, 160_000, 70_000),
    _row("CE", 24200, 60_000, -3_000),     # some call unwinding (negative, not writing)
    _row("PE", 23800, 140_000, 40_000),
]


def test_writing_totals_count_only_positive_delta():
    s = aggregate_flow(_BULLISH_CHAIN, spot=24010.0)
    # Put writing = 90k + 70k + 40k ; Call writing = 5k + 2k (the -3k is unwinding)
    assert s["total_put_writing"] == 200_000
    assert s["total_call_writing"] == 7_000


def test_pcr_is_put_oi_over_call_oi():
    s = aggregate_flow(_BULLISH_CHAIN, spot=24010.0)
    total_ce = 100_000 + 80_000 + 60_000
    total_pe = 180_000 + 160_000 + 140_000
    assert s["pcr"] == round(total_pe / total_ce, 3)
    assert s["pcr"] > 1.0


def test_max_pain_pull_is_signed_relative_to_spot():
    s = aggregate_flow(_BULLISH_CHAIN, spot=24010.0)
    assert s["max_pain"] is not None
    # pull = (max_pain - spot) / spot * 100, rounded
    expected = round((s["max_pain"] - 24010.0) / 24010.0 * 100.0, 2)
    assert s["max_pain_pull_pct"] == expected


def test_top_buildup_is_largest_abs_delta_first():
    s = aggregate_flow(_BULLISH_CHAIN, spot=24010.0)
    top = s["top_buildup"]
    assert top[0]["oi_change"] == 90_000          # biggest move overall
    assert top[0]["side"] == "PE"
    assert top[0]["direction"] == "writing"
    # monotonically non-increasing by |Δ-OI|
    mags = [abs(m["oi_change"]) for m in top]
    assert mags == sorted(mags, reverse=True)
    assert s["biggest_buildup"] == top[0]


def test_lean_bullish_when_putwriting_and_high_pcr():
    s = aggregate_flow(_BULLISH_CHAIN, spot=24010.0)
    assert s["writing_vote"] == "bullish"
    assert s["pcr_vote"] == "bullish"
    assert s["lean"] == "bullish"


def test_lean_bearish_when_callwriting_and_low_pcr():
    chain = [
        _row("CE", 24000, 200_000, 95_000),    # heavy fresh call writing
        _row("PE", 24000, 90_000, 3_000),
        _row("CE", 23900, 180_000, 60_000),
        _row("PE", 24100, 70_000, 1_000),
        _row("CE", 23800, 160_000, 40_000),
        _row("PE", 24200, 60_000, 500),
    ]
    s = aggregate_flow(chain, spot=24010.0)
    assert s["total_call_writing"] > s["total_put_writing"]
    assert s["pcr"] < 0.7
    assert s["writing_vote"] == "bearish"
    assert s["pcr_vote"] == "bearish"
    assert s["lean"] == "bearish"


def test_lean_neutral_on_conflict():
    # Put-writing dominant (bullish writing) but PCR low (bearish PCR) -> conflict.
    assert _combine_lean("bullish", "bearish") == "neutral"
    assert _combine_lean("bearish", "bullish") == "neutral"
    # Balanced writing + mid PCR -> neutral.
    assert _combine_lean("neutral", "neutral") == "neutral"


def test_combine_lean_single_directional_vote_wins():
    assert _combine_lean("bullish", "neutral") == "bullish"
    assert _combine_lean("neutral", "bearish") == "bearish"


def test_writing_vote_balanced_is_neutral():
    # Within the dominance band -> neutral (don't read noise as positioning).
    assert _writing_vote(100_000, 105_000) == "neutral"
    assert _writing_vote(0, 0) == "neutral"
    assert _writing_vote(10_000, 100_000) == "bullish"
    assert _writing_vote(100_000, 10_000) == "bearish"


def test_pcr_vote_bands():
    assert _pcr_vote(1.4) == "bullish"
    assert _pcr_vote(1.0) == "bullish"
    assert _pcr_vote(0.85) == "neutral"
    assert _pcr_vote(0.7) == "bearish"
    assert _pcr_vote(0.4) == "bearish"
    assert _pcr_vote(None) == "neutral"


def test_honest_empty_on_empty_chain():
    assert aggregate_flow([], spot=24000.0) is None
    assert aggregate_flow(None, spot=24000.0) is None


def test_honest_empty_on_all_zero_oi():
    chain = [_row("CE", 24000, 0, 0), _row("PE", 24000, 0, 0)]
    assert aggregate_flow(chain, spot=24000.0) is None


def test_max_pain_pull_none_without_spot():
    s = aggregate_flow(_BULLISH_CHAIN, spot=None)
    assert s is not None
    assert s["max_pain"] is not None
    assert s["max_pain_pull_pct"] is None
