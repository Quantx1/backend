"""Tests for ai.signals.voters — locked WEIGHTS + builder shape."""
import pytest

from backend.ai.signals import (
    EnsembleVoter,
    WEIGHTS,
    make_lgbm_voter,
    make_tft_voter,
    make_qlib_voter,
    make_regime_voter,
)


class TestWeights:
    def test_weights_are_the_four_voters(self):
        # Sentiment ("finbert_india") was removed from the ensemble 2026-06-06;
        # it's now a standalone on-demand engine, not a voter.
        assert set(WEIGHTS.keys()) == {
            "lgbm_signal_gate", "tft_swing", "qlib_alpha158", "hmm_regime",
        }

    def test_weights_positive(self):
        # compute_ensemble_score normalises by the sum of weights, so they
        # need not sum to exactly 1.0 — just be positive.
        assert all(0 < w <= 1 for w in WEIGHTS.values())
        assert sum(WEIGHTS.values()) == pytest.approx(0.9)


class TestLgbmVoter:
    def test_buy_direction_agrees(self):
        v = make_lgbm_voter(0.85, "BUY")
        assert v.name == "lgbm_signal_gate"
        assert v.score == 0.85
        assert v.direction_agrees is True
        assert v.weight == WEIGHTS["lgbm_signal_gate"]

    def test_sell_direction_disagrees(self):
        v = make_lgbm_voter(0.85, "SELL")
        assert v.direction_agrees is False


class TestTftVoter:
    def test_bullish_agrees(self):
        v = make_tft_voter(0.7, "bullish")
        assert v.name == "tft_swing"
        assert v.direction_agrees is True

    def test_bearish_disagrees(self):
        v = make_tft_voter(0.7, "bearish")
        assert v.direction_agrees is False


class TestQlibVoter:
    def test_high_score_agrees(self):
        v = make_qlib_voter(0.8)
        assert v.direction_agrees is True

    def test_low_score_disagrees(self):
        v = make_qlib_voter(0.3)
        assert v.direction_agrees is False

    def test_boundary_agrees(self):
        assert make_qlib_voter(0.5).direction_agrees is True


class TestRegimeVoter:
    def test_bull_regime_agrees(self):
        v = make_regime_voter(regime_id=0, bear_active=False)
        assert v.score == 1.0
        assert v.direction_agrees is True

    def test_sideways_agrees(self):
        v = make_regime_voter(regime_id=1, bear_active=False)
        assert v.score == 0.5
        assert v.direction_agrees is True

    def test_bear_disagrees(self):
        v = make_regime_voter(regime_id=2, bear_active=True)
        assert v.score == 0.0
        assert v.direction_agrees is False
