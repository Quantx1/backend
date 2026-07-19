"""Pure-function tests for ai.signals.ensemble.

Locked formulas — any change here should be paired with a memory update
explaining the new weighting.
"""
import pytest

from backend.ai.signals import EnsembleVoter, compute_ensemble_score, regime_bonus


def _v(name: str, weight: float, score: float, agrees: bool = True) -> EnsembleVoter:
    return EnsembleVoter(name=name, weight=weight, score=score, direction_agrees=agrees)


class TestComputeEnsembleScore:
    def test_empty_voters_raises(self):
        with pytest.raises(ValueError, match="no voters"):
            compute_ensemble_score([])

    def test_zero_weight_total_raises(self):
        with pytest.raises(ValueError, match="weights sum to <= 0"):
            compute_ensemble_score([_v("tft", 0.0, 0.8)])

    def test_single_voter_passthrough(self):
        # 100 * (1.0 * clip(0.8)) / 1.0 = 80.0
        assert compute_ensemble_score([_v("tft", 1.0, 0.8)]) == pytest.approx(80.0)

    def test_score_is_clipped_above(self):
        # score > 1 clipped to 1 → returns 100
        assert compute_ensemble_score([_v("tft", 1.0, 1.5)]) == pytest.approx(100.0)

    def test_score_is_clipped_below(self):
        # score < 0 clipped to 0 → returns 0
        assert compute_ensemble_score([_v("tft", 1.0, -0.3)]) == pytest.approx(0.0)

    def test_weighted_average(self):
        # 0.5 weight on 1.0 + 0.5 weight on 0.0 = 50.0
        voters = [_v("a", 0.5, 1.0), _v("b", 0.5, 0.0)]
        assert compute_ensemble_score(voters) == pytest.approx(50.0)

    def test_unequal_weights(self):
        # 0.8*1.0 + 0.2*0.5 = 0.9 → scaled by /1.0 → 90.0
        voters = [_v("a", 0.8, 1.0), _v("b", 0.2, 0.5)]
        assert compute_ensemble_score(voters) == pytest.approx(90.0)


class TestRegimeBonus:
    def test_bull_is_one(self):
        assert regime_bonus(0) == 1.0

    def test_sideways_is_half(self):
        assert regime_bonus(1) == 0.5

    def test_bear_is_zero(self):
        assert regime_bonus(2) == 0.0

    def test_none_defaults_to_half(self):
        assert regime_bonus(None) == 0.5

    def test_unknown_id_defaults_to_half(self):
        assert regime_bonus(99) == 0.5
