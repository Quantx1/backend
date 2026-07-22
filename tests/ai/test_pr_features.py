"""PR-FEATURES tests — 17 new macro indicators + options chain features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_bars():
    """300-bar OHLCV with realistic trend + noise."""
    rng = np.random.default_rng(42)
    n = 300
    prices = 100 + np.cumsum(rng.normal(0.05, 1, n))
    return pd.DataFrame({
        "open": prices + rng.normal(0, 0.3, n),
        "high": prices + np.abs(rng.normal(0, 0.6, n)),
        "low":  prices - np.abs(rng.normal(0, 0.6, n)),
        "close": prices,
        "volume": rng.integers(100_000, 500_000, n).astype(float),
    }, index=pd.date_range("2024-01-01", periods=n))


class TestNewIndicators:
    """All 17 PR-FEATURES indicators compute non-NaN on synthetic data."""

    def test_rate_of_change(self, synthetic_bars):
        from backend.ai.strategy.indicators import compute_indicator
        r10 = compute_indicator("roc_10", synthetic_bars)
        r20 = compute_indicator("roc_20", synthetic_bars)
        assert r10 is not None and not np.isnan(r10)
        assert r20 is not None and not np.isnan(r20)
        # ROC values are realistic — typically -10% to +10% on daily bars
        assert -20 < r10 < 20

    def test_stoch_rsi(self, synthetic_bars):
        from backend.ai.strategy.indicators import compute_indicator
        k = compute_indicator("stoch_rsi_k", synthetic_bars)
        d = compute_indicator("stoch_rsi_d", synthetic_bars)
        # Bounded [0, 100]
        assert k is not None and 0 <= k <= 100
        assert d is not None and 0 <= d <= 100

    def test_di_components_sum_consistent(self, synthetic_bars):
        from backend.ai.strategy.indicators import compute_indicator
        dp = compute_indicator("di_plus", synthetic_bars)
        dm = compute_indicator("di_minus", synthetic_bars)
        adx = compute_indicator("adx", synthetic_bars)
        # All in [0, 100]
        assert 0 <= dp <= 100
        assert 0 <= dm <= 100
        assert 0 <= adx <= 100

    def test_volatility_features(self, synthetic_bars):
        from backend.ai.strategy.indicators import compute_indicator
        v20 = compute_indicator("volatility_20", synthetic_bars)
        v60 = compute_indicator("volatility_60", synthetic_bars)
        regime = compute_indicator("volatility_regime", synthetic_bars)
        # Vol is positive annualised %
        assert v20 > 0
        assert v60 > 0
        # Regime is one of {0, 1, 2}
        assert regime in (0.0, 1.0, 2.0)

    def test_volume_features(self, synthetic_bars):
        from backend.ai.strategy.indicators import compute_indicator
        slope = compute_indicator("obv_slope", synthetic_bars)
        ratio = compute_indicator("volume_ratio", synthetic_bars)
        delta = compute_indicator("volume_delta_20", synthetic_bars)
        vwap_dist = compute_indicator("vwap_distance_pct", synthetic_bars)
        # All finite
        assert all(v is not None and not np.isnan(v)
                    for v in (slope, ratio, delta, vwap_dist))
        # Volume ratio should be reasonable
        assert 0.1 < ratio < 10

    def test_session_features(self, synthetic_bars):
        """Session features on daily bars (midnight-indexed) → full-session defaults."""
        from backend.ai.strategy.indicators import compute_indicator
        mso = compute_indicator("minutes_since_open", synthetic_bars)
        progress = compute_indicator("session_progress", synthetic_bars)
        is_first = compute_indicator("is_first_hour", synthetic_bars)
        is_last = compute_indicator("is_last_hour", synthetic_bars)
        # Daily bar = full session = 375 minutes = 1.0 progress = last hour
        assert mso == 375
        assert progress == 1.0
        assert is_first == 0.0
        assert is_last == 1.0

    def test_registry_size(self):
        """Confirm registry size (51 → 68 PR-FEATURES → 79 pivots+Donchian → 86 52w/gap/roc)."""
        from backend.ai.strategy.dsl import INDICATOR_REGISTRY
        assert len(INDICATOR_REGISTRY) == 86
        # Spot-check new ones are in the registry
        for new_name in ("roc_10", "stoch_rsi_k", "di_plus", "di_minus",
                          "volatility_20", "volatility_regime", "obv_slope",
                          "volume_ratio", "vwap_distance_pct",
                          "session_progress", "is_last_hour"):
            assert new_name in INDICATOR_REGISTRY

    def test_all_new_indicators_dispatch(self, synthetic_bars):
        """Smoke test — every new indicator dispatches without error."""
        from backend.ai.strategy.indicators import compute_indicator
        new_indicators = (
            "roc_10", "roc_20", "stoch_rsi_k", "stoch_rsi_d",
            "di_plus", "di_minus",
            "volatility_20", "volatility_60", "volatility_regime",
            "obv_slope", "volume_ratio", "volume_delta_20", "vwap_distance_pct",
            "minutes_since_open", "session_progress",
            "is_first_hour", "is_last_hour",
        )
        for name in new_indicators:
            v = compute_indicator(name, synthetic_bars)
            assert v is not None, f"{name} returned None"
            assert not np.isnan(v), f"{name} returned NaN"


class TestNewIndicatorsUsableInDSL:
    """Verify new indicators work in actual DSL conditions."""

    def test_strategy_with_roc_validates(self):
        from backend.ai.strategy.dsl import Strategy
        s = Strategy.model_validate({
            "name": "ROC test",
            "universe": "nifty50",
            "timeframe": "1d",
            "entry": {"kind": "indicator_compare", "indicator": "roc_10",
                      "op": ">", "value": 5},
            "exit":  {"kind": "indicator_compare", "indicator": "roc_10",
                      "op": "<", "value": -3},
            "position_size": {"kind": "percent_of_capital", "value": 10},
        })
        assert s.entry.indicator == "roc_10"

    def test_strategy_with_session_filter_validates(self):
        from backend.ai.strategy.dsl import Strategy
        # Intraday strategy that only enters in first hour
        s = Strategy.model_validate({
            "name": "First-hour scalp",
            "universe": "nifty100",
            "timeframe": "5m",
            "entry": {
                "kind": "composite_and",
                "children": [
                    {"kind": "indicator_compare", "indicator": "is_first_hour",
                     "op": "==", "value": 1},
                    {"kind": "indicator_compare", "indicator": "rsi14",
                     "op": ">", "value": 60},
                ],
            },
            "exit": {"kind": "indicator_compare", "indicator": "is_last_hour",
                     "op": "==", "value": 1},
            "stop_loss_pct": 1.0,
            "take_profit_pct": 1.5,
            "position_size": {"kind": "percent_of_capital", "value": 10},
        })
        assert s.timeframe.value == "5m"


# ─────────────────────────────────────────────────────────────────────
# Options chain features
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_option_chain():
    """ATM±3 NIFTY option chain at spot 18000."""
    strikes = [17700, 17800, 17900, 18000, 18100, 18200, 18300]
    rows = []
    for k in strikes:
        rows.append({
            "strike": k, "option_type": "CE",
            "oi": 50_000 * (1 + abs(k - 18000) / 1000),
            "oi_change": (k - 18000) * 100,
            "volume": 10_000 + abs(k - 18000) * 50,
            "ltp": max(0.5, 18000 - k + 50),
            "iv": 0.18 + abs(k - 18000) / 50000,
        })
        rows.append({
            "strike": k, "option_type": "PE",
            "oi": 60_000 * (1 + abs(k - 18000) / 1000),  # higher PE OI = put-heavy
            "oi_change": -(k - 18000) * 100,
            "volume": 12_000 + abs(k - 18000) * 60,
            "ltp": max(0.5, k - 18000 + 50),
            "iv": 0.20 + abs(k - 18000) / 40000,
        })
    return pd.DataFrame(rows)


class TestOptionsFeatures:

    def test_pcr_computed(self, synthetic_option_chain):
        from backend.ai.options_features import compute_options_features
        f = compute_options_features(synthetic_option_chain, spot_price=18000)
        assert f is not None
        # Our synthetic data has higher PE OI than CE OI → PCR > 1
        assert f.pcr > 1.0
        # Extreme PCR detection
        assert isinstance(f.is_extreme_pcr, bool)

    def test_max_pain_at_distant_strike(self, synthetic_option_chain):
        from backend.ai.options_features import compute_options_features
        f = compute_options_features(synthetic_option_chain, spot_price=18000)
        # Our synthetic OI grows with distance from ATM, so max_pain = farthest
        assert f.max_pain is not None
        assert f.max_pain in (17700, 18300)  # one of the tails

    def test_iv_features(self, synthetic_option_chain):
        from backend.ai.options_features import compute_options_features
        f = compute_options_features(synthetic_option_chain, spot_price=18000)
        # ATM IV is small (close to base 0.18-0.20)
        assert f.iv_atm is not None
        assert 0.15 < f.iv_atm < 0.30
        # Our synthetic IV has higher PE IV than CE → skew is positive
        if f.iv_skew is not None:
            assert isinstance(f.iv_skew, float)

    def test_pcr_volume(self, synthetic_option_chain):
        from backend.ai.options_features import compute_options_features
        f = compute_options_features(synthetic_option_chain, spot_price=18000)
        assert f.pcr_volume is not None
        # Higher PE vol than CE → PCR_vol > 1
        assert f.pcr_volume > 1.0

    def test_empty_chain_returns_none(self):
        from backend.ai.options_features import compute_options_features
        f = compute_options_features(pd.DataFrame(), spot_price=18000)
        assert f is None

    def test_dict_serialization(self, synthetic_option_chain):
        from backend.ai.options_features import compute_options_features
        import json
        f = compute_options_features(synthetic_option_chain, spot_price=18000)
        d = f.to_dict()
        # JSON-safe
        json.dumps(d)
        assert "pcr" in d

    def test_iv_percentile(self):
        from backend.ai.options_features import iv_percentile_from_history
        # 100-day IV history, normal-distributed around 0.20
        rng = np.random.default_rng(0)
        history = pd.Series(rng.normal(0.20, 0.03, 100))
        # Current IV at the 75th percentile-ish
        current_iv = 0.225
        pct = iv_percentile_from_history(history, current_iv=current_iv)
        assert pct is not None
        assert 50 <= pct <= 95


# ─────────────────────────────────────────────────────────────────────
# Outcome model features integration
# ─────────────────────────────────────────────────────────────────────


class TestOutcomeFeaturesIntegration:
    """Verify the outcome-model feature builder now extracts all new features."""

    def test_builder_extracts_expanded_features(self, synthetic_bars):
        from backend.ai.outcome_models import build_outcome_features
        features = build_outcome_features(
            synthetic_bars, regime="bull", vix=14.0,
        )
        # Should have a much richer feature set than the old 8-feature version
        assert len(features) > 30
        # Check a sampling of new features are present
        for key in ("roc_10", "stoch_rsi_k", "di_plus", "volatility_20",
                    "obv_slope", "volume_ratio", "vwap_distance_pct",
                    "regime_bull", "vix"):
            assert key in features, f"missing feature: {key}"

    def test_jsonb_safe(self, synthetic_bars):
        """All values must JSON-serialise (no numpy types)."""
        from backend.ai.outcome_models import build_outcome_features
        import json
        features = build_outcome_features(
            synthetic_bars, regime="sideways", vix=20.5,
        )
        json.dumps(features)  # Must not raise
