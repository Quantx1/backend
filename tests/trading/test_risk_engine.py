import pytest
from backend.ai.signals.style_types import Style
from backend.trading.risk_engine import derive_levels, RISK_PARAMS


def test_momentum_buy_levels_from_atr():
    entry, sl, target, rr = derive_levels("BUY", ref_price=100.0, atr=10.0, style=Style.MOMENTUM)
    assert entry == 100.0
    assert sl == 85.0      # 100 - 1.5*10
    assert target == 130.0 # 100 + 3.0*10
    assert rr == pytest.approx(2.0)  # (130-100)/(100-85)


def test_momentum_params_present():
    assert Style.MOMENTUM in RISK_PARAMS
    sl_mult, tp_mult = RISK_PARAMS[Style.MOMENTUM]
    assert (sl_mult, tp_mult) == (1.5, 3.0)


def test_rejects_nonpositive_atr():
    with pytest.raises(ValueError):
        derive_levels("BUY", ref_price=100.0, atr=0.0, style=Style.MOMENTUM)


def test_rejects_nonpositive_ref_price():
    with pytest.raises(ValueError):
        derive_levels("BUY", ref_price=0.0, atr=10.0, style=Style.MOMENTUM)


def test_rejects_unknown_style():
    with pytest.raises(ValueError):
        derive_levels("SELL", ref_price=100.0, atr=10.0, style=Style.MOMENTUM)
