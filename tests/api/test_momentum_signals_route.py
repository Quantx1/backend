from backend.ai.signals.style_types import MomentumSignal, Style


def test_momentum_route_shape(monkeypatch):
    from backend.api import signals_routes as sr

    fake = [MomentumSignal(symbol="AAA", style=Style.MOMENTUM, rank=1, percentile=1.0,
                           confidence=100.0, direction="BUY", entry_price=100.0,
                           stop_loss=85.0, target=130.0, risk_reward=2.0, reasons=["r"],
                           expected_return=0.04, top_decile_prob=1.0)]

    class _FakeEngine:
        status = "ok"
        def run(self, top_n=20, universe_limit=None):
            return fake

    monkeypatch.setattr(sr, "_momentum_engine", lambda: _FakeEngine())
    sr._momentum_cache.clear()
    payload = sr._compute_momentum(top_n=20)
    assert payload["status"] == "ok"
    assert payload["count"] == 1
    assert payload["signals"][0]["symbol"] == "AAA"
    assert payload["signals"][0]["style"] == "momentum"
    assert payload["signals"][0]["expected_return"] == 0.04
