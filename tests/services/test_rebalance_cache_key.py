from backend.api.portfolio_doctor_routes import _rebal_cache_key


class _P:
    def __init__(self, symbol, weight):
        self.symbol = symbol
        self.weight = weight


def test_rebal_cache_key_format_and_stability():
    a = [_P("TCS", 0.5), _P("INFY", 0.5)]
    b = [_P("INFY", 0.5), _P("TCS", 0.5)]   # order-independent
    ka, kb = _rebal_cache_key(a), _rebal_cache_key(b)
    assert ka == kb
    assert ka.startswith("rebal:")
    parts = ka.split(":")
    assert len(parts) == 3 and len(parts[1]) == 16 and len(parts[2]) == 10


def test_rebal_cache_key_changes_with_weights():
    a = [_P("TCS", 0.5), _P("INFY", 0.5)]
    c = [_P("TCS", 0.6), _P("INFY", 0.4)]
    assert _rebal_cache_key(a) != _rebal_cache_key(c)
