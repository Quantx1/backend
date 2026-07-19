import backend.middleware.llm_caps as caps
from backend.core.tiers import Tier


def _limiter():
    return caps.LlmFeatureLimiter(supabase_client=None)   # in-memory only


def test_consume_until_cap_then_denies():
    lim = _limiter()
    for i in range(10):
        allowed, used, cap = lim.consume("u1", "debate", Tier.ELITE)
        assert allowed is True
        assert cap == 10
    allowed, used, cap = lim.consume("u1", "debate", Tier.ELITE)
    assert allowed is False
    assert used == 10


def test_zero_cap_tier_is_denied_immediately():
    lim = _limiter()
    allowed, used, cap = lim.consume("u2", "debate", Tier.PRO)   # Pro debate cap = 0
    assert allowed is False
    assert cap == 0


def test_separate_users_have_separate_counters():
    lim = _limiter()
    for _ in range(5):
        lim.consume("a", "chat", Tier.FREE)   # chat free cap = 5
    allowed_a, _, _ = lim.consume("a", "chat", Tier.FREE)
    allowed_b, _, _ = lim.consume("b", "chat", Tier.FREE)
    assert allowed_a is False
    assert allowed_b is True


def test_window_key_is_month_for_portfolio_doctor():
    lim = _limiter()
    wk = lim._window_key("portfolio_doctor")
    assert len(wk) == 7 and wk[4] == "-"        # YYYY-MM
    assert len(lim._window_key("debate")) == 10  # YYYY-MM-DD
