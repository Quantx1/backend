import backend.ai.agents.copilot as cp


def _tr(result, tool="get_current_regime"):
    return [{"tool": tool, "args": {}, "result": result}]


def test_grounded_when_numbers_match():
    out = cp.validate_grounding("NIFTY at 24,850, VIX 13.2",
                                _tr({"nifty_close": 24850.0, "vix": 13.2}))
    assert out["grounded"] is True
    assert out["unsupported"] == []


def test_flags_fabricated_number():
    out = cp.validate_grounding("Target is 27,500 next week",
                                _tr({"nifty_close": 24850.0}))
    assert out["grounded"] is False
    assert "27,500" in out["unsupported"]


def test_probability_rendered_as_percent_is_grounded():
    out = cp.validate_grounding("62% chance of upside", _tr({"prob_bull": 0.62}))
    assert out["grounded"] is True


def test_rounding_tolerance():
    out = cp.validate_grounding("around 2,543",
                                _tr({"last_close": 2543.55}, tool="get_stock_snapshot"))
    assert out["grounded"] is True


def test_pure_knowledge_no_tools_is_grounded():
    out = cp.validate_grounding("RSI above 70 signals overbought", [])
    assert out["grounded"] is True


def test_safe_small_ints_not_flagged():
    out = cp.validate_grounding("Look at the last 3 months, top 5 names",
                                _tr({"nifty_close": 24850.0}))
    assert out["grounded"] is True


def test_zero_tool_reply_with_fake_citation_is_flagged():
    # A reply that cites [tool:...] when NO tools ran is fabricating evidence.
    out = cp.validate_grounding("NIFTY at 22,350 [tool:market_breadth]", [])
    assert out["grounded"] is False
    assert any("[tool:]" in u for u in out["unsupported"])


def test_zero_tool_reply_without_citation_still_bypasses():
    out = cp.validate_grounding("RSI above 70 signals overbought", [])
    assert out["grounded"] is True


def test_needs_tools_catches_live_market_formulations():
    for q in (
        "How is the market regime today and what should a swing trader do?",
        "how's the market looking",
        "what regime are we in",
        "where is the vix",
        "nifty support today?",
        "market now?",
    ):
        assert cp._NEEDS_TOOLS.search(q), q


def test_needs_tools_still_ignores_pure_chitchat():
    for q in ("explain rsi to me", "what is a moving average", "thanks!"):
        assert not cp._NEEDS_TOOLS.search(q), q
