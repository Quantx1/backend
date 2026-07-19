"""Unit tests for strategy registry state machine — PR-F.

Tests the pure-Python state transition logic. Does NOT hit Supabase —
those CRUD ops are integration-tested via the API endpoint tests in
tests/test_strategies_routes.py (added separately).
"""

from __future__ import annotations

import pytest

from backend.ai.strategy.registry import (
    StrategyStateError,
    allowed_transitions,
    is_terminal,
    validate_transition,
)


class TestStateMachine:

    def test_draft_can_go_to_backtest_paper_archived(self):
        assert allowed_transitions("draft") == {"backtest", "paper", "archived"}

    def test_backtest_can_go_to_draft_paper_live_archived(self):
        assert allowed_transitions("backtest") == {"draft", "paper", "live", "archived"}

    def test_paper_can_go_to_paused_live_archived(self):
        assert allowed_transitions("paper") == {"paused", "live", "archived"}

    def test_live_can_only_pause_or_archive(self):
        assert allowed_transitions("live") == {"paused", "archived"}

    def test_paused_can_resume_paper_or_live(self):
        assert allowed_transitions("paused") == {"paper", "live", "archived"}

    def test_archived_is_terminal(self):
        assert is_terminal("archived")
        assert allowed_transitions("archived") == set()

    def test_idempotent_same_state(self):
        # No-op transitions don't raise
        validate_transition("draft", "draft")
        validate_transition("live", "live")

    def test_invalid_transition_raises(self):
        with pytest.raises(StrategyStateError):
            validate_transition("draft", "live")  # must go through backtest/paper first
        with pytest.raises(StrategyStateError):
            validate_transition("live", "draft")
        with pytest.raises(StrategyStateError):
            validate_transition("archived", "draft")

    def test_paper_to_live_allowed(self):
        validate_transition("paper", "live")  # no raise

    def test_live_to_paused_allowed(self):
        validate_transition("live", "paused")  # no raise
