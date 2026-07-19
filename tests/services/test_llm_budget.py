"""LLM monthly budget meter + kill-switch tests.

The meter is a fast in-process month-to-date counter (baseline reconciled
from llm_usage_events + pending added per call). The kill-switch refuses
PAID calls once the monthly spend crosses the budget. Pure tests — a fake
supabase stub feeds the DB reconcile path.
"""

from __future__ import annotations

import pytest

from backend.observability.llm_budget import BudgetExceededError, UsageMeter
from backend.observability.llm_pricing import is_paid


class TestPricing:
    def test_paid_vs_free_models(self):
        assert is_paid("openrouter", "qwen/qwen3-32b") is True
        assert is_paid("openrouter", "qwen/qwen3-coder:free") is False    # free tier
        assert is_paid("openrouter", "openai/gpt-oss-120b:free") is False  # free tier
        assert is_paid("openrouter", "unknown-model") is False            # unpriced


class TestMeter:
    def test_records_and_reports_spend(self):
        m = UsageMeter()
        m.record_micros(5_000_000)   # $5
        assert m.spent_usd() == 5.0

    def test_over_budget_threshold(self):
        m = UsageMeter()
        m.record_micros(20_000_000)  # $20
        assert m.over_budget(20.0) is True   # >= budget
        assert m.over_budget(25.0) is False

    def test_enforce_raises_when_over(self):
        m = UsageMeter()
        m.record_micros(21_000_000)
        with pytest.raises(BudgetExceededError):
            m.enforce(20.0)

    def test_enforce_ok_when_under(self):
        m = UsageMeter()
        m.record_micros(1_000_000)
        m.enforce(20.0)  # must not raise

    def test_negative_micros_ignored(self):
        m = UsageMeter()
        m.record_micros(-5)
        assert m.spent_micros() == 0


class TestMonthRollover:
    def test_stale_month_resets_accumulator(self):
        m = UsageMeter()
        m.record_micros(20_000_000)
        # Force a stale month → next access rolls over to a clean slate.
        m._month_key = "1999-01"
        m._baseline_micros = 9_999
        assert m.spent_micros() == 0  # old month wiped on roll


class TestDbReconcile:
    def test_refresh_sets_baseline_from_db(self):
        rows = [{"micros_usd": 3_000_000}, {"micros_usd": 4_000_000}]

        class _Exec:
            data = rows

        class _Q:
            def select(self, *_a, **_k):
                return self

            def gte(self, *_a, **_k):
                return self

            def limit(self, *_a, **_k):
                return self

            def execute(self):
                return _Exec()

        class _SB:
            def table(self, _n):
                return _Q()

        m = UsageMeter()
        m.refresh_from_db(_SB())
        assert m.spent_usd() == 7.0  # 3 + 4

    def test_refresh_failure_is_safe(self):
        class _SB:
            def table(self, _n):
                raise RuntimeError("db down")

        m = UsageMeter()
        m.record_micros(2_000_000)
        m.refresh_from_db(_SB())   # must not raise
        assert m.spent_usd() == 2.0  # keeps prior pending
