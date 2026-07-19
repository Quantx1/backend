"""AI Trading Coach behavioral flags — pure detectors (no DB, no network)."""
from types import SimpleNamespace

from backend.services.portfolio.coach_flags import (
    coach_flags,
    coach_review,
    detect_loss_holding,
    detect_overtrading,
    detect_revenge,
)


def _rec(opened, closed, pnl, symbol="TCS"):
    """Closed-trade row shape the loader produces (executed_at = open time)."""
    return {"symbol": symbol, "net_pnl": pnl, "executed_at": opened,
            "created_at": opened, "closed_at": closed}


def _clean_day(day, pnl1, pnl2):
    """Two well-spaced 50-min trades on one day; reopen 80 min after a close."""
    return [
        _rec(f"{day}T09:20:00+05:30", f"{day}T10:10:00+05:30", pnl1),
        _rec(f"{day}T11:30:00+05:30", f"{day}T12:20:00+05:30", pnl2),
    ]


# ---------------------------------------------------------------- revenge ----

def test_revenge_fires_on_quick_reopens_after_losses():
    records = [
        _rec("2026-06-01T09:20:00+05:30", "2026-06-01T10:00:00+05:30", -100.0),
        _rec("2026-06-01T10:10:00+05:30", "2026-06-01T11:00:00+05:30", 50.0),   # 10 min after loss
        _rec("2026-06-01T11:30:00+05:30", "2026-06-01T12:00:00+05:30", -80.0),
        _rec("2026-06-01T12:15:00+05:30", "2026-06-01T13:00:00+05:30", 20.0),   # 15 min after loss
    ]
    r = detect_revenge(records, window_minutes=30)
    assert r["flagged"] is True
    assert r["occasions"] == 2
    assert r["evaluable"] == 4


def test_revenge_not_flagged_on_single_occasion_or_after_wins():
    records = [
        _rec("2026-06-01T09:20:00+05:30", "2026-06-01T10:00:00+05:30", 100.0),  # win close
        _rec("2026-06-01T10:05:00+05:30", "2026-06-01T11:00:00+05:30", -50.0),  # quick reopen after WIN
        _rec("2026-06-01T11:10:00+05:30", "2026-06-01T11:40:00+05:30", 30.0),   # 10 min after the loss — 1 occasion
    ]
    r = detect_revenge(records, window_minutes=30)
    assert r["flagged"] is False
    assert r["occasions"] == 1


def test_revenge_honest_fallback_when_timestamps_missing():
    records = [{"symbol": "TCS", "net_pnl": -100.0},
               {"symbol": "INFY", "net_pnl": 50.0}]
    r = detect_revenge(records)
    assert r["flagged"] is False
    assert r["occasions"] == 0
    assert r["evaluable"] == 0


# ------------------------------------------------------------ overtrading ----

def test_overtrading_fires_on_spike_day():
    records = []
    # 4 calm days, 2 trades each (trailing median 2)…
    for d in ("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"):
        records += _clean_day(d, 10.0, -10.0)
    # …then an 8-trade spike day (>= 4 trades and >= 2x median).
    for h in range(8):
        records.append(_rec(f"2026-06-05T{9 + h % 6:02d}:30:00+05:30",
                            "2026-06-05T15:15:00+05:30", 5.0))
    r = detect_overtrading(records)
    assert r["flagged"] is True
    assert r["spike_days"] == [{"day": "2026-06-05", "trades": 8, "trailing_median": 2.0}]
    assert r["days_observed"] == 5


def test_overtrading_needs_history_and_min_trades():
    # Spike on day 2 — only 1 prior day, median is meaningless → no flag.
    records = _clean_day("2026-06-01", 10.0, -10.0)
    for h in range(6):
        records.append(_rec(f"2026-06-02T{9 + h:02d}:30:00+05:30",
                            "2026-06-02T15:15:00+05:30", 5.0))
    assert detect_overtrading(records)["flagged"] is False


# ----------------------------------------------------------- loss holding ----

def test_loss_holding_fires_on_skewed_holds():
    records = []
    for d in ("2026-06-01", "2026-06-02", "2026-06-03"):
        records.append(_rec(f"{d}T09:20:00+05:30", f"{d}T10:20:00+05:30", 100.0))   # win, 60 min
        records.append(_rec(f"{d}T10:00:00+05:30", f"{d}T15:00:00+05:30", -100.0))  # loss, 300 min
    r = detect_loss_holding(records)
    assert r["flagged"] is True
    assert r["avg_winner_hold_min"] == 60.0
    assert r["avg_loser_hold_min"] == 300.0
    assert r["ratio"] == 5.0


def test_loss_holding_needs_both_populations():
    # Only 2 losers — never flagged regardless of skew.
    records = [
        _rec("2026-06-01T09:20:00+05:30", "2026-06-01T10:20:00+05:30", 100.0),
        _rec("2026-06-01T09:20:00+05:30", "2026-06-01T10:20:00+05:30", 80.0),
        _rec("2026-06-02T09:20:00+05:30", "2026-06-02T10:20:00+05:30", 60.0),
        _rec("2026-06-02T09:20:00+05:30", "2026-06-02T15:20:00+05:30", -50.0),
        _rec("2026-06-03T09:20:00+05:30", "2026-06-03T15:20:00+05:30", -40.0),
    ]
    r = detect_loss_holding(records)
    assert r["flagged"] is False
    assert r["winners"] == 3 and r["losers"] == 2


# ------------------------------------------------------------- coach_flags ----

def test_coach_flags_honest_empty_below_ten_trades():
    records = [_rec("2026-06-01T09:20:00+05:30", "2026-06-01T10:00:00+05:30", -10.0)] * 9
    assert coach_flags(records) == {"flags": [], "stats": {"n": 9}}


def test_coach_flags_clean_records_produce_no_flags():
    records = []
    for i, d in enumerate(("2026-06-01", "2026-06-02", "2026-06-03",
                           "2026-06-04", "2026-06-05", "2026-06-08")):
        # Alternate which slot wins; identical 50-min holds; 80-min reopen gaps.
        records += _clean_day(d, 100.0 if i % 2 else -100.0, -100.0 if i % 2 else 100.0)
    out = coach_flags(records)
    assert out["flags"] == []
    assert out["stats"]["n"] == 12
    assert out["stats"]["revenge"]["flagged"] is False
    assert out["stats"]["overtrading"]["flagged"] is False
    assert out["stats"]["loss_holding"]["flagged"] is False


def test_coach_flags_surfaces_all_three_keys():
    records = []
    for d in ("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"):
        records.append(_rec(f"{d}T09:20:00+05:30", f"{d}T10:00:00+05:30", 100.0))   # win, 40 min
        records.append(_rec(f"{d}T11:30:00+05:30", f"{d}T15:00:00+05:30", -100.0))  # loss, 210 min
    # Spike day: 8 trades, two of them opened minutes after losing closes.
    records.append(_rec("2026-06-05T09:20:00+05:30", "2026-06-05T10:00:00+05:30", -50.0))
    records.append(_rec("2026-06-05T10:10:00+05:30", "2026-06-05T13:30:00+05:30", -60.0))  # revenge 1
    records.append(_rec("2026-06-05T13:40:00+05:30", "2026-06-05T15:00:00+05:30", 20.0))   # revenge 2
    for h in range(5):
        records.append(_rec(f"2026-06-05T{10 + h}:45:00+05:30",
                            f"2026-06-05T{11 + h}:25:00+05:30", 10.0))
    out = coach_flags(records)
    keys = {f["key"] for f in out["flags"]}
    assert {"revenge_trading", "overtrading", "holding_losers"} <= keys
    for f in out["flags"]:
        assert f["label"] and f["detail"]


# ------------------------------------------------------------ coach_review ----

class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        return SimpleNamespace(data=self._rows)


class _FakeSB:
    def __init__(self, rows):
        self._rows = rows

    def table(self, _name):
        return _FakeQuery(self._rows)


def test_coach_review_honest_empty_when_db_unavailable(monkeypatch):
    import backend.core.database as db

    def _boom():
        raise RuntimeError("no db in tests")

    monkeypatch.setattr(db, "get_supabase_admin", _boom)
    out = coach_review("u-1", use_llm=True)
    assert out == {"flags": [], "stats": {"n": 0}, "narrative": None}


def test_coach_review_narrates_flags_with_daily_cache_key(monkeypatch):
    import backend.ai.agents.grounded as g
    import backend.core.database as db

    rows = []
    for d in ("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"):
        rows.append({"symbol": "TCS", "net_pnl": 100.0, "status": "closed",
                     "executed_at": f"{d}T09:20:00+05:30", "created_at": f"{d}T09:20:00+05:30",
                     "closed_at": f"{d}T10:20:00+05:30"})
        rows.append({"symbol": "INFY", "net_pnl": -100.0, "status": "closed",
                     "executed_at": f"{d}T10:00:00+05:30", "created_at": f"{d}T10:00:00+05:30",
                     "closed_at": f"{d}T15:00:00+05:30"})
        # Revenge reopen 10 min after each losing close.
        rows.append({"symbol": "SBIN", "net_pnl": 10.0, "status": "closed",
                     "executed_at": f"{d}T15:10:00+05:30", "created_at": f"{d}T15:10:00+05:30",
                     "closed_at": f"{d}T15:25:00+05:30"})
    rows.append({"symbol": "HDFC", "net_pnl": None, "status": "closed"})  # skipped (no pnl)

    monkeypatch.setattr(db, "get_supabase_admin", lambda: _FakeSB(rows))
    seen = {}

    def _fake_grounded(facts, question, **kwargs):
        seen["facts"] = facts
        seen["question"] = question
        seen.update(kwargs)
        return "coached"

    monkeypatch.setattr(g, "grounded_reason", _fake_grounded)

    out = coach_review("u-42", use_llm=True)
    assert out["narrative"] == "coached"
    assert any(f["key"] == "holding_losers" for f in out["flags"])
    assert seen["cache_key"].startswith("coach:u-42:")
    assert seen["user_id"] == "u-42"
    assert "Coach this trader" in seen["question"]
    assert seen["facts"]["flags"] == out["flags"]

    # Deterministic flags come back with 0 tokens when use_llm is off.
    out2 = coach_review("u-42", use_llm=False)
    assert out2["narrative"] is None
    assert out2["flags"] == out["flags"]


def test_coach_review_skips_llm_when_no_flags(monkeypatch):
    import backend.ai.agents.grounded as g
    import backend.core.database as db

    rows = []
    for i, d in enumerate(("2026-06-01", "2026-06-02", "2026-06-03",
                           "2026-06-04", "2026-06-05", "2026-06-08")):
        for opened, closed in (("09:20", "10:10"), ("11:30", "12:20")):
            rows.append({"symbol": "TCS", "net_pnl": 100.0 if i % 2 else -100.0,
                         "status": "closed",
                         "executed_at": f"{d}T{opened}:00+05:30",
                         "created_at": f"{d}T{opened}:00+05:30",
                         "closed_at": f"{d}T{closed}:00+05:30"})
    monkeypatch.setattr(db, "get_supabase_admin", lambda: _FakeSB(rows))

    def _never(*_a, **_k):
        raise AssertionError("grounded_reason must not be called without flags")

    monkeypatch.setattr(g, "grounded_reason", _never)
    out = coach_review("u-7", use_llm=True)
    assert out["flags"] == []
    assert out["narrative"] is None
