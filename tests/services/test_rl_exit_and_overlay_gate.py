"""Batch-1 safety: RL early-exit consult + event-risk overlay gate.

- enforce_stops_for_user is DORMANT (zeros, no writes) until a trained
  Q-table is loaded; when active it RATCHETS stops on EXIT (never loosens),
  writing the trade row monitor_positions reads.
- apply_ai_overlay blocks NEW entries for earnings-window symbols (entry-only).
"""
import backend.ai.exit_engine.rl_exit_scaffold as rlmod
from backend.trading.risk import RiskManagementEngine
from backend.services.strategy_runner.ai_overlay import (
    AIOverlaySettings,
    apply_ai_overlay,
)


# ── fake Supabase that records writes ──────────────────────────────────────


class _Q:
    def __init__(self, store, table):
        self.store = store
        self.table_name = table
        self._select = None
        self._update = None

    def select(self, *a, **k):
        return self

    def update(self, payload):
        self._update = payload
        return self

    def eq(self, col, val):
        self._eq = (col, val)
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        if self._update is not None:
            self.store["writes"].append((self.table_name, self._update))
            return type("R", (), {"data": []})()
        return type("R", (), {"data": self.store["positions"]})()


class _SB:
    def __init__(self, positions):
        self.store = {"positions": positions, "writes": []}

    def table(self, name):
        return _Q(self.store, name)


class _FakeAgent:
    def __init__(self, action, loaded=True):
        self._action = action
        self.is_loaded = loaded

    def decide(self, state, **k):
        return self._action


async def _run(engine):
    return await engine.enforce_stops_for_user("u1")


# ── dormant ───────────────────────────────────────────────────────────────


async def test_dormant_when_agent_not_loaded(monkeypatch):
    monkeypatch.setattr(rlmod, "ENABLE_RL_EXIT", True)
    monkeypatch.setattr(rlmod, "get_rl_exit_agent", lambda: _FakeAgent("EXIT", loaded=False))
    sb = _SB([{"id": "p1", "trade_id": "t1", "symbol": "TCS", "average_price": 100,
               "direction": "LONG", "trades": {"stop_loss": 90, "target": 120}}])
    engine = RiskManagementEngine(sb)
    out = await _run(engine)
    assert out == {"exits_emitted": 0, "alerts_fired": 0}
    assert sb.store["writes"] == []  # no ratchet


async def test_dormant_when_flag_off(monkeypatch):
    monkeypatch.setattr(rlmod, "ENABLE_RL_EXIT", False)
    monkeypatch.setattr(rlmod, "get_rl_exit_agent", lambda: _FakeAgent("EXIT", loaded=True))
    engine = RiskManagementEngine(_SB([]))
    assert await _run(engine) == {"exits_emitted": 0, "alerts_fired": 0}


# ── active ────────────────────────────────────────────────────────────────


async def test_exit_ratchets_stop_to_current_on_both_tables(monkeypatch):
    monkeypatch.setattr(rlmod, "ENABLE_RL_EXIT", True)
    monkeypatch.setattr(rlmod, "get_rl_exit_agent", lambda: _FakeAgent("EXIT", loaded=True))
    sb = _SB([{"id": "p1", "trade_id": "t1", "symbol": "TCS", "average_price": 100,
               "direction": "LONG", "trades": {"stop_loss": 90, "target": 120}}])
    engine = RiskManagementEngine(sb)

    async def _price(sym):
        return 110.0
    monkeypatch.setattr(engine, "_latest_price", _price)

    out = await _run(engine)
    assert out["exits_emitted"] == 1
    tables = {t for t, _ in sb.store["writes"]}
    assert tables == {"positions", "trades"}  # trade row is what monitor reads
    for _, payload in sb.store["writes"]:
        assert payload["stop_loss"] == 110.0  # ratcheted UP to current


async def test_hold_does_nothing(monkeypatch):
    monkeypatch.setattr(rlmod, "ENABLE_RL_EXIT", True)
    monkeypatch.setattr(rlmod, "get_rl_exit_agent", lambda: _FakeAgent("HOLD", loaded=True))
    sb = _SB([{"id": "p1", "trade_id": "t1", "symbol": "TCS", "average_price": 100,
               "direction": "LONG", "trades": {"stop_loss": 90, "target": 120}}])
    engine = RiskManagementEngine(sb)

    async def _price(sym):
        return 110.0
    monkeypatch.setattr(engine, "_latest_price", _price)

    out = await _run(engine)
    assert out == {"exits_emitted": 0, "alerts_fired": 0}
    assert sb.store["writes"] == []


async def test_exit_never_loosens_stop(monkeypatch):
    # current price BELOW the existing stop → ratchet-to-current would loosen;
    # must be refused (the hard stop already protects).
    monkeypatch.setattr(rlmod, "ENABLE_RL_EXIT", True)
    monkeypatch.setattr(rlmod, "get_rl_exit_agent", lambda: _FakeAgent("EXIT", loaded=True))
    sb = _SB([{"id": "p1", "trade_id": "t1", "symbol": "TCS", "average_price": 100,
               "direction": "LONG", "trades": {"stop_loss": 95, "target": 120}}])
    engine = RiskManagementEngine(sb)

    async def _price(sym):
        return 92.0  # below stop
    monkeypatch.setattr(engine, "_latest_price", _price)

    out = await _run(engine)
    assert out["exits_emitted"] == 0
    assert sb.store["writes"] == []


# ── overlay event gate ─────────────────────────────────────────────────────


def test_overlay_blocks_entry_in_event_window():
    d = apply_ai_overlay(
        supabase=None, settings=AIOverlaySettings(), user_id="u1", symbol="TCS",
        current_vix=12.0, current_regime="bull", event_blackout={"TCS"},
    )
    assert d.allowed is False
    assert d.block_reason == "event_risk:earnings"


def test_overlay_allows_when_not_in_window():
    d = apply_ai_overlay(
        supabase=None, settings=AIOverlaySettings(), user_id="u1", symbol="INFY",
        current_vix=12.0, current_regime="bull", event_blackout={"TCS"},
    )
    assert d.allowed is True


def test_overlay_event_gate_can_be_disabled():
    s = AIOverlaySettings(event_gate_enabled=False)
    d = apply_ai_overlay(
        supabase=None, settings=s, user_id="u1", symbol="TCS",
        current_vix=12.0, current_regime="bull", event_blackout={"TCS"},
    )
    assert d.allowed is True
