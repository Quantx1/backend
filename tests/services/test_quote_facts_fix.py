"""Quote facts regression — `get_quote` returns a `Quote` DATACLASS (no `.get`).

why_moving/market_explainer used dict-style `q.get("...")` on it; the
AttributeError was swallowed by the blocks' broad try/excepts so the price
facts silently never populated. These tests pin attribute access working
against a dataclass-shaped quote. Pure: provider + Supabase + NSE + breadth
+ sector-rotation are all stubbed (no network/DB/LLM).
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class _FakeQuote:
    ltp: Optional[float] = 100.0
    close: Optional[float] = 98.0
    change_percent: Optional[float] = 2.04
    volume: Optional[float] = 1_000_000.0


class _FakeProvider:
    def get_quote(self, sym):
        return _FakeQuote()

    def get_historical(self, *a, **k):
        return None

    def get_quotes_batch(self, symbols):
        return {}


def _blocked(*a, **k):
    raise RuntimeError("blocked in test (pure — no network/DB)")


def _isolate(monkeypatch):
    """Fake provider in, every other external dependency honest-empty."""
    monkeypatch.setattr("backend.data.market.get_market_data_provider",
                        lambda: _FakeProvider())
    monkeypatch.setattr("backend.core.database.get_supabase_admin", _blocked)
    monkeypatch.setattr("backend.data.screener.nse_data.get_nse_data", _blocked)
    monkeypatch.setattr("backend.services.scanners.breadth.breadth", lambda: {})
    monkeypatch.setattr("backend.services.scanners.sector_rotation.sector_rotation",
                        lambda: [])


def test_why_moving_price_fact_populates_from_quote_dataclass(monkeypatch):
    import backend.services.explain.why_moving as wm
    _isolate(monkeypatch)
    facts = wm.assemble_facts("TCS")
    price = facts.get("price") or {}
    assert price.get("change_pct") == 2.04
    assert price.get("ltp") == 100.0


def test_market_explainer_nifty_fact_populates_from_quote_dataclass(monkeypatch):
    import backend.services.explain.market_explainer as me
    _isolate(monkeypatch)
    facts = me._assemble_facts()
    nifty = facts.get("nifty") or {}
    assert nifty.get("change_pct") == 2.04
    assert nifty.get("ltp") == 100.0
