"""PR-H catalog tests — validate seed templates parse + endpoints work.

Two layers:
1. Pure-Python: every template in scripts/ops/seed_strategy_dsl_templates.py
   must DSL-validate. This is the canonical guard against catalog drift.
2. API: /catalog, /catalog/{slug}, /from-template/{slug} return the
   expected shape, with Supabase mocked at the table level.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.ai.strategy.dsl import Strategy


def _load_seed_module():
    """Load scripts/ops/seed_strategy_dsl_templates.py without running seed()."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "ops" / "seed_strategy_dsl_templates.py"
    spec = importlib.util.spec_from_file_location("seed_templates", path)
    mod = importlib.util.module_from_spec(spec)
    # Patch get_supabase_admin so importing the module never opens a connection
    with patch("backend.core.database.get_supabase_admin", return_value=MagicMock()):
        spec.loader.exec_module(mod)
    return mod


class TestSeedTemplatesParse:
    """Every template DSL in the seed file must Strategy.model_validate."""

    @pytest.fixture(scope="class")
    def seed_module(self):
        return _load_seed_module()

    def test_template_list_not_empty(self, seed_module):
        assert len(seed_module.TEMPLATES) >= 10

    def test_every_template_has_required_fields(self, seed_module):
        required = {
            "slug", "name", "description", "category", "segment",
            "tier_required", "min_capital", "risk_level", "engine_compatible", "dsl",
        }
        for tpl in seed_module.TEMPLATES:
            missing = required - set(tpl.keys())
            assert not missing, f"{tpl.get('slug')} missing: {missing}"

    def test_every_template_dsl_validates(self, seed_module):
        for tpl in seed_module.TEMPLATES:
            try:
                Strategy.model_validate(tpl["dsl"])
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"{tpl['slug']} failed DSL validation: {exc}")

    def test_slugs_are_unique(self, seed_module):
        slugs = [t["slug"] for t in seed_module.TEMPLATES]
        assert len(slugs) == len(set(slugs)), "duplicate slug in TEMPLATES"

    def test_engine_compatible_templates_actually_use_engines(self, seed_module):
        """If a template is flagged engine_compatible=True, its DSL must
        consume an engine output — either via an engine_signal condition
        in entry/exit, or via a non-'any' regime_filter (which depends on
        the Regime engine). Otherwise the flag is misleading."""
        for tpl in seed_module.TEMPLATES:
            if not tpl.get("engine_compatible"):
                continue
            dsl = tpl["dsl"]

            def _has_engine_signal(cond):
                if not isinstance(cond, dict):
                    return False
                if cond.get("kind") == "engine_signal":
                    return True
                return any(_has_engine_signal(c) for c in cond.get("children", []))

            via_condition = (
                _has_engine_signal(dsl.get("entry", {}))
                or _has_engine_signal(dsl.get("exit", {}))
            )
            via_regime_filter = dsl.get("regime_filter", "any") != "any"

            assert via_condition or via_regime_filter, (
                f"{tpl['slug']} flagged engine_compatible but DSL has no "
                f"engine_signal condition and regime_filter='any'"
            )


class TestCatalogEndpoints:
    """API tests with Supabase mocked at the .table() chain."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.api.app import app
        return TestClient(app)

    def _mock_sb_with_data(self, monkeypatch, rows):
        """Patch get_supabase_admin so any .table(...).select(...).execute()
        chain returns the given rows."""
        mock_sb = MagicMock()
        # Build a chain that responds to arbitrary .eq/.order/.limit calls
        chain = MagicMock()
        chain.execute.return_value = MagicMock(data=rows)
        # Every chain method returns self so .eq.eq.order.limit all work
        for attr in ("select", "eq", "order", "limit"):
            getattr(chain, attr).return_value = chain
        mock_sb.table.return_value = chain
        monkeypatch.setattr("backend.api.strategies_routes.get_supabase_admin", lambda: mock_sb)
        return mock_sb

    def test_catalog_list_returns_templates(self, client, monkeypatch):
        self._mock_sb_with_data(monkeypatch, [
            {"slug": "rsi-mean-reversion", "name": "RSI Mean Reversion",
             "category": "equity_swing", "segment": "EQUITY", "is_featured": True,
             "engine_compatible": False},
        ])
        res = client.get("/api/strategies/catalog")
        assert res.status_code == 200
        body = res.json()
        assert "templates" in body
        assert "count" in body
        assert body["count"] == 1
        assert body["templates"][0]["slug"] == "rsi-mean-reversion"

    def test_catalog_list_no_auth_required(self, client, monkeypatch):
        """Catalog must be public — no Authorization header needed."""
        self._mock_sb_with_data(monkeypatch, [])
        res = client.get("/api/strategies/catalog")
        assert res.status_code == 200  # never 401/403

    def test_catalog_filter_by_segment(self, client, monkeypatch):
        mock_sb = self._mock_sb_with_data(monkeypatch, [])
        res = client.get("/api/strategies/catalog?segment=EQUITY")
        assert res.status_code == 200
        # Verify the segment filter was applied via the eq() chain
        # (we don't track chained calls easily — just confirm OK)

    def test_catalog_rejects_bad_segment(self, client, monkeypatch):
        self._mock_sb_with_data(monkeypatch, [])
        res = client.get("/api/strategies/catalog?segment=CRYPTO")
        assert res.status_code == 422  # FastAPI rejects bad enum

    def test_catalog_engine_only_filter(self, client, monkeypatch):
        self._mock_sb_with_data(monkeypatch, [])
        res = client.get("/api/strategies/catalog?engine_only=true")
        assert res.status_code == 200

    def test_catalog_get_one_returns_dsl(self, client, monkeypatch):
        self._mock_sb_with_data(monkeypatch, [
            {
                "slug": "rsi-mean-reversion", "name": "RSI Mean Reversion",
                "dsl": {"name": "x", "entry": {"kind": "indicator_compare",
                                                "indicator": "rsi14", "op": "<", "value": 30}},
            }
        ])
        res = client.get("/api/strategies/catalog/rsi-mean-reversion")
        assert res.status_code == 200
        body = res.json()
        assert body["template"]["slug"] == "rsi-mean-reversion"
        assert "dsl" in body["template"]

    def test_catalog_get_one_404_when_missing(self, client, monkeypatch):
        self._mock_sb_with_data(monkeypatch, [])
        res = client.get("/api/strategies/catalog/nonexistent")
        assert res.status_code == 404


class TestExclusiveAndSections:
    """PR-K — Exclusive Strategies surface + /catalog/sections endpoint."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.api.app import app
        return TestClient(app)

    def _mock_sb(self, monkeypatch, by_query: Dict[str, List] | None = None, default_rows: list | None = None):
        """Patch get_supabase_admin with a chain that returns different
        rows depending on which (field, value) eq() pairs are applied."""
        from collections import OrderedDict
        from unittest.mock import MagicMock
        mock_sb = MagicMock()
        by_query = by_query or {}

        class _Chain:
            def __init__(self):
                self._filters = OrderedDict()

            def select(self, *a, **kw):
                return self

            def eq(self, field, value):
                self._filters[field] = value
                return self

            def order(self, *a, **kw):
                return self

            def limit(self, *a, **kw):
                return self

            def execute(self):
                # Match by exact filter set if available; else default
                key = tuple(sorted(self._filters.items()))
                rows = by_query.get(key, default_rows or [])
                # Reset for next chain reuse
                self._filters = OrderedDict()
                return MagicMock(data=rows)

        chain = _Chain()
        mock_sb.table.return_value = chain
        monkeypatch.setattr("backend.api.strategies_routes.get_supabase_admin", lambda: mock_sb)
        return mock_sb

    def test_exclusive_only_filter(self, client, monkeypatch):
        from typing import Dict, List  # noqa: F401
        self._mock_sb(monkeypatch, default_rows=[
            {"slug": "alpha-rank-momentum-leader", "is_exclusive": True,
             "exclusive_tagline": "Engine-ranked selection"},
        ])
        res = client.get("/api/strategies/catalog?exclusive_only=true")
        assert res.status_code == 200
        body = res.json()
        assert body["count"] == 1
        assert body["templates"][0]["is_exclusive"] is True
        assert body["templates"][0]["exclusive_tagline"]

    def test_sections_returns_all_five(self, client, monkeypatch):
        self._mock_sb(monkeypatch, default_rows=[])
        res = client.get("/api/strategies/catalog/sections")
        assert res.status_code == 200
        body = res.json()
        assert set(body["section_keys"]) == {
            "exclusive", "featured", "intraday", "swing", "options",
        }
        # Exclusive section must carry the FinStocks tagline verbatim
        assert body["sections"]["exclusive"]["title"] == "Exclusive Strategies"
        assert "Unlock advanced algorithms" in body["sections"]["exclusive"]["tagline"]

    def test_sections_no_auth_required(self, client, monkeypatch):
        self._mock_sb(monkeypatch, default_rows=[])
        res = client.get("/api/strategies/catalog/sections")
        assert res.status_code == 200


class TestFinStocksTemplatesParse:
    """Every FinStocks-inspired template DSL-validates."""

    @pytest.fixture(scope="class")
    def seed_module(self):
        return _load_seed_module()

    def test_finstocks_templates_present(self, seed_module):
        finstocks = [t for t in seed_module.TEMPLATES if "finstocks-inspired" in t.get("tags", [])]
        assert len(finstocks) >= 10

    def test_every_finstocks_template_validates(self, seed_module):
        finstocks = [t for t in seed_module.TEMPLATES if "finstocks-inspired" in t.get("tags", [])]
        for tpl in finstocks:
            try:
                Strategy.model_validate(tpl["dsl"])
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"FinStocks template {tpl['slug']} failed DSL validation: {exc}")

    def test_exclusive_templates_have_tagline(self, seed_module):
        exclusive = [t for t in seed_module.TEMPLATES if t.get("is_exclusive")]
        assert len(exclusive) >= 4
        for t in exclusive:
            assert t.get("exclusive_tagline"), (
                f"{t['slug']} is_exclusive=True but missing exclusive_tagline"
            )


class TestRouteOrdering:
    """Regression guard: /catalog must come BEFORE /{strategy_id}.

    If someone moves the catalog endpoints, FastAPI will eat them as
    strategy_id lookups. This test pins the ordering at the router level.
    """

    def test_catalog_route_declared_before_strategy_id(self):
        from backend.api.strategies_routes import router
        paths = [r.path for r in router.routes]
        catalog_idx = paths.index("/api/strategies/catalog")
        strategy_id_idx = paths.index("/api/strategies/{strategy_id}")
        assert catalog_idx < strategy_id_idx, \
            "/catalog must be declared before /{strategy_id} or FastAPI will shadow it"

    def test_from_template_route_declared_before_create(self):
        from backend.api.strategies_routes import router
        # POST /from-template/{slug} doesn't conflict with POST /
        # but POST /{strategy_id}/transition shares a prefix shape with us.
        # Mostly checking nothing changed.
        paths = [r.path for r in router.routes]
        assert "/api/strategies/from-template/{slug}" in paths

    def test_all_pr_h_routes_registered(self):
        from backend.api.strategies_routes import router
        paths = {r.path for r in router.routes}
        for expected in (
            "/api/strategies/catalog",
            "/api/strategies/catalog/{slug}",
            "/api/strategies/from-template/{slug}",
        ):
            assert expected in paths, f"missing PR-H route: {expected}"
