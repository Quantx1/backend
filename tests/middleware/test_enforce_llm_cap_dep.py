import asyncio

import pytest
from fastapi import HTTPException

import backend.middleware.llm_caps as caps
from backend.core.tiers import Tier


class _FakeUser:
    def __init__(self, tier, is_admin=False):
        self.user_id = "u-test"
        self.tier = tier
        self.is_admin = is_admin


def _run(dep, user):
    return asyncio.run(dep(user=user))


def test_dependency_allows_under_cap_then_402(monkeypatch):
    monkeypatch.setattr(caps, "_limiter", caps.LlmFeatureLimiter(None))
    dep = caps.enforce_llm_cap("debate")            # Elite cap = 10
    user = _FakeUser(Tier.ELITE)
    for _ in range(10):
        assert _run(dep, user) is user
    with pytest.raises(HTTPException) as ei:
        _run(dep, user)
    assert ei.value.status_code == 402
    assert ei.value.detail["feature"] == "debate"


def test_admin_bypasses_cap(monkeypatch):
    monkeypatch.setattr(caps, "_limiter", caps.LlmFeatureLimiter(None))
    dep = caps.enforce_llm_cap("debate")
    admin = _FakeUser(Tier.FREE, is_admin=True)      # cap 0 but admin bypasses
    assert _run(dep, admin) is admin
