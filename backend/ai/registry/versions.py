"""
Postgres helpers over ``public.model_versions``.

The table was created in PR 2 (see ``2026_04_19_pr2_v1_ai_stack.sql``).
Schema:
    id UUID, model_name TEXT, version INTEGER, artifact_uri TEXT,
    trained_at TIMESTAMPTZ, trained_by TEXT, metrics JSONB,
    git_sha TEXT, is_prod BOOLEAN, is_shadow BOOLEAN,
    is_retired BOOLEAN, notes TEXT

A partial unique index enforces at most one ``is_prod=true`` row per
model_name. ``promote()`` takes care of flipping the old prod row down
before raising the new one — see ``promote_version``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.core.database import get_supabase_admin

logger = logging.getLogger(__name__)

_TABLE = "model_versions"


def _client():
    return get_supabase_admin()


# --------------------------------------------------------------------- read


def list_versions(model_name: str) -> List[Dict[str, Any]]:
    """All non-retired versions of a model, newest-first."""
    resp = (
        _client()
        .table(_TABLE)
        .select("*")
        .eq("model_name", model_name)
        .eq("is_retired", False)
        .order("version", desc=True)
        .execute()
    )
    return resp.data or []


def get_prod(model_name: str) -> Optional[Dict[str, Any]]:
    resp = (
        _client()
        .table(_TABLE)
        .select("*")
        .eq("model_name", model_name)
        .eq("is_prod", True)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    return rows[0] if rows else None


def get_shadow(model_name: str) -> Optional[Dict[str, Any]]:
    """Latest shadow version (highest version number)."""
    resp = (
        _client()
        .table(_TABLE)
        .select("*")
        .eq("model_name", model_name)
        .eq("is_shadow", True)
        .eq("is_retired", False)
        .order("version", desc=True)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    return rows[0] if rows else None


def get_version(model_name: str, version: int) -> Optional[Dict[str, Any]]:
    resp = (
        _client()
        .table(_TABLE)
        .select("*")
        .eq("model_name", model_name)
        .eq("version", version)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    return rows[0] if rows else None


def next_version(model_name: str) -> int:
    """Next integer version for this model (1 if never seen)."""
    resp = (
        _client()
        .table(_TABLE)
        .select("version")
        .eq("model_name", model_name)
        .order("version", desc=True)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    return (rows[0]["version"] + 1) if rows else 1


# -------------------------------------------------------------------- write


def _json_safe(obj: Any) -> Any:
    """Recursively replace NaN/inf with None (JSON-compliant).

    PostgREST + Supabase reject non-finite floats in JSONB payloads.
    A trainer that emits NaN for a degenerate fold (e.g., all-constant
    predictions → NaN IC) would otherwise crash the runner before any
    model_versions row gets written. We sanitize so the row writes;
    the promote_gate then correctly flags NaN-converted-to-None as a
    failed metric and refuses promotion.
    """
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def insert_version(
    model_name: str,
    version: int,
    artifact_uri: str,
    metrics: Optional[Dict[str, Any]] = None,
    trained_by: Optional[str] = None,
    git_sha: Optional[str] = None,
    notes: Optional[str] = None,
    is_prod: bool = False,
    is_shadow: bool = False,
) -> Dict[str, Any]:
    """Insert a new model_versions row. Caller must ensure at most one
    ``is_prod=true`` row exists per model_name (use ``promote_version``)."""
    payload = {
        "model_name": model_name,
        "version": version,
        "artifact_uri": artifact_uri,
        "metrics": _json_safe(metrics or {}),
        "trained_by": trained_by,
        "git_sha": git_sha,
        "notes": notes,
        "is_prod": is_prod,
        "is_shadow": is_shadow,
        "is_retired": False,
    }
    resp = _client().table(_TABLE).insert(payload).execute()
    row = (resp.data or [None])[0]
    if row is None:
        raise RuntimeError(f"Insert into model_versions returned no row: {payload}")
    logger.info(
        "model_versions INSERT name=%s v=%s prod=%s shadow=%s",
        model_name, version, is_prod, is_shadow,
    )
    return row


def promote_version(model_name: str, version: int) -> Dict[str, Any]:
    """Mark ``version`` as prod, demoting any previous prod row.

    Order matters: we demote first (partial unique index on is_prod=true
    would otherwise collide on insert).
    """
    client = _client()
    # Step 1 — demote current prod rows for this model (if any).
    client.table(_TABLE).update({"is_prod": False, "is_shadow": False}).eq(
        "model_name", model_name
    ).eq("is_prod", True).execute()

    # Step 2 — promote the target.
    resp = (
        client.table(_TABLE)
        .update({"is_prod": True, "is_shadow": False, "is_retired": False})
        .eq("model_name", model_name)
        .eq("version", version)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        raise LookupError(f"No model_versions row for {model_name} v{version}")
    logger.info("model_versions PROMOTE %s → v%s", model_name, version)
    return rows[0]


def mark_shadow_version(model_name: str, version: int) -> Dict[str, Any]:
    """Flag a version as shadow (runs in parallel, does not ship)."""
    resp = (
        _client()
        .table(_TABLE)
        .update({"is_shadow": True, "is_prod": False})
        .eq("model_name", model_name)
        .eq("version", version)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        raise LookupError(f"No model_versions row for {model_name} v{version}")
    return rows[0]


def retire_version(model_name: str, version: int) -> Dict[str, Any]:
    resp = (
        _client()
        .table(_TABLE)
        .update({"is_retired": True, "is_prod": False, "is_shadow": False})
        .eq("model_name", model_name)
        .eq("version", version)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        raise LookupError(f"No model_versions row for {model_name} v{version}")
    logger.info("model_versions RETIRE %s v%s", model_name, version)
    return rows[0]
