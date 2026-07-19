"""
Admin EOD scanner monitoring endpoints.

  GET /admin/eod/runs       recent EOD scan runs (started_at, status, counts)
  GET /admin/eod/universe   candidate universe for a given trade_date

Read-only operational visibility into the nightly EOD scan job. Powers
the admin command-center "EOD scanner" tile.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from ._deps import AdminUser, get_admin_user, get_supabase_admin

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# SCHEMAS
# ============================================================================


class EODScanRunItem(BaseModel):
    id: str
    trade_date: str
    status: str
    source: Optional[str]
    scan_type: Optional[str]
    candidate_count: int = 0
    signal_count: int = 0
    started_at: Optional[str]
    finished_at: Optional[str]
    error: Optional[str]


class EODScanRunsResponse(BaseModel):
    runs: List[EODScanRunItem]


class DailyUniverseItem(BaseModel):
    trade_date: str
    symbol: str
    source: Optional[str]
    scan_type: Optional[str]


class DailyUniverseResponse(BaseModel):
    trade_date: str
    total: int
    candidates: List[DailyUniverseItem]


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get("/eod/runs", response_model=EODScanRunsResponse)
async def get_eod_runs(
    limit: int = Query(10, ge=1, le=50),
    admin: AdminUser = Depends(get_admin_user),
):
    """Get recent EOD scan runs for monitoring."""
    supabase = get_supabase_admin()
    result = (
        supabase.table("eod_scan_runs")
        .select("*")
        .order("started_at", desc=True)
        .limit(limit)
        .execute()
    )
    runs = []
    for row in result.data or []:
        runs.append(EODScanRunItem(
            id=row.get("id"),
            trade_date=row.get("trade_date"),
            status=row.get("status"),
            source=row.get("source"),
            scan_type=row.get("scan_type"),
            candidate_count=row.get("candidate_count", 0),
            signal_count=row.get("signal_count", 0),
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
            error=row.get("error"),
        ))
    return EODScanRunsResponse(runs=runs)


@router.get("/eod/universe", response_model=DailyUniverseResponse)
async def get_eod_universe(
    trade_date: Optional[str] = None,
    limit: int = Query(200, ge=1, le=1000),
    admin: AdminUser = Depends(get_admin_user),
):
    """Get EOD candidate universe for a given trade date.

    If no ``trade_date`` is provided, returns the latest available date.
    """
    supabase = get_supabase_admin()

    if not trade_date:
        latest = (
            supabase.table("daily_universe")
            .select("trade_date")
            .order("trade_date", desc=True)
            .limit(1)
            .execute()
        )
        if latest.data:
            trade_date = latest.data[0].get("trade_date")
        else:
            return DailyUniverseResponse(trade_date="", total=0, candidates=[])

    result = (
        supabase.table("daily_universe")
        .select("trade_date, symbol, source, scan_type")
        .eq("trade_date", trade_date)
        .order("symbol", desc=False)
        .limit(limit)
        .execute()
    )

    candidates = [
        DailyUniverseItem(
            trade_date=row.get("trade_date"),
            symbol=row.get("symbol"),
            source=row.get("source"),
            scan_type=row.get("scan_type"),
        )
        for row in (result.data or [])
    ]

    return DailyUniverseResponse(
        trade_date=trade_date or "",
        total=len(result.data or []),
        candidates=candidates,
    )
