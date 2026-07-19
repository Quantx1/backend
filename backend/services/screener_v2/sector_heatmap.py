"""Sector heatmap aggregator (PR-S13).

Builds a sector × metric grid the frontend renders as a colored heatmap:

  Sector          |  Avg Change% | Median Change% | Breadth (% up) | Top Hit Count | Volume Surge %
  Banking         |       +1.34  |        +0.92    |      67%       |       8       |     22%
  IT              |       -0.42  |        -0.18    |      35%       |       4       |     12%
  ...

`Breadth` = percent of peers with change% > 0
`Top Hit Count` = how many sector members showed up in today's Power
  Screener confluence run (joins scanner_outcomes today rows by sector)
`Volume Surge %` = percent of peers with vol_ratio > 1.5

One-shot endpoint returning the full grid sorted by best-to-worst.
Cheap: reads the already-computed summary_df, no extra fetches.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SectorRow:
    """One sector's heatmap row."""
    sector: str
    peer_count: int
    avg_change_pct: float
    median_change_pct: float
    breadth_pct: int                    # % of peers up
    volume_surge_pct: int               # % of peers with vol_ratio > 1.5
    rsi_oversold_count: int             # # peers with RSI < 30
    rsi_overbought_count: int           # # peers with RSI > 70
    top_movers: List[Dict[str, Any]]    # top 3 by |change_pct|

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_sector_heatmap(
    summary_df: pd.DataFrame,
    stock_info: Dict[str, Dict[str, str]],
) -> List[SectorRow]:
    """Aggregate summary_df rows by sector."""
    if summary_df is None or summary_df.empty:
        return []
    if "symbol" not in summary_df.columns:
        return []

    # Attach sector to each row
    def _sec(sym: str) -> str:
        info = stock_info.get(sym, {})
        return info.get("sector") or "Other"

    df = summary_df.copy()
    df["sector"] = df["symbol"].apply(_sec)
    # Drop symbols with no sector mapping
    df = df[df["sector"] != "Other"]
    if df.empty:
        return []

    rows: List[SectorRow] = []
    for sector, grp in df.groupby("sector"):
        if grp.empty:
            continue
        changes = grp["change_pct"].astype(float)
        vol_ratios = grp.get("volume_ratio", pd.Series([1.0] * len(grp))).astype(float)
        rsi = grp.get("rsi_14", pd.Series([50.0] * len(grp))).astype(float)

        # Top 3 movers by absolute change
        top = grp.assign(abs_chg=changes.abs()).sort_values("abs_chg", ascending=False).head(3)
        top_movers = [
            {
                "symbol": r["symbol"],
                "name": stock_info.get(r["symbol"], {}).get("name", r["symbol"]),
                "change_pct": round(float(r["change_pct"]), 2),
                "close": round(float(r.get("close", 0)), 2),
            }
            for _, r in top.iterrows()
        ]

        rows.append(SectorRow(
            sector=sector,
            peer_count=len(grp),
            avg_change_pct=round(float(changes.mean()), 2),
            median_change_pct=round(float(changes.median()), 2),
            breadth_pct=int(round((changes > 0).mean() * 100)),
            volume_surge_pct=int(round((vol_ratios > 1.5).mean() * 100)),
            rsi_oversold_count=int((rsi < 30).sum()),
            rsi_overbought_count=int((rsi > 70).sum()),
            top_movers=top_movers,
        ))

    # Sort by avg_change_pct descending (best sector at top)
    rows.sort(key=lambda r: r.avg_change_pct, reverse=True)
    return rows
