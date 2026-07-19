"""Canonical L2 market-depth model + broker parsers.

Honest-empty by construction: levels with no price/qty are dropped, so a
``MarketDepth`` with ``levels() == 0`` means "no depth" — callers must NOT
emit a depth update for it (never fabricate a synthetic ladder)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DepthLevel:
    price: float
    quantity: int
    orders: int = 0

    def to_dict(self) -> Dict:
        return {"price": self.price, "quantity": self.quantity, "orders": self.orders}


@dataclass
class MarketDepth:
    symbol: str
    bids: List[DepthLevel] = field(default_factory=list)  # index 0 = best bid
    asks: List[DepthLevel] = field(default_factory=list)
    source: str = "broker"

    def levels(self) -> int:
        return max(len(self.bids), len(self.asks))

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "levels": self.levels(),
            "source": self.source,
            "bids": [b.to_dict() for b in self.bids],
            "asks": [a.to_dict() for a in self.asks],
        }


def analyze_depth(d: "MarketDepth") -> Dict:
    """Deterministic L2 liquidity read (0 LLM tokens): total qty per side,
    order-book imbalance, best bid/ask + spread, and the single biggest
    'wall' level on each side. All pure arithmetic over the real ladder."""
    bid_qty = sum(b.quantity for b in d.bids)
    ask_qty = sum(a.quantity for a in d.asks)
    tot = bid_qty + ask_qty
    imbalance = round((bid_qty - ask_qty) / tot, 4) if tot else 0.0
    best_bid = d.bids[0].price if d.bids else None
    best_ask = d.asks[0].price if d.asks else None
    spread = round(best_ask - best_bid, 2) if (best_bid and best_ask) else None
    spread_pct = (
        round((best_ask - best_bid) / best_bid * 100, 3)
        if (best_bid and best_ask and best_bid > 0) else None
    )
    pressure = "buy_pressure" if imbalance >= 0.2 else "sell_pressure" if imbalance <= -0.2 else "balanced"
    return {
        "total_bid_qty": bid_qty,
        "total_ask_qty": ask_qty,
        "imbalance": imbalance,          # [-1, 1]; + = bids dominate
        "pressure": pressure,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "spread_pct": spread_pct,
        "bid_wall": max(d.bids, key=lambda x: x.quantity).to_dict() if d.bids else None,
        "ask_wall": max(d.asks, key=lambda x: x.quantity).to_dict() if d.asks else None,
        "levels": d.levels(),
    }


def _levels(rows: List[Dict]) -> List[DepthLevel]:
    out: List[DepthLevel] = []
    for r in rows or []:
        try:
            price = float(r.get("price") or 0)
            qty = int(r.get("quantity") or 0)
            orders = int(r.get("orders") or 0)
        except (TypeError, ValueError):
            continue  # malformed level — skip, never throw (keeps the rest of the tick alive)
        if price <= 0 or qty <= 0:
            continue  # honest-empty: drop padding/empty levels
        out.append(DepthLevel(price=price, quantity=qty, orders=orders))
    return out


def from_kite_depth(symbol: str, depth: Dict) -> MarketDepth:
    """Build MarketDepth from a KiteTicker MODE_FULL ``tick['depth']``
    ``{'buy': [{price, quantity, orders}, ...], 'sell': [...]}`` (5 levels each)."""
    depth = depth or {}
    return MarketDepth(symbol=symbol, bids=_levels(depth.get("buy")), asks=_levels(depth.get("sell")), source="broker")


def from_upstox_marketlevel(symbol: str, market_level) -> MarketDepth:
    """Build MarketDepth from an Upstox V3 ``marketFF.marketLevel`` list.

    Each entry carries one bid + one ask level:
    ``{bp, bq, bno, ap, aq, ano}`` (bid price/qty/orders, ask price/qty/orders).
    `full_d30` yields up to 30 entries. Honest-empty: levels with price<=0 or
    qty<=0 are dropped (same rule as ``_levels``)."""
    bids: List[DepthLevel] = []
    asks: List[DepthLevel] = []
    for lvl in market_level or []:
        try:
            bp, bq = float(lvl.get("bp") or 0), int(lvl.get("bq") or 0)
            ap, aq = float(lvl.get("ap") or 0), int(lvl.get("aq") or 0)
            bno, ano = int(lvl.get("bno") or 0), int(lvl.get("ano") or 0)
        except (TypeError, ValueError):
            continue
        if bp > 0 and bq > 0:
            bids.append(DepthLevel(price=bp, quantity=bq, orders=bno))
        if ap > 0 and aq > 0:
            asks.append(DepthLevel(price=ap, quantity=aq, orders=ano))
    return MarketDepth(symbol=symbol, bids=bids, asks=asks, source="broker")
