"""NSE F&O lot sizes — Jan 2026 revision.

Source (verified 2026-05-31 across 5 vendors):
  - https://www.sahi.com/blogs/nifty-lot-size-2026-bank-nifty-sensex
  - https://hdfcsky.com/news/nse-revises-market-lot-sizes-for-major-index-derivatives-effective-january-2026
  - https://www.venturasecurities.com/blog/fo-lot-size-changes-in-india-what-traders-need-to-know-effective-jan-2026/
  - https://www.kotakneo.com/investing-guide/articles/bank-nifty-lot-size/
  - https://www.icicidirect.com/faqs/fno/what-are-the-new-lot-sizes-for-index-derivatives

Prior to Jan 2026: Nifty=75, BankNifty=35. Hard-code these post-revision
values; do NOT pull from training data because that knowledge is stale.

NSE reviews and publishes a notice each cycle; refresh when NSE issues
the next revision notification.
"""

LOT_SIZES = {
    # Index derivatives (Jan 2026 revision)
    "NIFTY": 65,
    "BANKNIFTY": 30,
    "FINNIFTY": 60,
    "MIDCPNIFTY": 120,    # Nifty Midcap Select
    "BANKEX": 30,
    "SENSEX": 20,
    # Stock futures keep stock-specific lot sizes managed by NSE — we don't
    # hard-code them here; broker quote endpoint carries the per-stock lot.
}


# Tick size (smallest price increment) for index futures
FUTURE_TICK_SIZES = {
    "NIFTY": 0.05,
    "BANKNIFTY": 0.05,
    "FINNIFTY": 0.05,
    "MIDCPNIFTY": 0.05,
    "SENSEX": 0.05,
}


def lot_value(symbol: str, spot_price: float) -> float:
    """Notional rupee value of one lot of an index future.

    >>> lot_value("NIFTY", 24000)      # 24000 × 65 = 15.6 lakh
    1560000.0
    >>> lot_value("BANKNIFTY", 51000)  # 51000 × 30 = 15.3 lakh
    1530000.0
    """
    lot = LOT_SIZES.get(symbol.upper())
    if not lot or spot_price <= 0:
        return 0.0
    return float(lot * spot_price)
