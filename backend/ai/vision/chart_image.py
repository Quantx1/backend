"""
Server-side chart image renderer for B2 vision analysis.

Renders a 120-bar OHLC candlestick chart (last ~6 months of daily data)
with volume sub-plot and 20/50 EMA overlays. Output is a PNG byte
string that the vision-capable open model (served via the OpenRouter
gateway) can ingest directly.

Design:
    - Pure matplotlib (no mplfinance hard dep) so we don't add weight.
    - Dark theme matches the platform palette.
    - 800×500 px — large enough for the model to resolve candle wicks,
      small enough to stay under the 20 MB/inline-image cap by a wide margin.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


CHART_WIDTH_PX = 800
CHART_HEIGHT_PX = 500
DPI = 100
BARS = 120


def _fetch_ohlcv(symbol: str, bars: int = BARS):
    """Best-effort OHLCV fetch. Yfinance fallback keeps the module
    testable locally without the full data layer."""
    try:
        import yfinance as yf
        ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        df = yf.Ticker(ticker).history(period="1y", interval="1d")
        if df is None or df.empty:
            return None
        df = df.tail(bars).reset_index()
        return df
    except Exception as exc:
        logger.warning("chart_image ohlcv fetch failed %s: %s", symbol, exc)
        return None


def render_chart_png(symbol: str, bars: int = BARS) -> Optional[bytes]:
    """Return PNG bytes of a 120-bar candlestick chart for ``symbol``.
    Returns None on any failure (e.g. no OHLCV, matplotlib missing)."""
    df = _fetch_ohlcv(symbol, bars=bars)
    if df is None or len(df) < 20:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        logger.warning("matplotlib missing: %s", exc)
        return None

    close = df["Close"].astype(float).values
    open_ = df["Open"].astype(float).values
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    vol = df["Volume"].astype(float).values if "Volume" in df.columns else np.zeros(len(close))

    # Simple 20 / 50 EMAs.
    def ema(series, span):
        alpha = 2.0 / (span + 1.0)
        out = np.zeros_like(series)
        out[0] = series[0]
        for i in range(1, len(series)):
            out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
        return out

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, sharex=True,
        figsize=(CHART_WIDTH_PX / DPI, CHART_HEIGHT_PX / DPI),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="#0A0D14",
    )
    for ax in (ax_price, ax_vol):
        ax.set_facecolor("#0A0D14")
        ax.tick_params(colors="#6b7280", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#242838")
        ax.grid(True, color="#1a1f2e", linewidth=0.5, alpha=0.7)

    x = np.arange(len(close))
    # Candles.
    for i in range(len(close)):
        up = close[i] >= open_[i]
        body_color = "#05B878" if up else "#FF5947"
        # High-low wick.
        ax_price.vlines(x[i], low[i], high[i], color=body_color, linewidth=0.7, alpha=0.8)
        # Body.
        body_low = min(open_[i], close[i])
        body_high = max(open_[i], close[i])
        ax_price.add_patch(plt.Rectangle(
            (x[i] - 0.3, body_low), 0.6, max(body_high - body_low, 0.01),
            color=body_color, alpha=0.95,
        ))

    # EMAs.
    ax_price.plot(x, ema20, color="#4FECCD", linewidth=1.2, alpha=0.9, label="EMA 20")
    ax_price.plot(x, ema50, color="#FFD166", linewidth=1.2, alpha=0.9, label="EMA 50")
    ax_price.legend(loc="upper left", facecolor="#111520", edgecolor="#242838", labelcolor="#DADADA", fontsize=7)

    ax_price.set_title(
        f"{symbol.upper().replace('.NS', '')} — last {len(close)} daily candles",
        color="#DADADA", fontsize=9, pad=8,
    )
    ax_price.set_ylabel("Price (₹)", color="#6b7280", fontsize=7)

    # Volume bars.
    vol_colors = ["#05B878" if close[i] >= open_[i] else "#FF5947" for i in range(len(close))]
    ax_vol.bar(x, vol, color=vol_colors, alpha=0.6, width=0.6)
    ax_vol.set_ylabel("Volume", color="#6b7280", fontsize=7)
    ax_vol.set_yticks([])

    # Tight layout.
    plt.tight_layout(pad=1.0)

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=DPI, facecolor="#0A0D14", edgecolor="none")
    except Exception as exc:
        logger.warning("chart render savefig failed: %s", exc)
        plt.close(fig)
        return None
    plt.close(fig)
    return buf.getvalue()


__all__ = ["render_chart_png", "BARS", "CHART_WIDTH_PX", "CHART_HEIGHT_PX"]
