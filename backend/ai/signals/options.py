"""Options-chain signal generation + position monitoring.

Extracted from ``services/signal_generator.py`` in PR-A3.5. Public API::

    eng = OptionsSignalEngine(supabase, market_data_provider, fo_engine)
    await eng.generate_signals(save=True)
    await eng.monitor_positions()

``fo_engine`` may be ``None`` in environments where ``FOTradingEngine``
isn't importable; the engine falls back to chain-derived lot size.
"""
from __future__ import annotations

import logging
from datetime import date as date_cls, datetime
from typing import Dict, List, Optional

import pandas as pd

from .persistence import save_signals
from .types import GeneratedSignal

logger = logging.getLogger(__name__)


class OptionsSignalEngine:
    """Generates options-chain signals + monitors live positions.

    All dependencies are passed in at construction so the engine is
    testable in isolation and has no module-level singletons.
    """

    def __init__(
        self,
        supabase,
        market_data_provider,
        fo_engine,  # type: ignore[no-untyped-def]
        catalog_cache: Optional[Dict[str, str]] = None,
    ):
        self.supabase = supabase
        self.market_data_provider = market_data_provider
        self.fo_engine = fo_engine
        self._catalog_cache = catalog_cache if catalog_cache is not None else {}

    # ──────────────────────────────────────────────────────────────────
    # Public entry points
    # ──────────────────────────────────────────────────────────────────

    async def generate_signals(self, save: bool = True) -> List[GeneratedSignal]:
        """Generate options signals for users with active OPTIONS deployments.

        Called at 9:30 AM+ when live chain data is available.

        Flow:
        1. Get all active OPTIONS deployments (not paused)
        2. For each deployment, load the strategy class + params
        3. Build OptionsChainSnapshot from FO engine
        4. Run strategy.scan(chain, params)
        5. Convert OptionsTradeSignal → GeneratedSignal, save to DB
        """
        logger.info("Starting options signal generation...")
        all_signals: List[GeneratedSignal] = []

        try:
            deployments = self.supabase.table("user_strategy_deployments").select(
                "*, strategy_catalog(*)"
            ).eq("is_active", True).eq("is_paused", False).execute()

            if not deployments.data:
                logger.info("No active options deployments")
                return all_signals

            options_deployments = [
                d for d in deployments.data
                if d.get("strategy_catalog", {}).get("segment") == "OPTIONS"
            ]

            if not options_deployments:
                logger.info("No OPTIONS deployments found")
                return all_signals

            logger.info(f"Processing {len(options_deployments)} OPTIONS deployments")

            for deployment in options_deployments:
                try:
                    catalog = deployment.get("strategy_catalog", {})
                    strategy_class_path = catalog.get("strategy_class", "")
                    default_params = catalog.get("default_params", {})
                    custom_params = deployment.get("custom_params", {})
                    params = {**default_params, **custom_params}

                    strategy = self._load_strategy_class(strategy_class_path)
                    if strategy is None:
                        logger.warning(f"Could not load strategy: {strategy_class_path}")
                        continue

                    symbol = catalog.get("supported_symbols", ["NIFTY"])[0]
                    chain = await self._build_chain(symbol, params)
                    if chain is None:
                        continue

                    # Inject market context for strategies that need it.
                    if "_iv_20d_mean" not in params:
                        params["_iv_20d_mean"] = await self._get_iv_20d_mean(symbol, chain)
                    if "_iv_20d_std" not in params:
                        params["_iv_20d_std"] = await self._get_iv_20d_std(symbol, chain)
                    if "_prev_close" not in params:
                        params["_prev_close"] = chain.spot_price * 0.998
                    if "_adx" not in params:
                        params["_adx"] = await self._get_adx(symbol)

                    signal = strategy.scan(chain, params)
                    if signal is None:
                        continue

                    legs_desc = " + ".join(
                        f"{leg.direction} {leg.strike}{leg.option_type}"
                        for leg in signal.legs
                    )
                    gen_signal = GeneratedSignal(
                        symbol=signal.symbol,
                        exchange="NFO",
                        segment="OPTIONS",
                        direction="LONG" if signal.net_premium < 0 else "SHORT",
                        confidence=signal.confidence,
                        entry_price=abs(signal.net_premium),
                        stop_loss=signal.max_loss,
                        target_1=signal.max_profit if signal.max_profit != float('inf') else 0,
                        target_2=None,
                        target_3=None,
                        risk_reward=round(
                            signal.max_profit / max(signal.max_loss, 1), 2
                        ) if signal.max_loss > 0 and signal.max_profit != float('inf') else 0,
                        catboost_score=0,
                        tft_score=0,
                        stockformer_score=signal.confidence,
                        lgbm_score=0,
                        model_agreement=1,
                        reasons=signal.reasons,
                        is_premium=True,
                        strategy_name=catalog.get("name", ""),
                        strategy_catalog_id=catalog.get("id"),
                        lot_size=chain.lot_size,
                    )
                    all_signals.append(gen_signal)

                    self.supabase.table("user_strategy_deployments").update({
                        "last_signal_at": datetime.utcnow().isoformat(),
                    }).eq("id", deployment["id"]).execute()

                    logger.info(
                        f"Options signal: {signal.symbol} {legs_desc} "
                        f"conf={signal.confidence:.0f}"
                    )

                except Exception as e:
                    logger.error(f"Options deployment scan error: {e}")
                    continue

            if save and all_signals:
                await save_signals(
                    self.supabase, all_signals, catalog_cache=self._catalog_cache,
                )

            logger.info(
                f"Options signal generation complete: {len(all_signals)} signals"
            )

        except Exception as e:
            logger.error(f"Options signal generation failed: {e}")

        return all_signals

    async def monitor_positions(self) -> None:
        """Monitor active options positions and check exit conditions.

        Called every 15 minutes during market hours.
        """
        logger.info("Monitoring options positions...")

        try:
            positions = self.supabase.table("positions").select(
                "*, trades(*)"
            ).eq("is_active", True).eq("segment", "OPTIONS").execute()

            if not positions.data:
                return

            for pos in positions.data:
                try:
                    trade = pos.get("trades", {})
                    strategy_catalog_id = trade.get("strategy_catalog_id")

                    if not strategy_catalog_id:
                        continue

                    catalog_result = self.supabase.table("strategy_catalog").select("*").eq(
                        "id", strategy_catalog_id
                    ).single().execute()

                    if not catalog_result.data:
                        continue

                    catalog = catalog_result.data
                    strategy = self._load_strategy_class(catalog.get("strategy_class", ""))
                    if strategy is None:
                        continue

                    params = catalog.get("default_params", {})

                    chain = await self._build_chain(pos["symbol"], params)
                    if chain is None:
                        continue

                    position_dict = {
                        "legs": [
                            {
                                "strike": trade.get("strike_price", 0),
                                "option_type": trade.get("option_type", "CE"),
                                "entry_price": trade.get("entry_price", 0),
                            }
                        ],
                        "entry_price": trade.get("entry_price", 0),
                        "entry_time": trade.get("created_at", ""),
                        "entry_date": trade.get("created_at", "")[:10]
                        if trade.get("created_at") else "",
                        "highest_since_entry": pos.get(
                            "highest_since_entry", trade.get("entry_price", 0)
                        ),
                    }

                    exit_signal = strategy.should_exit(chain, position_dict, params)
                    if exit_signal:
                        logger.info(
                            f"Options exit signal: {pos['symbol']} "
                            f"reason={exit_signal.reason}"
                        )
                        current_price = exit_signal.exit_price or pos.get("current_price", 0)
                        await self._close_position(pos, current_price, exit_signal.reason)

                except Exception as e:
                    logger.error(
                        f"Options position monitor error for {pos.get('symbol')}: {e}"
                    )

        except Exception as e:
            logger.error(f"Options position monitoring failed: {e}")

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    async def _close_position(self, position: Dict, exit_price: float, reason: str) -> None:
        """Close an options position and update P&L."""
        try:
            entry_price = position.get("average_price") or position.get("entry_price", 0)
            quantity = position.get("quantity", 1)
            direction = position.get("direction", "LONG")

            if direction == "LONG":
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity

            self.supabase.table("positions").update({
                "is_active": False,
                "current_price": exit_price,
                "status": "closed",
                "closed_at": datetime.utcnow().isoformat(),
            }).eq("id", position["id"]).execute()

            trade_id = position.get("trade_id")
            if trade_id:
                self.supabase.table("trades").update({
                    "exit_price": exit_price,
                    "exit_reason": reason,
                    "status": "closed",
                    "realized_pnl": pnl,
                    "closed_at": datetime.utcnow().isoformat(),
                }).eq("id", trade_id).execute()

            signal_id = position.get("signal_id")
            if signal_id:
                signal = self.supabase.table("signals").select(
                    "strategy_catalog_id"
                ).eq("id", signal_id).single().execute()
                if signal.data and signal.data.get("strategy_catalog_id"):
                    user_id = position.get("user_id")
                    if user_id:
                        self.supabase.rpc("increment_deployment_stats", {
                            "p_user_id": user_id,
                            "p_strategy_id": signal.data["strategy_catalog_id"],
                            "p_pnl": pnl,
                            "p_is_win": pnl > 0,
                        }).execute()

            logger.info(
                f"Options position closed: {position.get('symbol')} "
                f"reason={reason} pnl={pnl:.0f}"
            )

        except Exception as e:
            logger.error(f"Close options position error: {e}")

    def _load_strategy_class(self, class_path: str):
        """Dynamically load a strategy class from its dotted path."""
        try:
            parts = class_path.rsplit(".", 1)
            if len(parts) != 2:
                return None
            module_path, class_name = parts
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls()
        except Exception as e:
            logger.warning(f"Failed to load strategy class {class_path}: {e}")
            return None

    async def _build_chain(self, symbol: str, params: Dict):
        """Build OptionsChainSnapshot from live broker/market data.

        Returns None when no live chain is available — no synthetic fallback
        (no-fallbacks lock). The F&O signal is simply skipped for that symbol.
        """
        try:
            from ml.strategies.options_base import OptionsChainSnapshot, OptionSnapshot

            raw_chain = await self.market_data_provider.get_option_chain_async(symbol)

            if not raw_chain:
                logger.warning(f"No options chain data for {symbol}")
                return None

            spot_quote = await self.market_data_provider.get_quote_async(symbol)
            spot_price = spot_quote.ltp if spot_quote else 0
            if spot_price <= 0:
                defaults = {"NIFTY": 24000, "BANKNIFTY": 51000, "FINNIFTY": 23000}
                spot_price = defaults.get(symbol, 0)
            if spot_price <= 0:
                return None

            strikes = sorted(set(c['strike'] for c in raw_chain))
            strike_gap = min(
                (strikes[i + 1] - strikes[i]) for i in range(len(strikes) - 1)
            ) if len(strikes) > 1 else 50

            lot_size = raw_chain[0].get('lot_size', 1) if raw_chain else 1
            if self.fo_engine is not None:
                lot_size = self.fo_engine().get_lot_size(symbol) or lot_size

            atm_strike = round(spot_price / strike_gap) * strike_gap
            expiry_str = raw_chain[0].get('expiry', '')

            expiry_date = date_cls.today()
            if expiry_str:
                try:
                    expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                except ValueError:
                    pass

            snapshots = []
            total_call_oi = 0
            total_put_oi = 0
            for c in raw_chain:
                snap = OptionSnapshot(
                    strike=c['strike'],
                    option_type=c['option_type'],
                    expiry=expiry_date,
                    ltp=c.get('ltp', 0),
                    bid=c.get('bid', 0),
                    ask=c.get('ask', 0),
                    iv=c.get('iv', 0),
                    oi=c.get('oi', 0),
                    oi_change=c.get('oi_change', 0),
                    volume=c.get('volume', 0),
                    delta=c.get('delta', 0),
                    gamma=c.get('gamma', 0),
                    theta=c.get('theta', 0),
                    vega=c.get('vega', 0),
                )
                snapshots.append(snap)
                if c['option_type'] == 'CE':
                    total_call_oi += c.get('oi', 0)
                else:
                    total_put_oi += c.get('oi', 0)

            pcr = total_put_oi / max(total_call_oi, 1)
            atm_ivs = [s.iv for s in snapshots if s.strike == atm_strike and s.iv > 0]
            iv_index = sum(atm_ivs) / len(atm_ivs) if atm_ivs else 15.0

            chain_snapshot = OptionsChainSnapshot(
                symbol=symbol,
                spot_price=spot_price,
                atm_strike=atm_strike,
                strike_gap=strike_gap,
                lot_size=lot_size,
                expiry=expiry_date,
                chain=snapshots,
                iv_index=iv_index,
                pcr=pcr,
                timestamp=datetime.now(),
            )
            logger.info(
                f"Built options chain for {symbol}: {len(snapshots)} contracts, "
                f"spot={spot_price}, ATM={atm_strike}, PCR={pcr:.2f}, IV={iv_index:.1f}"
            )
            return chain_snapshot

        except Exception as e:
            logger.warning(f"Build options chain error for {symbol}: {e}")
            return None

    async def _get_iv_20d_mean(self, symbol: str, chain) -> float:
        """Get 20-day mean IV from INDIA VIX or estimate from chain."""
        try:
            vix_quote = self.market_data_provider.get_quote("VIX")
            if vix_quote and vix_quote.ltp > 0:
                return vix_quote.ltp * 0.85
        except Exception:
            pass
        return chain.iv_index * 0.80

    async def _get_iv_20d_std(self, symbol: str, chain) -> float:
        """Get 20-day IV standard deviation (estimate)."""
        try:
            vix_quote = self.market_data_provider.get_quote("VIX")
            if vix_quote and vix_quote.ltp > 0:
                return vix_quote.ltp * 0.18
        except Exception:
            pass
        return chain.iv_index * 0.15

    async def _get_adx(self, symbol: str) -> float:
        """Get ADX from recent historical data."""
        try:
            hist = self.market_data_provider.get_historical(
                symbol, period="3mo", interval="1d",
            )
            if hist is not None and len(hist) >= 20:
                import ta
                adx = ta.trend.ADXIndicator(
                    hist['high'], hist['low'], hist['close'], window=14,
                )
                val = adx.adx().iloc[-1]
                if not pd.isna(val):
                    return float(val)
        except Exception:
            pass
        return 25.0  # neutral default
