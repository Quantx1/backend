"""Data layer — market-data providers, broker connectivity, market calendar,
universe loaders, screener, tick collector.

Public sub-packages and modules::

    from backend.data.brokers    import BrokerFactory, BrokerTickerManager, ...
    from backend.data.providers  import KiteDataProvider, YFinanceProvider, ...
    from backend.data.market     import MarketDataProvider, get_market_data_provider
    from backend.data.market_calendar import is_market_open, next_trading_day
    from backend.data.market_overview import determine_market_condition, ...
    from backend.data.universe   import UniverseScreener
    from backend.data.screener   import LiveScreenerEngine
    from backend.data.tick_collector import ...
"""
