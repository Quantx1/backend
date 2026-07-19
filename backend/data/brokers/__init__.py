"""Broker connectivity — credentials, integration adapters, ticker mapping.

Public API::

    from backend.data.brokers import (
        # credentials
        encrypt_credentials, decrypt_credentials, mask_credentials,
        # integration
        BaseBroker, BrokerFactory, TradeExecutor,
        ZerodhaBroker, AngelOneBroker, UpstoxBroker,
        # ticker mapping
        BrokerTickerManager,
    )
"""
from .credentials import (
    decrypt_credentials,
    encrypt_credentials,
    generate_encryption_key,
    mask_credentials,
)
from .integration import (
    AngelOneBroker,
    BaseBroker,
    BrokerFactory,
    GTTOrder,
    Order,
    OrderStatus,
    OrderType,
    Position,
    ProductType,
    TradeExecutor,
    TransactionType,
    UpstoxBroker,
    ZerodhaBroker,
)
from .ticker_mapping import (
    AngelOneTickerAdapter,
    BrokerTickerManager,
    BrokerTickerSource,
    UpstoxTickerAdapter,
    ZerodhaTickerAdapter,
)

__all__ = [
    # credentials
    "decrypt_credentials", "encrypt_credentials",
    "generate_encryption_key", "mask_credentials",
    # integration
    "AngelOneBroker", "BaseBroker", "BrokerFactory", "GTTOrder",
    "Order", "OrderStatus", "OrderType", "Position", "ProductType",
    "TradeExecutor", "TransactionType", "UpstoxBroker", "ZerodhaBroker",
    # ticker mapping
    "AngelOneTickerAdapter", "BrokerTickerManager", "BrokerTickerSource",
    "UpstoxTickerAdapter", "ZerodhaTickerAdapter",
]
