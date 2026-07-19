"""Unit tests for the manual ad-hoc order param validation.

These exercise ``_validate_adhoc_order`` directly — the pure, DB/broker-free
slice of ``POST /api/broker/order``. The other gates (global halt, per-user
kill-switch, tier, connected+fresh broker) are reused from code that is
tested elsewhere, so we don't re-test them here.
"""

import pytest
from fastapi import HTTPException

from backend.api.broker_routes import AdHocOrderRequest, _validate_adhoc_order
from backend.data.brokers.integration import OrderType, ProductType, TransactionType


def _req(**kw):
    base = dict(symbol="RELIANCE", transaction_type="BUY", quantity=1)
    base.update(kw)
    return AdHocOrderRequest(**base)


def test_valid_market_order_parses_enums():
    txn, otype, product = _validate_adhoc_order(_req())
    assert txn is TransactionType.BUY
    assert otype is OrderType.MARKET
    assert product is ProductType.CNC


def test_valid_limit_order_with_price():
    txn, otype, product = _validate_adhoc_order(
        _req(transaction_type="sell", order_type="LIMIT", price=2500.5, product="MIS")
    )
    assert txn is TransactionType.SELL
    assert otype is OrderType.LIMIT
    assert product is ProductType.MIS


def test_bad_transaction_type_422():
    with pytest.raises(HTTPException) as exc:
        _validate_adhoc_order(_req(transaction_type="HODL"))
    assert exc.value.status_code == 422


def test_sl_order_type_rejected_422():
    with pytest.raises(HTTPException) as exc:
        _validate_adhoc_order(_req(order_type="SL"))
    assert exc.value.status_code == 422
    assert "MARKET and LIMIT" in str(exc.value.detail)


def test_quantity_below_one_422():
    with pytest.raises(HTTPException) as exc:
        _validate_adhoc_order(_req(quantity=0))
    assert exc.value.status_code == 422


def test_limit_without_price_422():
    with pytest.raises(HTTPException) as exc:
        _validate_adhoc_order(_req(order_type="LIMIT", price=0))
    assert exc.value.status_code == 422
    assert "price" in str(exc.value.detail).lower()
