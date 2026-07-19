"""AutoPilot live-path hardening (2026-06-22).

Covers the safety fixes:
  - Zerodha enctoken places a real SL-M broker stop instead of skipping GTT.
"""
from backend.data.brokers.integration import (
    GTTOrder, Order, OrderStatus, OrderType, ProductType, TransactionType, ZerodhaBroker,
)


def _enctoken_zerodha():
    # Bypass __init__ (it would try to authenticate over the network); we only
    # need the enctoken-mode flag + a stubbed place_order.
    b = ZerodhaBroker.__new__(ZerodhaBroker)
    b._enctoken = "tok"
    b.kite = None
    b.is_authenticated = True
    return b


def test_enctoken_places_slm_stop_not_skip():
    b = _enctoken_zerodha()
    captured = {}

    def fake_place_order(order: Order) -> Order:
        captured["order"] = order
        order.order_id = "SLM-123"
        order.status = OrderStatus.PENDING
        return order

    b.place_order = fake_place_order

    gtt = GTTOrder(
        symbol="RELIANCE", exchange="NSE", trigger_type="single",
        trigger_values=[2400.0],
        orders=[{"transaction_type": "SELL", "quantity": 10, "price": 2400.0}],
    )
    out = b.place_gtt_order(gtt)

    # Previously returned status="skipped" (no broker stop). Now a real SL-M.
    assert out.status == "sl_placed"
    assert out.gtt_id == "SLM-123"
    o = captured["order"]
    assert o.order_type == OrderType.SL_M
    assert o.transaction_type == TransactionType.SELL
    assert o.quantity == 10
    assert o.trigger_price == 2400.0
    assert o.product == ProductType.CNC


def test_enctoken_slm_rejection_marks_sl_failed():
    b = _enctoken_zerodha()

    def reject(order: Order) -> Order:
        order.status = OrderStatus.REJECTED
        order.message = "insufficient funds"
        return order

    b.place_order = reject
    gtt = GTTOrder(
        symbol="TCS", exchange="NSE", trigger_type="single",
        trigger_values=[3500.0],
        orders=[{"transaction_type": "SELL", "quantity": 5, "price": 3500.0}],
    )
    out = b.place_gtt_order(gtt)
    # Orchestrator maps sl_failed -> failed -> unprotected alert (no false "placed").
    assert out.status == "sl_failed"
