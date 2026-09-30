from datetime import datetime

from quant_pod.common.events import Bar, Fill, Order, OrderType, Side


def test_bar_mid():
    bar = Bar(symbol="AAPL", timestamp=datetime(2024, 1, 1), open=10, high=12, low=8, close=11)
    assert bar.mid() == 10.0


def test_order_defaults_unique_ids():
    o1 = Order(symbol="AAPL", side=Side.BUY, quantity=10, order_type=OrderType.MARKET)
    o2 = Order(symbol="AAPL", side=Side.BUY, quantity=10, order_type=OrderType.MARKET)
    assert o1.order_id != o2.order_id
    assert o1.created_ns > 0


def test_fill_tick_to_fill_ns():
    order = Order(symbol="AAPL", side=Side.BUY, quantity=10)
    fill = Fill(order_id=order.order_id, symbol="AAPL", side=Side.BUY, quantity=10, price=100.0)
    assert fill.tick_to_fill_ns(order.created_ns) >= 0
