"""Adapts the C++ hot path to the same match(order, reference_price) -> Fill
interface as PySimExchange, so trader.engine can swap backends freely.

Uses `hotpath.synthetic_fill_price`, a single allocation-free FFI call, for
the per-order pricing in the live trader's hot path -- see the comment above
it in bindings.cpp for why the naive two-call OrderBook-per-order version
was dropped (it benchmarked slower than pure Python). `hotpath.OrderBook`
itself is still a real price-time-priority matching engine, exercised and
benchmarked on its own in tests/test_orderbook.py.
"""
from __future__ import annotations

import time

from quant_pod.common.events import Fill, Order, Side
from quant_pod.trader.hotpath import synthetic_fill_price


class CppSimExchange:
    def __init__(self, slippage_bps: float = 1.0):
        self.slippage_bps = slippage_bps

    def match(self, order: Order, reference_price: float) -> Fill:
        fill_price = synthetic_fill_price(order.side == Side.BUY, reference_price, self.slippage_bps)
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            filled_ns=time.perf_counter_ns(),
        )
