"""Python reference matching engine.

Not a real limit order book -- it fills market orders immediately against the
current bar's price (with configurable slippage), which is enough to drive
the trader's fill/latency accounting. Exists mainly as a correctness baseline
and pure-Python latency comparison point against the C++ `hotpath.OrderBook`.
"""
from __future__ import annotations

import time

from quant_pod.common.events import Fill, Order, Side


class PySimExchange:
    def __init__(self, slippage_bps: float = 1.0):
        self.slippage_bps = slippage_bps

    def match(self, order: Order, reference_price: float) -> Fill:
        direction = 1 if order.side == Side.BUY else -1
        fill_price = reference_price * (1 + direction * self.slippage_bps / 10_000)
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            filled_ns=time.perf_counter_ns(),
        )
