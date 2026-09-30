"""Pure-Python price-time-priority order book, mirroring hotpath.OrderBook's
interface exactly. Exists only as a benchmarking baseline (see
scripts/benchmark_hotpath.py) to measure where the C++ matching engine's
advantage actually shows up: not on a single trivial price calc (FFI call
overhead loses there, see hotpath_exchange.py's docstring), but on matching
against a deep book with many resting orders.
"""
from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass


@dataclass
class PyFill:
    resting_order_id: int
    incoming_order_id: int
    price: float
    quantity: float


class PyOrderBook:
    def __init__(self):
        self._bid_prices: list[float] = []  # ascending; best bid is the last element
        self._ask_prices: list[float] = []  # ascending; best ask is the first element
        self._bids: dict[float, deque] = {}
        self._asks: dict[float, deque] = {}

    def add_limit_order(self, order_id: int, is_buy: bool, price: float, quantity: float) -> list[PyFill]:
        fills, quantity = self._match(order_id, is_buy, price, quantity, has_limit=True)
        if quantity > 0:
            book, prices = (self._bids, self._bid_prices) if is_buy else (self._asks, self._ask_prices)
            if price not in book:
                bisect.insort(prices, price)
                book[price] = deque()
            book[price].append([order_id, quantity])
        return fills

    def add_market_order(self, order_id: int, is_buy: bool, quantity: float) -> list[PyFill]:
        fills, _ = self._match(order_id, is_buy, 0.0, quantity, has_limit=False)
        return fills

    def best_bid(self) -> float:
        return self._bid_prices[-1] if self._bid_prices else 0.0

    def best_ask(self) -> float:
        return self._ask_prices[0] if self._ask_prices else 0.0

    def _match(self, incoming_id: int, is_buy: bool, limit_price: float, quantity: float, has_limit: bool):
        fills: list[PyFill] = []
        if is_buy:
            book, prices = self._asks, self._ask_prices
            better = lambda level: (not has_limit) or level <= limit_price
        else:
            book, prices = self._bids, self._bid_prices
            better = lambda level: (not has_limit) or level >= limit_price

        while quantity > 0 and prices:
            level = prices[0] if is_buy else prices[-1]
            if not better(level):
                break
            queue = book[level]
            while quantity > 0 and queue:
                resting = queue[0]
                matched = min(quantity, resting[1])
                fills.append(PyFill(resting[0], incoming_id, level, matched))
                quantity -= matched
                resting[1] -= matched
                if resting[1] <= 0:
                    queue.popleft()
            if not queue:
                del book[level]
                if is_buy:
                    prices.pop(0)
                else:
                    prices.pop()
            else:
                break

        return fills, quantity
