"""Shared event schemas used across the ingester, research, analyzer, and trader modules.

Keeping one canonical set of types here is what lets the same Strategy code run
unmodified in the vectorized backtester and in the live event-driven trader.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Bar(BaseModel):
    """A single OHLCV bar for one symbol. The unit the ingester stores and replays."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def mid(self) -> float:
        return (self.high + self.low) / 2


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class Order(BaseModel):
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    strategy_id: str = "default"
    order_id: str = Field(default_factory=lambda: _new_id("ord"))
    created_ns: int = Field(default_factory=lambda: _now_ns())


class Fill(BaseModel):
    order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    filled_ns: int = Field(default_factory=lambda: _now_ns())

    def tick_to_fill_ns(self, order_created_ns: int) -> int:
        return self.filled_ns - order_created_ns


class Signal(BaseModel):
    """Output of a Strategy: desired target position (in shares) for a symbol."""

    symbol: str
    target_position: float
    timestamp: datetime
    strategy_id: str = "default"


def _now_ns() -> int:
    import time

    return time.perf_counter_ns()


_id_counter = 0


def _new_id(prefix: str) -> str:
    global _id_counter
    _id_counter += 1
    return f"{prefix}-{_id_counter}"
