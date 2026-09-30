"""Low-latency(-simulated) trading engine.

Subscribes to the ingester's replay feed over ZeroMQ, runs a Strategy live
(bar by bar, exactly as the backtester would), sends orders through a
pluggable matching engine (Python reference or C++ hot path), and records
tick-to-fill latency so the two backends can be benchmarked against each
other.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import zmq
import zmq.asyncio

from quant_pod.common.config import REPLAY_ZMQ_ADDR
from quant_pod.common.events import Bar, Order, OrderType, Side
from quant_pod.common.logging import get_logger
from quant_pod.ingester.replay import END_OF_STREAM
from quant_pod.research.strategy import Strategy
from quant_pod.trader.bar_buffer import BarBuffer
from quant_pod.trader.latency import LatencyTracker
from quant_pod.trader.risk_limits import RiskLimits
from quant_pod.trader.sim_exchange import PySimExchange

log = get_logger("trader")


class Exchange:
    """Protocol every matching-engine backend implements: match(order, reference_price) -> Fill."""


def _make_exchange(engine: str, slippage_bps: float = 1.0):
    if engine == "python":
        return PySimExchange(slippage_bps=slippage_bps)
    if engine == "cpp":
        from quant_pod.trader.hotpath_exchange import CppSimExchange

        return CppSimExchange(slippage_bps=slippage_bps)
    raise ValueError(f"unknown engine {engine!r}, expected 'python' or 'cpp'")


@dataclass
class TraderResult:
    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    tick_latency: LatencyTracker
    matching_latency: LatencyTracker


async def run_trader(
    symbol: str,
    strategy: Strategy,
    engine: str = "python",
    initial_cash: float = 100_000.0,
    risk_limits: RiskLimits | None = None,
    zmq_addr: str = REPLAY_ZMQ_ADDR,
) -> TraderResult:
    risk_limits = risk_limits or RiskLimits()
    exchange = _make_exchange(engine)
    # Every tick pays for a bar-buffer append + strategy evaluation, whether
    # or not it produces a trade -- that's the real steady-state hot-path
    # cost, so it's timed on every tick, not just the ones that trade.
    tick_latency = LatencyTracker(label=f"{strategy.strategy_id}/tick")
    matching_latency = LatencyTracker(label=f"{strategy.strategy_id}/matching[{engine}]")

    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(zmq_addr)
    sock.setsockopt(zmq.SUBSCRIBE, symbol.encode())

    log.info("Trader started: symbol=%s engine=%s strategy=%s", symbol, engine, strategy.strategy_id)

    bar_buffer = BarBuffer()
    position = 0.0
    cash = initial_cash
    equity_points: list[tuple[pd.Timestamp, float]] = []
    trade_records: list[dict] = []

    try:
        while True:
            _, payload = await sock.recv_multipart()
            tick_received_ns = time.perf_counter_ns()

            if payload == END_OF_STREAM:
                break

            bar = Bar.model_validate_json(payload)
            bar_buffer.append(bar)
            bars_df = bar_buffer.as_dataframe()

            target_position = strategy.target_position_for(bars_df)
            tick_latency.record(time.perf_counter_ns() - tick_received_ns)
            order_qty = target_position - position

            if order_qty != 0:
                order = Order(
                    symbol=symbol,
                    side=Side.BUY if order_qty > 0 else Side.SELL,
                    quantity=abs(order_qty),
                    order_type=OrderType.MARKET,
                    strategy_id=strategy.strategy_id,
                )

                ok, reason = risk_limits.check(order, position, bar.close)
                if not ok:
                    log.warning("Order rejected by risk limits: %s", reason)
                else:
                    match_start_ns = time.perf_counter_ns()
                    fill = exchange.match(order, bar.close)
                    matching_latency.record(fill.filled_ns - match_start_ns)

                    direction = 1 if fill.side == Side.BUY else -1
                    cash -= direction * fill.quantity * fill.price
                    position += direction * fill.quantity
                    trade_records.append(
                        {
                            "timestamp": bar.timestamp,
                            "quantity": direction * fill.quantity,
                            "price": fill.price,
                            "commission": 0.0,
                        }
                    )

            equity_points.append((bar.timestamp, cash + position * bar.close))

    finally:
        sock.close()
        ctx.term()

    equity_curve = pd.Series(dict(equity_points), name="equity")
    equity_curve.index.name = "timestamp"
    returns = equity_curve.pct_change().fillna(0.0).rename("returns")
    trades = pd.DataFrame(trade_records)

    log.info("Trader finished: %d bars processed, %d trades", len(bar_buffer), len(trades))
    tick_latency.print_summary()
    matching_latency.print_summary()

    return TraderResult(
        equity_curve=equity_curve,
        returns=returns,
        trades=trades,
        tick_latency=tick_latency,
        matching_latency=matching_latency,
    )
