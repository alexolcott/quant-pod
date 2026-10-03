"""Live event loop for the Avellaneda-Stoikov market maker.

Subscribes to the same ZeroMQ bar-replay feed `trader.engine.run_trader`
uses, but doesn't route a single target-position order through
`Exchange.match` the way that loop does: each tick it quotes *both* sides of
the book, reusing `RiskLimits` as a pre-trade check on each side's
hypothetical fill and `LatencyTracker` to time the per-tick quoting hot path
-- the same low-latency-measurement story `trader.engine` tells, applied to a
two-sided quoter instead of a single order.

Fills are still drawn from `marketmaking.simulator.fill_probability` rather
than matched through an `Exchange` -- see that module's docstring for why
(no cancellation in either matching-engine backend, and this strategy
replaces both quotes every tick).

Returns a `quant_pod.marketmaking.simulator.MarketMakerResult` -- the exact
type the research-side simulator produces -- so every analysis tool built for
it (`summarize`, `markout.compute_markouts`, ...) runs unmodified on a live
run too: the same "same code, live or simulated" story `research.strategy.
Strategy` tells for the bar-based strategies, extended to the quoter.

Volatility is estimated from a rolling window of the live bar closes (there's
no precomputed sigma to hand it, unlike the research simulator, which knows
its own synthetic price path's parameters). The AS time-to-horizon term uses
a constant, caller-supplied `time_horizon` every tick rather than a single
hard terminal time, since a live feed doesn't know in advance how many bars
remain -- in effect "size for unwinding this inventory over roughly this many
bars," refreshed every tick, rather than one end-of-day liquidation.
"""
from __future__ import annotations

import time

import numpy as np
import zmq
import zmq.asyncio

from quant_pod.common.config import REPLAY_ZMQ_ADDR
from quant_pod.common.events import Bar, Order, OrderType, Side
from quant_pod.common.logging import get_logger
from quant_pod.ingester.replay import END_OF_STREAM
from quant_pod.marketmaking.avellaneda_stoikov import optimal_spread, reservation_price
from quant_pod.marketmaking.simulator import Fill, MarketMakerResult, fill_probability
from quant_pod.trader.bar_buffer import BarBuffer
from quant_pod.trader.latency import LatencyTracker
from quant_pod.trader.risk_limits import RiskLimits

log = get_logger("market_maker_trader")


def _hypothetical_order(symbol: str, side: Side, quantity: float) -> Order:
    return Order(symbol=symbol, side=side, quantity=quantity, order_type=OrderType.LIMIT, strategy_id="market_maker")


async def run_market_maker_trader(
    symbol: str,
    gamma: float,
    kappa: float,
    arrival_rate: float = 1.0,  # per-BAR intensity here, not per-second like the research simulator's default
    fill_size: float = 1.0,
    time_horizon: float = 1.0,
    vol_window: int = 20,
    default_sigma: float = 1.0,
    initial_cash: float = 100_000.0,
    risk_limits: RiskLimits | None = None,
    zmq_addr: str = REPLAY_ZMQ_ADDR,
    seed: int | None = None,
) -> MarketMakerResult:
    risk_limits = risk_limits or RiskLimits()
    rng = np.random.default_rng(seed)
    quote_latency = LatencyTracker(label="market_maker/quote")
    fill_latency = LatencyTracker(label="market_maker/fill_decision")

    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(zmq_addr)
    sock.setsockopt(zmq.SUBSCRIBE, symbol.encode())

    log.info("Market maker trader started: symbol=%s gamma=%.3f kappa=%.2f A=%.2f", symbol, gamma, kappa, arrival_rate)

    bar_buffer = BarBuffer()
    position = 0.0
    cash = initial_cash
    mids: list[float] = []
    bids: list[float] = []
    asks: list[float] = []
    inventories: list[float] = []
    cashes: list[float] = []
    equities: list[float] = []
    fills: list[Fill] = []

    try:
        step = 0
        while True:
            _, payload = await sock.recv_multipart()
            tick_received_ns = time.perf_counter_ns()

            if payload == END_OF_STREAM:
                break

            bar = Bar.model_validate_json(payload)
            bar_buffer.append(bar)
            mid = bar.mid()

            # tail() BEFORE diff(), not after: diffing the full accumulated
            # history every tick would make this O(n) per tick (O(n^2) over a
            # run), exactly what BarBuffer's O(1)-amortized append exists to
            # avoid elsewhere in this hot path.
            sigma = bar_buffer.as_dataframe()["close"].tail(vol_window + 1).diff().std()
            sigma = default_sigma if not np.isfinite(sigma) or sigma == 0 else float(sigma)

            # Quote and record state using the PRE-fill position/cash carried
            # from the previous tick's fill decision -- matching
            # `simulator.run_market_maker_sim`'s indexing convention, where
            # mid[t]/bid[t]/ask[t]/inventory[t]/cash[t]/equity[t] are all the
            # state observed *before* tick t's own fill is applied, and that
            # fill only shows up in the state at t+1. Appending post-fill
            # state at the same index (an earlier version of this function
            # did) would desync `inventory`/`equity` from the `bid`/`ask`
            # they were actually quoted against, breaking the "same analysis
            # tools work on live or simulated results" claim above.
            r = reservation_price(mid, position, gamma, sigma, time_horizon)
            spread = optimal_spread(gamma, sigma, time_horizon, kappa)
            bid, ask = r - spread / 2, r + spread / 2

            if not risk_limits.check(_hypothetical_order(symbol, Side.BUY, fill_size), position, mid)[0]:
                bid = np.nan
            if not risk_limits.check(_hypothetical_order(symbol, Side.SELL, fill_size), position, mid)[0]:
                ask = np.nan
            quote_latency.record(time.perf_counter_ns() - tick_received_ns)

            mids.append(mid)
            bids.append(bid)
            asks.append(ask)
            inventories.append(position)
            cashes.append(cash)
            equities.append(cash + position * mid)

            fill_start_ns = time.perf_counter_ns()
            p_bid_hit = fill_probability(bid, mid, 1.0, kappa, arrival_rate, dt=1.0)
            p_ask_hit = fill_probability(ask, mid, -1.0, kappa, arrival_rate, dt=1.0)
            if rng.random() < p_bid_hit:
                position += fill_size
                cash -= fill_size * bid
                fills.append(Fill(step, "buy", bid, fill_size))
            if rng.random() < p_ask_hit:
                position -= fill_size
                cash += fill_size * ask
                fills.append(Fill(step, "sell", ask, fill_size))
            fill_latency.record(time.perf_counter_ns() - fill_start_ns)

            step += 1

    finally:
        sock.close()
        ctx.term()

    log.info("Market maker trader finished: %d bars processed, %d fills", step, len(fills))
    quote_latency.print_summary()
    fill_latency.print_summary()

    return MarketMakerResult(
        mid=np.array(mids), bid=np.array(bids), ask=np.array(asks),
        inventory=np.array(inventories), cash=np.array(cashes), equity=np.array(equities),
        fills=fills,
    )
