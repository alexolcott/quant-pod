"""Replays the Avellaneda-Stoikov quoter over real stored bar history instead
of a synthetic price path (`order_flow.py`) or a live ZeroMQ feed
(`trader.market_maker_engine`) -- the third way to drive the same quoting +
fill-probability machinery (`avellaneda_stoikov.py`, `simulator.fill_probability`).
Used by the dashboard's "real symbol" data source so different stocks/periods
can be compared without writing a script.

Volatility is estimated from a rolling window of the bars' own closes --
there's no ground-truth sigma for real data, unlike the synthetic simulator.
Like the live engine, and unlike `simulator.run_market_maker_sim` (which
decays time_remaining to zero over its one simulated horizon), this uses a
CONSTANT `time_horizon`, refreshed every bar: a real historical window can
span years, and decaying to a hard "liquidate by the last bar" horizon over
that span would be an artifact of the replay, not a property of the strategy.

Returns a `simulator.MarketMakerResult`, so every tool built for it
(`summarize`, `markout.compute_markouts`) works unmodified here too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_pod.marketmaking.avellaneda_stoikov import optimal_spread, reservation_price
from quant_pod.marketmaking.simulator import Fill, MarketMakerResult, fill_probability


def run_market_maker_on_bars(
    bars: pd.DataFrame,
    gamma: float,
    kappa: float,
    arrival_rate: float = 1.0,  # per-BAR intensity, not per-second like the synthetic simulator's default
    fill_size: float = 1.0,
    time_horizon: float = 20.0,
    vol_window: int = 20,
    default_sigma: float = 1.0,
    initial_cash: float = 0.0,
    max_inventory: float | None = None,
    seed: int | None = None,
) -> MarketMakerResult:
    """`bars` must have a `close` column (as `ingester.store.read_bars` returns).
    `max_inventory` works exactly as in `run_market_maker_sim`: once a fill
    would breach it, that side is pulled (quoted NaN, zero fill probability)
    until inventory drifts back under it.
    """
    rng = np.random.default_rng(seed)
    closes = bars["close"]
    n = len(bars)

    mids = np.empty(n)
    bids = np.empty(n)
    asks = np.empty(n)
    inventories = np.empty(n)
    cashes = np.empty(n)
    equities = np.empty(n)
    fills: list[Fill] = []

    position, cash = 0.0, initial_cash
    for step in range(n):
        mid = float(closes.iloc[step])
        window = closes.iloc[max(0, step - vol_window) : step + 1]
        sigma = window.diff().std()
        sigma = default_sigma if not np.isfinite(sigma) or sigma == 0 else float(sigma)

        # Pre-fill state recorded alongside the quote it was computed from --
        # matches `simulator.run_market_maker_sim`'s indexing convention (see
        # that module and `trader.market_maker_engine`, which fixed a bug from
        # getting this backwards: appending post-fill state desyncs
        # `inventory`/`equity` from the `bid`/`ask` they were quoted against).
        r = reservation_price(mid, position, gamma, sigma, time_horizon)
        spread = optimal_spread(gamma, sigma, time_horizon, kappa)
        bid, ask = r - spread / 2, r + spread / 2
        if max_inventory is not None:
            if position + fill_size > max_inventory:
                bid = np.nan
            if position - fill_size < -max_inventory:
                ask = np.nan

        mids[step], bids[step], asks[step] = mid, bid, ask
        inventories[step], cashes[step] = position, cash
        equities[step] = cash + position * mid

        p_bid = fill_probability(bid, mid, 1.0, kappa, arrival_rate, dt=1.0)
        p_ask = fill_probability(ask, mid, -1.0, kappa, arrival_rate, dt=1.0)
        if rng.random() < p_bid:
            position += fill_size
            cash -= fill_size * bid
            fills.append(Fill(step, "buy", bid, fill_size))
        if rng.random() < p_ask:
            position -= fill_size
            cash += fill_size * ask
            fills.append(Fill(step, "sell", ask, fill_size))

    return MarketMakerResult(mid=mids, bid=bids, ask=asks, inventory=inventories, cash=cashes, equity=equities, fills=fills)
