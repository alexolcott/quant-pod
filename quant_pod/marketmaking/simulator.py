"""Steps the Avellaneda-Stoikov quoter through a simulated mid-price path,
drawing fills directly from the model's own arrival-intensity assumption.

Fills are **not** matched through `quant_pod.trader.hotpath.OrderBook`. That
engine has no way to cancel a resting order (see its tests/README), and this
strategy replaces both of its quotes every step -- routed through a real book,
every prior quote would be left resting forever instead of being replaced.
Extending the book with cancellation would be a reasonable next step if this
grows into a model with persistent, competing resting liquidity; for now,
whether a quote gets hit is drawn straight from
lambda(delta) = A * exp(-kappa * delta), exactly the intensity the
Avellaneda-Stoikov spread formula itself assumes -- the standard way this
model family is simulated in the literature.

Terminal inventory is marked to the final mid-price, not force-liquidated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_pod.marketmaking.avellaneda_stoikov import AvellanedaStoikovQuoter
from quant_pod.marketmaking.order_flow import simulate_mid_price_path


@dataclass
class Fill:
    step: int
    side: str  # "buy" (maker's bid was hit) or "sell" (maker's ask was hit)
    price: float
    quantity: float


@dataclass
class MarketMakerResult:
    mid: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    inventory: np.ndarray
    cash: np.ndarray
    equity: np.ndarray
    fills: list[Fill]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "mid": self.mid, "bid": self.bid, "ask": self.ask,
                "inventory": self.inventory, "cash": self.cash, "equity": self.equity,
            }
        )


def fill_probability(quote_price: float, mid: float, side_sign: float, kappa: float, arrival_rate: float, dt: float) -> float:
    """P(a quote `dt` time units away from the next arrival check gets hit),
    from the same lambda(delta) = A * exp(-kappa * delta) intensity the AS
    spread formula assumes. side_sign is +1 for a bid (distance = mid - quote)
    and -1 for an ask (distance = quote - mid). NaN quote_price means the side
    isn't quoting (see `max_inventory`) and so can never be hit.

    Public (not just used by `run_market_maker_sim`): `trader.market_maker_engine`
    reuses this directly so the live loop draws fills from the exact same
    model the research simulator does.
    """
    if np.isnan(quote_price):
        return 0.0
    delta = max(side_sign * (mid - quote_price), 0.0)
    return -np.expm1(-arrival_rate * np.exp(-kappa * delta) * dt)


def run_market_maker_sim(
    quoter: AvellanedaStoikovQuoter,
    mid0: float = 100.0,
    sigma: float = 0.3,
    dt: float = 1.0 / 23_400,  # one simulated second of a 6.5h trading day
    horizon: float = 1.0,  # one trading day, in the same units as time_remaining
    n_steps: int | None = None,
    arrival_rate: float = 140.0,  # "A" in lambda(delta) = A * exp(-kappa * delta)
    fill_size: float = 1.0,
    initial_cash: float = 0.0,
    max_inventory: float | None = None,
    seed: int | None = None,
) -> MarketMakerResult:
    """`max_inventory` is a hard position limit layered on top of the AS quotes
    (which only discourage runaway inventory via skew, never forbid it): once a
    fill would push |inventory| past the limit, that side is pulled (quoted as
    NaN, probability of a fill on it is 0) until inventory drifts back under it.
    """
    if n_steps is None:
        n_steps = int(round(horizon / dt))

    mid_path = simulate_mid_price_path(mid0, sigma, dt, n_steps, seed=seed)
    rng = np.random.default_rng(None if seed is None else seed + 1)

    bid = np.empty(n_steps + 1)
    ask = np.empty(n_steps + 1)
    inventory = np.zeros(n_steps + 1)
    cash = np.empty(n_steps + 1)
    equity = np.empty(n_steps + 1)
    cash[0] = initial_cash
    equity[0] = initial_cash + inventory[0] * mid_path[0]
    fills: list[Fill] = []

    for step in range(n_steps + 1):
        mid = mid_path[step]
        time_remaining = max(horizon - step * dt, 0.0)
        q = inventory[step]
        b, a = quoter.quote(mid, q, time_remaining)
        if max_inventory is not None:
            if q + fill_size > max_inventory:
                b = np.nan
            if q - fill_size < -max_inventory:
                a = np.nan
        bid[step], ask[step] = b, a

        if step == n_steps:
            break

        p_bid_hit = fill_probability(b, mid, 1.0, quoter.kappa, arrival_rate, dt)
        p_ask_hit = fill_probability(a, mid, -1.0, quoter.kappa, arrival_rate, dt)

        next_inventory, next_cash = q, cash[step]
        if rng.random() < p_bid_hit:
            next_inventory += fill_size
            next_cash -= fill_size * b
            fills.append(Fill(step, "buy", b, fill_size))
        if rng.random() < p_ask_hit:
            next_inventory -= fill_size
            next_cash += fill_size * a
            fills.append(Fill(step, "sell", a, fill_size))

        inventory[step + 1] = next_inventory
        cash[step + 1] = next_cash
        equity[step + 1] = next_cash + next_inventory * mid_path[step + 1]

    return MarketMakerResult(mid=mid_path, bid=bid, ask=ask, inventory=inventory, cash=cash, equity=equity, fills=fills)


def summarize(result: MarketMakerResult) -> dict:
    spread = result.ask - result.bid
    return {
        "total_pnl": float(result.equity[-1] - result.equity[0]),
        "final_inventory": float(result.inventory[-1]),
        "max_abs_inventory": float(np.max(np.abs(result.inventory))),
        "inventory_std": float(np.std(result.inventory)),
        "num_fills": len(result.fills),
        "avg_spread_bps": float(np.nanmean(spread / result.mid) * 10_000),
        "pct_steps_quoting_bid": float(np.mean(~np.isnan(result.bid))),
        "pct_steps_quoting_ask": float(np.mean(~np.isnan(result.ask))),
    }
