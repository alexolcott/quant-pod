"""The exogenous mid-price process a market maker reacts to.

Deliberately owns only the "true value" process, not order arrivals/fills --
whether an order crosses a quote depends on where the quote sits, which only
exists once a strategy is quoting (see `simulator.py`).
"""
from __future__ import annotations

import numpy as np


def simulate_mid_price_path(
    mid0: float,
    sigma: float,
    dt: float,
    n_steps: int,
    seed: int | None = None,
) -> np.ndarray:
    """Arithmetic Brownian motion: mid[t+1] = mid[t] + N(0, sigma^2 * dt).

    Arithmetic (not geometric) to match the price process the
    Avellaneda-Stoikov (2008) model is derived under -- so `sigma` is in price
    units, not a return vol, and the path can go negative for large enough
    sigma * sqrt(horizon) relative to mid0.

    Returns an array of length `n_steps + 1` (the starting price plus one
    value per step).
    """
    rng = np.random.default_rng(seed)
    increments = rng.normal(loc=0.0, scale=sigma * np.sqrt(dt), size=n_steps)
    return mid0 + np.concatenate(([0.0], np.cumsum(increments)))
