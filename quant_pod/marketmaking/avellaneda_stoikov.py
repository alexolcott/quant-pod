"""Avellaneda-Stoikov (2008) optimal market-making quotes.

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)                     [reservation price]
    spread     = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)

`gamma` is the market maker's risk aversion, `sigma` the mid-price volatility,
and `kappa` the decay rate of the assumed order-arrival intensity
lambda(delta) = A * exp(-kappa * delta) for a counterparty hitting a quote
`delta` away from mid. `A` sets the overall arrival rate but, notably, drops
out of the optimal spread entirely -- it governs how often you get filled,
not how wide you should quote.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def reservation_price(mid: float, inventory: float, gamma: float, sigma: float, time_remaining: float) -> float:
    """The price at which the maker is indifferent to holding `inventory` --
    below mid when long (eager to sell), above mid when short (eager to buy).
    """
    return mid - inventory * gamma * sigma**2 * time_remaining


def optimal_spread(gamma: float, sigma: float, time_remaining: float, kappa: float) -> float:
    """Total bid-ask spread around the reservation price. Always positive, but
    *not* monotonic in `gamma`: the inventory-risk term grows with gamma while
    the (2/gamma)*ln(1+gamma/kappa) term shrinks, so which direction the total
    moves depends on which term dominates for a given sigma/time_remaining/kappa.
    """
    if gamma <= 0:
        # lim_{gamma->0} (2/gamma) * ln(1 + gamma/kappa) = 2/kappa (risk-neutral limit).
        return 2.0 / kappa
    return gamma * sigma**2 * time_remaining + (2.0 / gamma) * np.log(1.0 + gamma / kappa)


@dataclass
class AvellanedaStoikovQuoter:
    """Holds the model's calibrated parameters; quoting itself is stateless."""

    gamma: float
    kappa: float
    sigma: float

    def quote(self, mid: float, inventory: float, time_remaining: float) -> tuple[float, float]:
        """Returns (bid, ask), centered on the reservation price -- not on `mid`."""
        r = reservation_price(mid, inventory, self.gamma, self.sigma, time_remaining)
        spread = optimal_spread(self.gamma, self.sigma, time_remaining, self.kappa)
        return r - spread / 2, r + spread / 2
